import numpy as np
import os
import tensorflow as tf

from collections import defaultdict
from concepts_xai.methods.CME.CtlModel import CtLModel
from concepts_xai.utils.utils import tf_data_split
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.python.keras.engine import data_adapter


def produce_bottleneck(model, layer_idx):
    """
    Partitions a Keras model into two disjoint computational graphs: (1) an
    encoder that maps from inputs to activations from layer with index
    `layer_idx` and (2) a decoder model that maps activations from layer
    with index `layer_idx` to the output of the given model.

    For this operation to be successful, the layer at index `layer_idx`
    must be a bottleneck of the input model (i.e., there may not be any
    connections from the layers preceding the bottleneck layer with those
    topologically after the bottleneck layer).

    :param tf.keras.Model model: A model which we will split into two disjoint
        submodels.
    :param int layer_idx:  A valid layer index in the given model. It must
        be a valid index in the array of layers represented by model.layers.

    :return Tuple[tf.keras.Model, tf.keras.Model]: a tuple of models
        (encoder, decoder) representing the input to bottleneck model and the
        bottleneck to output model, respectively.
    """

    # Let's start by making sure we get a full understanding of the input
    # model's topology
    in_edges = defaultdict(set)
    out_edges = defaultdict(set)
    name_to_layer = {}
    for src in model.layers:
        name_to_layer[src.name] = src
        for dst_node in src._outbound_nodes:
            in_edges[dst_node.layer.name].add(src.name)
            out_edges[src.name].add(dst_node.layer.name)

    # Now let's find the layer we will use as our bottleneck layer
    if len(model.layers) <= layer_idx:
        raise ValueError(
            f"Requested to use layer with index {layer_idx} as the bottleneck "
            f"layer however given model '{model.name}' has only "
            f"{len(model.layers)} indexable layers."
        )
    bottleneck_layer = model.layers[layer_idx]

    # Once we have the bottleneck, let's look at all the nodes that precede it
    # and follow it in the computational graph defined by `model`. For this to
    # be considered a valid bottleneck, the set of nodes after the bottleneck
    # layer must be disjoint from the set of nodes preceding the bottleneck
    # layer (i.e., there must be no edges from the subgraph preceding the
    # bottleneck layer into the subgraph defined by nodes that are topologically
    # after the bottleneck layer).
    preceding_nodes = set()
    frontier = [bottleneck_layer.name]
    while frontier:
        next_node = frontier.pop()
        for src_name in in_edges[next_node]:
            if src_name in preceding_nodes:
                # Then we have already dealt with it
                continue
            preceding_nodes.add(src_name)
            frontier.append(src_name)

    # And walk the graph to compute the nodes after the bottleneck layer
    posterior_nodes = set()
    frontier = [bottleneck_layer.name]
    while frontier:
        next_node = frontier.pop()
        for dst_name in out_edges[next_node]:
            if dst_name in posterior_nodes:
                # Then we have already dealt with it
                continue
            posterior_nodes.add(dst_name)
            frontier.append(dst_name)

    if (posterior_nodes & preceding_nodes):
        raise ValueError(
            f"Requested bottleneck layer {bottleneck_layer.name} (index "
            f"{layer_idx}) does not partition the computational graph of "
            f"provided the model '{model.name}' into two disjoint subgraphs ("
            f"i.e., there is a connection between layers preceeding the "
            f"bottleneck and layers after the bottleneck layer)."
        )

    # We can now compute the size of the actual bottleneck
    if isinstance(bottleneck_layer.output, list):
        raise ValueError(
            f"Currently we do not support as a bottleneck layer a layer that "
            f"has more than one output. Requested bottleneck layer "
            f"{bottleneck_layer.name} (at index {layer_idx}) has "
            f"{len(bottleneck_layer.output)} outputs."
        )
    # Else let's check the number of concepts we are expecting vs the number
    # of entries
    num_concepts = bottleneck_layer.output.shape[-1]

    # With this, building the encoder is now trivial
    encoder = tf.keras.Model(
        inputs=model.inputs,
        outputs=bottleneck_layer.output,
    )

    decoder_input = tf.keras.layers.Input(num_concepts)
    decoder_outputs = []
    name_to_layer[bottleneck_layer.name] = decoder_input
    for layer in model.layers:
        if (layer.name == bottleneck_layer.name) or (
            layer.name not in posterior_nodes
        ):
            continue
        # Otherwise let's make sure we feed it with the input corresponding
        # to the new computational graph we are constructing from the bottleneck
        # layer (NOTE: this works as we are iterating over layers in topological
        # order)
        input_layers = []
        for input_name in in_edges[layer.name]:
            input_layers.append(name_to_layer[input_name])
        if len(input_layers) == 1:
            input_layers = input_layers[0]
        # Generate the new node
        new_node = layer(input_layers)
        name_to_layer[layer.name] = new_node

        # And add it to the output if this was an original output
        if layer.name in model.output_names:
            decoder_outputs.append(new_node)
    decoder = tf.keras.Model(
        inputs=decoder_input,
        outputs=decoder_outputs
    )

    return encoder, decoder


class JointConceptBottleneckModel(tf.keras.Model):
    """
    Main class for implementing a Joint Concept Bottleneck Model with the
    given encoder mapping input features to concepts and the given
    decoder which maps concept encodings to labels.
    This class encapsulates the joint training process of a CBM while allowing
    an arbitrary encoder/decoder model to be used in its construction.

    Note that it generalizes the original CBM by Koh et al. by allowing the
    encoder to produce non-binary concepts rather than assuming all concepts
    are binary in nature.
    """

    def __init__(
        self,
        encoder,
        decoder,
        task_loss,
        alpha=0.01,
        metrics=None,
        **kwargs
    ):
        """
        Constructs a new trainable joint CBM which can be then trained,
        optimized and/or used for prediction as any other Keras model can.

        When using this model for prediction, it will return a tuple
        (labels, concepts) indicating the predicted label probabilities as
        well as the predicted concepts probabilities.

        :param tf.keras.Model encoder: A valid keras model that maps input
            features to a set of concepts. If the output of this model is
            a single vector, then every entry  of this vector is assumed to be
            one binary concept. Otherwise, if the output of the encoder is a
            list of vectors, then we assume that each vector represents a
            probability distribution over different classes for each concept (
            i.e., we assume one concept per vector).
        :param tf.keras.Model decoder: A valid keras model mapping a concept
            vector to a set of task-specific labels. We assume that if the
            encoder output as list of concepts, then the input to this model
            is the concatenation of all the output vectors of the encoder in
            the same order as produced by the encoder.
        :param tf.keras.losses.Loss task_loss: The loss to be used for the
            specific task labels.
        :param float alpha: A parameter indicating how much weight should one
            assign the loss coming from training the bottleneck. If 0, then
            there is no learning enforced in the bottleneck.
        :param List[tf.keras.metrics.Metric]  metrics: A list of possible
            metrics of interest which one may want to monitor during training.
        :param Dict[Any, Any] kwargs: Keras Layer specific kwargs to be passed
            to the parent constructor.
        """
        super(JointConceptBottleneckModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(
            name="total_loss"
        )
        self.concept_loss_tracker = tf.keras.metrics.Mean(
            name="concept_loss"
        )
        self.task_loss_tracker = tf.keras.metrics.Mean(
            name="task_loss"
        )
        self.concept_accuracy_tracker = tf.keras.metrics.Mean(
            name="concept_accuracy"
        )
        self._acc_metric = \
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
        self._bin_acc_metric = \
            tf.keras.metrics.BinaryAccuracy()
        self.alpha = alpha
        self.task_loss = task_loss
        self.extra_metrics = metrics or []

        # dummy call to build the model
        self(tf.zeros(list(map(
            lambda x: 1 if x is None else x,
            self.encoder.input_shape
        ))))

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.concept_loss_tracker,
            self.task_loss_tracker,
            self.concept_accuracy_tracker,
        ] + self.extra_metrics

    @tf.function
    def predict_from_concepts(self, concepts):
        """
        Given a set of concepts (e.g., coming from an intervention), this
        function returns the predicted labels for those concepts.

        :param np.ndarray concepts: A matrix of concepts predictions from which
            we wish to obtain classes for. It shape must be
            (n_samples, n_concepts) if concepts are binary. Otherwise, it should
            have shape (n_samples, <input entries to decoder model>).

        :returns np.ndarray: Label probability predictions for the given set
            of concepts.
        """
        if isinstance(concepts, list):
            if len(concepts) > 1:
                compacted_vector = tf.keras.layers.Concatenate(axis=-1)(
                    concepts
                )
            else:
                compacted_vector = concepts[0]
        else:
            compacted_vector = concepts
        return self.decoder(compacted_vector)

    def call(self, inputs):
        # We will use the log of the variance rather than the actual variance
        # for stability purposes
        outputs, concepts, _ = self._call_fn(inputs)
        return outputs, concepts

    @tf.function
    def _call_fn(self, inputs, **kwargs):
        # This method is separate from the call method above as it allows one
        # to overwrite this class and include an extra set of losses (returned
        # as the third element in the tuple) which could, for example, include
        # some decorrelation regularization term between concept predictions.
        concepts = self.encoder(inputs, **kwargs)
        return self.predict_from_concepts(concepts), concepts, []

    @tf.function
    def _compute_losses(
        self,
        predicted_labels,
        predicted_concepts,
        true_labels,
        true_concepts,
    ):
        """
        Helper function for computing all the losses we require for training
        our joint CBM.
        """

        # Updates stateful loss metrics.
        task_loss = self.task_loss(true_labels, predicted_labels)
        concept_loss = 0.0
        concept_accuracy = 0.0
        # If generating model does not produce a list of outputs, then we will
        # assume all concepts are binary
        if isinstance(predicted_concepts, list):
            for i, predicted_vec in enumerate(predicted_concepts):
                true_vec = true_concepts[:, i]
                if (len(predicted_vec.shape) == 1) or (
                    predicted_vec.shape[-1] == 1
                ):
                    # Then use binary loss here
                    concept_loss += tf.keras.losses.BinaryCrossentropy(
                        from_logits=False,
                    )(
                        true_vec,
                        predicted_vec,
                    )
                    if len(predicted_vec.shape) == 2:
                        # Then, let's remove the degenerate dimension
                        predicted_vec = tf.squeeze(predicted_vec, axis=-1)
                    concept_accuracy += self._bin_acc_metric(
                        true_vec,
                        predicted_vec,
                    )
                else:
                    # Otherwise use normal cross entropy
                    concept_loss += \
                        tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=False,
                        )(
                            true_vec,
                            predicted_vec,
                        )
                    concept_accuracy += self._acc_metric(
                        true_vec,
                        predicted_vec,
                    )

            # And time to normalize over all the different heads
            concept_loss = concept_loss / len(predicted_concepts)
            concept_accuracy = concept_accuracy / len(predicted_concepts)
        else:
            # Then use binary loss here as we are given a single vector and we
            # will assume in that instance they all represent independent
            # binary concepts
            concept_loss += tf.keras.losses.BinaryCrossentropy(
                from_logits=False,
            )(
                true_concepts,
                predicted_concepts,
            )
            normalize_acc = False
            concept_accuracy = self._bin_acc_metric(
                true_concepts,
                predicted_concepts,
            )
        return task_loss, concept_loss, concept_accuracy

    def test_step(self, data):
        """
        Overwrite function for the Keras model indicating how a test step
        will operate.

        :param Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]] data: The input
            training data is expected to be provided in the form
            (input_features, (true_labels, true_concepts)).
        """
        # Massage the data
        data = data_adapter.expand_1d(data)
        input_features, (true_labels, true_concepts), sample_weight = \
            data_adapter.unpack_x_y_sample_weight(data)

        # Obtain a prediction of labels and concepts
        predicted_labels, predicted_concepts, extra_losses = self._call_fn(
            input_features,
            training=False,
        )
        # Compute the actual losses
        task_loss, concept_loss, concept_accuracy = self._compute_losses(
            predicted_labels=predicted_labels,
            predicted_concepts=predicted_concepts,
            true_labels=true_labels,
            true_concepts=true_concepts,
        )

        # Accumulate both the concept and task-specific loss into a single value
        total_loss = (
            task_loss +
            self.alpha * concept_loss
        )
        for extra_loss in extra_losses:
            total_loss += extra_loss
        result = {
            self.concept_accuracy_tracker.name: concept_accuracy,
            self.concept_loss_tracker.name: concept_loss,
            self.task_loss_tracker.name: task_loss,
            self.total_loss_tracker.name: total_loss,
        }
        for metric in self.extra_metrics:
            result[metric.name] = metric(
                true_labels,
                predicted_labels,
                sample_weight,
            )
        return result

    @tf.function
    def train_step(self, data):
        """
        Overwrite function for the Keras model indicating how a train step
        will operate.

        :param Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]] data: The input
            training data is expected to be provided in the form
            (input_features, (true_labels, true_concepts)).
        """
        # Massage the data
        data = data_adapter.expand_1d(data)
        input_features, (true_labels, true_concepts), sample_weight = \
            data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            # Obtain a prediction of labels and concepts
            predicted_labels, predicted_concepts, extra_losses = self._call_fn(
                input_features
            )
            # Compute the actual losses
            task_loss, concept_loss, concept_accuracy = self._compute_losses(
                predicted_labels=predicted_labels,
                predicted_concepts=predicted_concepts,
                true_labels=true_labels,
                true_concepts=true_concepts,
            )
            # Accumulate both the concept and task-specific loss into a single
            # value
            total_loss = (
                task_loss +
                self.alpha * concept_loss
            )
            # And include any extra losses coming from this process
            for extra_loss in extra_losses:
                total_loss += extra_loss

        num_concepts = (
            len(predicted_concepts) if isinstance(predicted_concepts, list) else
            predicted_concepts.shape[-1]
        )
        concept_accuracy = concept_accuracy / num_concepts
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss, sample_weight)
        self.task_loss_tracker.update_state(task_loss, sample_weight)
        self.concept_loss_tracker.update_state(concept_loss, sample_weight)
        self.concept_accuracy_tracker.update_state(
            concept_accuracy,
            sample_weight,
        )
        for metric in self.extra_metrics:
            metric.update_state(true_labels, predicted_labels, sample_weight)
        return {
            metric.name: metric.result()
            for metric in self.metrics
        }
