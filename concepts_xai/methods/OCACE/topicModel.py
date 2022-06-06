import concepts_xai.evaluation.metrics.completeness as completeness
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import scipy

'''
Re-implementation of the "On Completeness-aware Concept-Based Explanations in
Deep Neural Networks":

1) See https://arxiv.org/abs/1910.07969 for the original paper

2)  See https://github.com/chihkuanyeh/concept_exp for the original paper
    implementation

'''


class TopicModel(tf.keras.Model):
    """Base class of a topic model."""

    def __init__(
        self,
        concepts_to_labels_model,
        n_channels,
        n_concepts,
        g_model=None,
        threshold=0.5,
        loss_fn=tf.keras.losses.sparse_categorical_crossentropy,
        top_k=32,
        lambda1=0.1,
        lambda2=0.1,
        seed=None,
        eps=1e-5,
        data_format="channels_last",
        allow_gradient_flow_to_c2l=False,
        acc_metric=None,
        initial_topic_vector=None,
        **kwargs,
    ):
        super(TopicModel, self).__init__(**kwargs)

        initializer = tf.keras.initializers.RandomUniform(
            minval=-0.5,
            maxval=0.5,
            seed=seed,
        )

        # Initialize our topic vector tensor which we will learn
        # as part of our training
        if initial_topic_vector is not None:
            self.topic_vector = self.add_weight(
                name="topic_vector",
                shape=(n_channels, n_concepts),
                dtype=tf.float32,
                initializer=lambda *args, **kwargs: initial_topic_vector,
                trainable=True,
            )
        else:
            self.topic_vector = self.add_weight(
                name="topic_vector",
                shape=(n_channels, n_concepts),
                dtype=tf.float32,
                initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.5,
                    maxval=0.5,
                    seed=seed,
                ),
                trainable=True,
            )

        # Initialize the g model which will be in charge of reconstructing
        # the model latent activations from the concept scores alone
        self.g_model = g_model
        if self.g_model is None:
            self.g_model = completeness._get_default_model(
                num_concepts=n_concepts,
                num_hidden_acts=n_channels,
            )

        # Set the concept-to-label predictor model
        self.concepts_to_labels_model = concepts_to_labels_model

        # Set remaining model hyperparams
        self.eps = eps
        self.threshold = threshold
        self.n_concepts = n_concepts
        self.loss_fn = loss_fn
        self.top_k = top_k
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.n_channels = n_channels
        self.allow_gradient_flow_to_c2l = allow_gradient_flow_to_c2l
        assert data_format in ["channels_last", "channels_first"], (
            f'Expected data format to be either "channels_last" or '
            f'"channels_first" however we obtained "{data_format}".'
        )
        if data_format == "channels_last":
            self._channel_axis = -1
        else:
            raise ValueError(
                'Currently we only support "channels_last" data_format'
            )

        self.metric_names = ["loss", "mean_sim", "accuracy"]
        self.metrics_dict = {
            name: tf.keras.metrics.Mean(name=name)
            for name in self.metric_names
        }
        self._acc_metric = (
            acc_metric or tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
        )

    @property
    def metrics(self):
        return [self.metrics_dict[name] for name in self.metric_names]

    def update_metrics(self, losses):
        for (loss_name, loss) in losses:
            self.metrics_dict[loss_name].update_state(loss)

    def concept_scores(self, x, compute_reg_terms=False):
        # Compute the concept representation by first normalizing across the
        # channel dimension both the concept vectors and the input
        assert x.shape[self._channel_axis] == self.n_channels, (
            f'Expected input to have {self.n_channels} elements in its '
            f'channels axis (defined as axis {self._channel_axis}). '
            f'Instead, we found the input to have shape {x.shape}.'
        )

        x_norm = tf.math.l2_normalize(x, axis=self._channel_axis)
        topic_vector_norm = tf.math.l2_normalize(self.topic_vector, axis=0)
        # Compute the concept probability scores
        topic_prob = K.dot(x, topic_vector_norm)
        topic_prob_norm = K.dot(x_norm, topic_vector_norm)

        # Threshold them if they are below the given threshold value
        if self.threshold is not None:
            topic_prob = topic_prob * tf.cast(
                (topic_prob_norm > self.threshold),
                tf.float32,
            )
        topic_prob_sum = tf.reduce_sum(
            topic_prob,
            axis=self._channel_axis,
            keepdims=True,
        )
        # And normalize the actual scores
        topic_prob = topic_prob / (topic_prob_sum + self.eps)
        if not compute_reg_terms:
            return topic_prob

        # Compute the regularization loss terms
        reshaped_topic_probs = tf.transpose(
            tf.reshape(topic_prob_norm, (-1, self.n_concepts))
        )
        reg_loss_1 = tf.reduce_mean(
            tf.nn.top_k(
                reshaped_topic_probs,
                k=tf.math.minimum(
                    self.top_k,
                    tf.shape(reshaped_topic_probs)[-1]
                ),
                sorted=True,
            ).values
        )

        reg_loss_2 = tf.reduce_mean(
            K.dot(tf.transpose(topic_vector_norm), topic_vector_norm) -
            tf.eye(self.n_concepts)
        )
        return topic_prob, reg_loss_1, reg_loss_2

    def _compute_loss(self, x, y_true, training):
        # First, compute the concept scores for the given samples
        scores, reg_loss_1, reg_loss_2 = self.concept_scores(
            x,
            compute_reg_terms=True
        )

        # Then predict the labels after reconstructing the activations
        # from the scores via the g model
        y_pred = self.concepts_to_labels_model(
            self.g_model(scores),
            training=training,
        )

        # Compute the task loss
        log_prob_loss = tf.reduce_mean(self.loss_fn(y_true, y_pred))

        # And include them into the total loss
        total_loss = (
            log_prob_loss -
            self.lambda1 * reg_loss_1 +
            self.lambda2 * reg_loss_2
        )

        # Compute the accuracy metric to track
        self._acc_metric.update_state(y_true, y_pred)
        return total_loss, reg_loss_1, self._acc_metric.result()

    def train_step(self, inputs):
        x, y = inputs

        # We first need to make sure that we set our concepts to labels model
        # so that we do not optimize over its parameters
        prev_trainable = self.concepts_to_labels_model.trainable
        if not self.allow_gradient_flow_to_c2l:
            self.concepts_to_labels_model.trainable = False
        with tf.GradientTape() as tape:
            loss, mean_sim, acc = self._compute_loss(
                x,
                y,
                # Only train the decoder if requested by the user
                training=self.allow_gradient_flow_to_c2l,
            )

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )
        self.update_metrics([
            ("loss", loss),
            ("mean_sim", mean_sim),
            ("accuracy", acc),
        ])

        # And recover the previous step of the concept to labels model
        self.concepts_to_labels_model.trainable = prev_trainable

        return {
            name: self.metrics_dict[name].result()
            for name in self.metric_names
        }

    def test_step(self, inputs):
        x, y = inputs
        loss, mean_sim, acc = self._compute_loss(x, y, training=False)
        self.update_metrics([
            ("loss", loss),
            ("mean_sim", mean_sim),
            ("accuracy", acc)
        ])

        return {
            name: self.metrics_dict[name].result()
            for name in self.metric_names
        }

    def call(self, x, **kwargs):
        concept_scores = self.concept_scores(x)
        predicted_labels = self.concepts_to_labels_model(
            self.g_model(concept_scores),
            training=False,
        )
        return predicted_labels, concept_scores

