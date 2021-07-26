import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
import numpy as np
import os

from methods.CME.CtlModel import CtLModel
from utils.utils import tf_data_split

class ConceptBottleneckModel(object):

    def __init__(self, model, layer_id, n_classes, n_c_vals_list, save_path, multi_task=False,
                        end_to_end=False, c_epochs=10, batch_size=64,
                 overwrite=True,
                 c_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), lambda_param=0.1):
        '''
        :param model:          Keras model to build Bottleneck model from
        :param layer_id:       layer_id of the model layer to use as the output activations to feed into the bottleneck
        :param n_c_vals_list:  list of length (n_concepts), where n_c_vals_list[i] denotes the number of
                               concept values of concept_i
        :param multi_task: whether to setup a separate softmax output for each concept (i.e. a multi-task setup),
                           or whether to simply include all concept values as a sigmoid layer (i.e. a multi-label setup)
        :param end_to_end: whether to create an end-to-end keras model (with the concept model being part of the model),
                           or keep as separate models
        :param c_epochs:    number of epochs to use during training of the concept extractor
        :param c_extr_path: path to the concept extractor model
        '''

        self.layer_id           = layer_id
        self.n_classes          = n_classes
        self.n_c_vals_list      = n_c_vals_list
        self.n_concepts         = len(self.n_c_vals_list)
        self.multi_task         = multi_task
        self.end_to_end         = end_to_end
        self.save_path          = save_path
        self.overwrite          = overwrite
        self.lambda_param       = lambda_param
        if save_path.split(".")[-1] == "h5":
            self.save_path = save_path[:-2] + "tf"

        self.n_epochs           = c_epochs
        self.batch_size         = batch_size
        # This is needed during binarisation to store offsets of values
        self.offsets  = tf.convert_to_tensor(np.array([sum(self.n_c_vals_list[:i]) for i in range(len(n_c_vals_list))]))
        self.n_total_c_vals = sum(self.n_c_vals_list)

        self.c_optimizer = c_optimizer

        self.concept_extractor  = self._build_concept_extractor(model)
        self.label_predictor    = self._build_label_predictor()
        self.cb_model           = self._build_cb_model()


    def _build_concept_extractor(self, model):

        # Retrieve "base" of model
        base = tf.keras.Model(inputs=model.inputs, outputs=model.layers[self.layer_id].output)
        x = base.output

        if self.multi_task:
            # Create 1 softmax output per concept
            c_outputs = []
            for i, n_c_vals in enumerate(self.n_c_vals_list):
                output_c = Dense(n_c_vals, activation='softmax', name='c_' + str(i))(x)
                c_outputs.append(output_c)
            concept_model = tf.keras.Model(inputs=base.input, outputs=c_outputs)

            # Create 1 metric and 1 loss per output
            losses  = {}
            metrics = {}
            for i in range(len(self.n_c_vals_list)):
                losses['c_' + str(i)]  = 'sparse_categorical_crossentropy'
                metrics['c_' + str(i)] = 'accuracy'

            self.concept_extractor_losses = losses
            self.concept_extractor_metrics = metrics

            if not self.end_to_end:
                concept_model.compile(optimizer=self.c_optimizer, loss=losses, metrics=metrics)

        else:
            # Create 1 sigmoid layer for all values of all concepts
            total_c_vals = sum(self.n_c_vals_list)

            c_outputs       = Dense(total_c_vals, activation='sigmoid', name='bottleneck')(x)
            concept_model   = tf.keras.Model(inputs=base.input, outputs=[c_outputs])

            self.concept_extractor_losses =  ["binary_crossentropy"]
            self.concept_extractor_metrics = ["accuracy"]

            if not self.end_to_end:
                concept_model.compile(optimizer=self.c_optimizer,
                                      loss=self.concept_extractor_losses[0],
                                      metrics=self.concept_extractor_metrics[0])

        return concept_model


    def _build_label_predictor(self):

        if self.end_to_end:

            if self.multi_task:
                raise NotImplementedError("Currently, label predictor not implemented...")

            else:
                # TODO: currently, create a 2-layer MLP
                # TODO: in the future, consider passing in an arbitrary model as your LP
                input =  Input(self.concept_extractor.output.shape[1:])
                output = Dense(2*self.n_classes, activation='relu')(input)
                output = Dense(self.n_classes, activation='softmax', name='lp')(output)
                label_predictor = tf.keras.Model(inputs=input, outputs=[output])

            self.label_predictor_loss   =  'sparse_categorical_crossentropy'
            self.label_predictor_metric =  "accuracy"

            if not self.end_to_end:
                label_predictor.compile(optimizer='Adam',
                                        loss=self.label_predictor_loss,
                                        metrics=self.label_predictor_metric)

        else:
            # Use a Decision Tree task label predictor
            params = {"method"     : "DT",
                      "n_concepts" : len(self.n_c_vals_list),
                      "n_classes"  : self.n_classes}

            label_predictor = CtLModel(**params)

        return label_predictor


    def _build_cb_model(self):

        if self.end_to_end:

            # Concatenate concept extractor with the label predictor
            input_shape     = self.concept_extractor.input_shape[1:]
            input_layer     = Input(input_shape)
            cp_output       = self.concept_extractor(input_layer)
            lp_output       = self.label_predictor(cp_output)

            if self.multi_task:
                merged_outputs  = cp_output + [lp_output]
            else:
                merged_outputs  = [cp_output, lp_output]

            # Define the CB model
            cb_model    = tf.keras.Model(input_layer, merged_outputs)

            # Merge bottleneck and output losses and metrics
            cp_names     = [cb_model.layers[-2].name]
            lp_name      = cb_model.layers[-1].name
            all_losses   = {cp_names[i] : self.concept_extractor_losses[i] for i in range(len(cp_names))}
            all_losses   = {**all_losses, **{lp_name : self.label_predictor_loss}}
            all_metrics  = {cp_names[i] : self.concept_extractor_metrics[i] for i in range(len(cp_names))}
            all_metrics  = {**all_metrics, **{lp_name : self.label_predictor_metric}}
            loss_weights = {cp_name : 1.0 for cp_name in cp_names}
            loss_weights = {**loss_weights, **{lp_name : self.lambda_param}}

            cb_model.compile(optimizer=self.c_optimizer, loss=all_losses,
                             metrics=all_metrics, loss_weights=loss_weights)

        else:
            cb_model = None

        return cb_model


    def convert_to_multioutput(self, c):
        multi_output = tuple([c[i:i + 1] for i in range(self.n_concepts)])
        return multi_output


    def binarize_map(self, c):
        z = c + self.offsets
        vals = tf.one_hot(indices=z, depth=self.n_total_c_vals, on_value=1, off_value=0)
        vals = tf.reduce_sum(vals, axis=0)
        return vals


    def train(self, data_gen_l):
        '''
        :param data_gen_l: tf.dataset generator, returning tuples (x, c, y) for image, concept, and label data
        '''

        # TODO: currently, loads concept extractor only
        # Load concept extractor, if the model already exists
        if (not self.overwrite) and os.path.exists(self.save_path):
            self.concept_extractor.load_weights(self.save_path)
            return

        # TODO: will implement end-to-end learning
        if self.end_to_end:
            if self.multi_task:
                raise NotImplementedError("Multi-task end-to-end learning not implemented yet...")

            else:
                ds_l = data_gen_l.map(lambda x, c, y: (x, tuple([self.binarize_map(c), y])))

                # Train concept extractor using a validation split
                ds_l_train, ds_l_val = tf_data_split(ds_l, test_size=0.2)
                callbacks = []

                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.save_path, verbose=True,
                                                                 save_best_only=True, monitor='val_loss',
                                                                 mode='auto', save_freq='epoch')
                callbacks.append(cp_callback)

                self.cb_model.fit(ds_l_train.batch(self.batch_size), epochs=self.n_epochs,
                                           validation_data=ds_l_val.batch(self.batch_size),
                                           callbacks=callbacks)

                self.concept_extractor = self.cb_model.layers[-2]

        else:
            if self.multi_task:
                # Filter out y-label data via mapping, and convert c to a multi-output format
                ds_l = data_gen_l.map(lambda x, c, y: (x, self.convert_to_multioutput(c)))
            else:
                # Binarize concept values, in case of the multi-label setup
                ds_l = data_gen_l.map(lambda x, c, y: (x, self.binarize_map(c)))

            # Train concept extractor using a validation split
            ds_l_train, ds_l_val = tf_data_split(ds_l, test_size=0.2)
            callbacks = []
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.save_path, verbose=True,
                                                             save_best_only=True, monitor='val_loss',
                                                             mode='auto', save_freq='epoch')
            callbacks.append(cp_callback)

            self.concept_extractor.fit(ds_l_train.batch(self.batch_size), epochs=self.n_epochs,
                                       validation_data=ds_l_val.batch(self.batch_size),
                                       callbacks=callbacks)


    def _debinarize(self, predicted_c):
        offsets = list(self.offsets.numpy())
        offsets.append(sum(self.n_c_vals_list))
        predicted_debinarized = []
        for i in range(len(offsets) - 1):
            start, stop = offsets[i], offsets[i + 1]
            arr = predicted_c[:, start:stop]
            arr = np.argmax(arr, axis=1)
            predicted_debinarized.append(arr)
        predicted_c = np.stack(predicted_debinarized, axis=-1)

        return predicted_c


    def predict(self, data_gen_u, logits=False, y_labels=False):
        '''
        :param data_gen_u: tf.dataset generator, returning tuples (x) of image data
        '''

        # TODO: implement end-to-end prediction
        if self.end_to_end:

            if self.multi_task:
                raise NotImplementedError()
            else:
                predicted_c, predicted_y =  self.cb_model.predict(data_gen_u.batch(self.batch_size))

                if not logits:
                    predicted_c = self._debinarize(predicted_c)

                if y_labels:
                    return predicted_c, predicted_y
                else:
                    return predicted_c

        else:
            if self.multi_task:
                predicted = self.concept_extractor.predict(data_gen_u.batch(self.batch_size))
                # Compute class values for every concept and concatenate the results
                predicted = [np.expand_dims(np.argmax(predicted[i], axis=-1), axis=-1) for i in range(len(predicted))]
                predicted = np.hstack(predicted)
                return predicted

            else:
                predicted = self.concept_extractor.predict(data_gen_u.batch(self.batch_size))
                predicted = self._debinarize(predicted)
                predicted = (predicted > 0.5).astype(np.int32)

                return predicted

