from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

'''
Implementation of the Semi-Supervised Concept Classifier (SSCClassifier)

Contains implementations of the following:
    1) An abstract SSCClassifier class, specifying the requirements of a generic SSCC concept classifier
    2) Implementations of the SSCClassifier, serving as wrappers of various concept-based methods
        defined in the .methods folder
'''

class SSCClassifier(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        '''
        Initialise the SSCClassifier wrapper
        :param kwargs: any extra arguments needed to initialise the concept predictor model
        '''
        pass

    @abstractmethod
    def fit(self, data_gen_l, data_gen_u):
        '''
        Method for training the underlying SSCC model
        :param data_gen_l: Tensorflow Dataset, returning triples of points (x_data, c_data, y_data)
                            for input, concept, and label data, respectively
        :param data_gen_u: Tensorflow Dataset, returning tuples of points (x_data, y_data)
                            In the fully-unsupervised case (where y-labels are not available as well),
                            y_data should be set to -1
        :return: Trains the underlying SSCC concept labeler
        '''
        pass


    @abstractmethod
    def predict(self, data_gen_u):
        '''
        Method for predicting the concept labels from the input data
        :param data_gen_u: Tensorflow Dataset, returning tuples of points (x_data, y_data)
                            In the fully-unsupervised case (where y-labels are not available as well),
                            y_data should be set to -1
        :return: numpy array of predicted samples, corresponding to the input points
        '''
        pass



from methods.CME.ItCModel import ItCModel

class SSCC_CME(SSCClassifier):
    '''
    CME Concept Predictor Wrapper
    '''

    def __init__(self, **kwargs):
        '''
        :param kwargs: Required arguments:
                            - "model"       : the underlying trained model to extract concepts from
                            - "n_concepts"  : the number of concepts
        '''
        super().__init__(**kwargs)
        # Extract model parameter
        model = kwargs["base_model"]
        # Copy all the other parameters into the params dict.
        params = {key:val for (key, val) in kwargs.items() if key != "base_model"}
        # Create the underlying ItCModel concept labeler
        self.concept_predictor = ItCModel(model, **params)


    def fit(self, data_gen_l, data_gen_u):
        # Train underlying CME concept predictor
        # The default ItCModel accepts tf_datasets without the y-labels during training
        # Thus, we filter out the y-labels from data_gen_l and data_gen_u via a mapping
        ds_l, ds_u = remove_ds_el(data_gen_l), remove_ds_el(data_gen_u)
        self.concept_predictor.train(ds_l, ds_u)


    def predict(self, data_gen_u):
        # Run the underlying CME predictor
        # The default ItCModel accepts tf_datasets without the y-labels during prediction
        # Thus, we filter out the y-labels from data_gen_u via a mapping
        ds_u = remove_ds_el(data_gen_u)
        predicted = self.concept_predictor.predict_concepts(ds_u)
        return predicted




from methods.CBM.CBModel import ConceptBottleneckModel
from evaluation.metrics.accuracy import compute_accuracies

class SSCC_CBM(SSCClassifier):
    '''
    CBM Model wrapper
    '''
    def __init__(self, **kwargs):
        '''
        :param kwargs: Required arguments:
                        - "model"           : the underlying trained model to extract concepts from
                        - "layer_id"        : layer of the model to use as bottleneck
                        - "n_classes"       : number of classes
                        - "n_concept_vals"  : number of concept values
                        - "multi_task"      : whether to use the multi-task setup, or sigmoid setup (Optional)
                        - "end_to_end"      : whether to create an end-to-end keras model
        '''
        super().__init__(**kwargs)
        # Extract required parameters
        model           = kwargs["base_model"]
        layer_id        = kwargs["layer_id"]
        n_classes       = kwargs["n_classes"]
        n_concept_vals  = kwargs["n_concept_vals"]
        multi_task      = kwargs.get("multi_task", False)
        end_to_end      = kwargs.get("end_to_end", False)
        c_epochs        = kwargs.get("epochs", 10)
        c_extr_path     = kwargs["save_path"]
        c_optimizer     = kwargs.get("c_optimizer", None)
        overwrite       = kwargs.get("overwrite", True)
        lambda_param    = kwargs.get("lambda_param", 1.0)

        # Create the underlying CBM concept model
        self.n_epochs          = c_epochs
        self.log_path          = kwargs["log_path"]
        self.do_logged_fit     = kwargs.get("logged_fit", False)
        self.n_concepts        = len(n_concept_vals)
        self.n_train_predictor = kwargs.get("n_train_predictor", 2000)
        self.concept_predictor = ConceptBottleneckModel(model, layer_id, n_classes, n_concept_vals, c_extr_path,
                                                        multi_task=multi_task, end_to_end=end_to_end, c_epochs=c_epochs,
                                                        c_optimizer=c_optimizer, overwrite=True, lambda_param=lambda_param)


    def fit(self, data_gen_l, data_gen_u):
        if self.do_logged_fit:
            self.logged_fit(data_gen_l, n_epochs=self.n_epochs, frequency=5)
        else:
            self.concept_predictor.train(data_gen_l)


    def logged_fit(self, data_gen_l, n_epochs=60, frequency=10):

        all_c_accs = []
        pred_overwrite = self.concept_predictor.overwrite
        self.concept_predictor.overwrite = True
        self.concept_predictor.n_epochs = frequency

        for i in range(0, n_epochs, frequency):
            self.concept_predictor.train(data_gen_l)

            # Train the concept predictor models
            c_true = np.array([elem[1].numpy() for elem in data_gen_l])
            c_pred = self.predict(data_gen_l)
            c_accs = compute_accuracies(c_true, c_pred)
            all_c_accs.append(c_accs)
            print("Accuracies: ", c_accs)
            print("\n" * 3)
            print(f"Ran {i + frequency}/{n_epochs} epochs...")

        all_c_accs = np.array(all_c_accs)
        fpath = os.path.join(self.log_path, "freq_accuracies.txt")
        np.savetxt(fpath, all_c_accs, fmt='%2.4f')

        self.concept_predictor.overwrite = pred_overwrite
        self.concept_predictor.n_epochs  = self.n_epochs


    def predict(self, data_gen_u):
        ds_u = remove_ds_el(data_gen_u)
        predicted = self.concept_predictor.predict(ds_u)
        return predicted




from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

from methods.VAE.weak_vae import GroupVAEArgmax

class SSCC_Group_VAEArgmax(SSCClassifier):
    '''
    Weakly-supervised VAE Model wrapper
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_changed_c        = kwargs["k"]
        self.n_concept_vals     = kwargs["n_concept_vals"]
        self.n_concepts         = len(self.n_concept_vals)
        self.batch_size         = kwargs.get("batch_size", 256)
        self.n_epochs           = kwargs.get("epochs", 10)
        loss_fn                 = kwargs["loss_fn"]
        encoder_fn              = kwargs["encoder_fn"]
        decoder_fn              = kwargs["decoder_fn"]
        latent_dim              = kwargs["latent_dim"]
        input_shape             = kwargs["input_shape"]
        self.log_path           = kwargs["log_path"]
        self.save_path          = kwargs["save_path"]
        self.do_logged_fit      = kwargs.get("logged_fit", False)
        self.overwrite          = kwargs.get("overwrite", False)
        self.n_train_predictor  = kwargs.get("n_train_predictor", 2000)
        self.optimizer          = kwargs.get("optimizer", tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-8, learning_rate=0.001))
        self.vae                = GroupVAEArgmax(latent_dim, encoder_fn, decoder_fn, loss_fn, input_shape)
        self.vae.compile(optimizer=self.optimizer)
        # Here, we rely on using an ensemble of models (one per concept) for predicting the concepts from the
        # latent factors
        # self.concept_predictors = [LogisticRegression(max_iter=200) for _ in range(self.n_concepts)]
        self.concept_predictors = [GradientBoostingClassifier() for _ in range(self.n_concepts)]

        # Load concept extractor, if the model already exists
        if (not self.overwrite) and (os.path.exists(self.save_path)):
            self.vae(np.zeros([1]+input_shape))   # Need to call model before loading, in used version of Tf
            self.vae.load_weights(self.save_path)


    def _build_paired_dataset(self, ds):

        random_state = np.random.RandomState(0)

        # TODO: temporarily rely on non-tf-data implementation
        from datasets.dataset_utils import get_latent_bases, latent_to_index
        latent_sizes    = np.array(self.n_concept_vals)
        latents_bases   = get_latent_bases(latent_sizes)
        x_data          = np.array([elem[0].numpy() for elem in ds])
        c_data          = np.array([elem[1].numpy() for elem in ds])
        c_ids           = np.array([latent_to_index(elem[1].numpy(), latents_bases) for elem in ds])

        x_modified      = []
        label_ids       = []
        x_filtered_data = []

        n_pairs, n_non_pairs = 0, 0

        for i in range(c_data.shape[0]):
            c = c_data[i]
            x = x_data[i]

            if self.n_changed_c == -1:
                k_observed = random_state.randint(1, self.n_concepts)
            else:
                k_observed = self.n_changed_c

            index_list  = random_state.choice(c.shape[0], random_state.choice([1, k_observed]), replace=False)
            idx         = -1
            c_m         = np.copy(c)

            for index in index_list:
                v          = np.random.choice(np.arange(self.n_concept_vals[index]))
                c_m[index] = v
                idx = index

            x_m = np.where(c_ids == latent_to_index(c_m, latents_bases))[0]

            if len(x_m) > 0:
                n_pairs+=1
                np.random.shuffle(x_m)
                x_m = x_m[0]
                x_m = x_data[x_m]
            else:
                n_non_pairs+=1
                continue

            x_filtered_data.append(x)
            x_modified.append(x_m)
            label_ids.append(idx)

        x_filtered_data = np.array(x_filtered_data)
        x_modified      = np.array(x_modified)
        label_ids       = np.array(label_ids)
        x_pairs         = np.concatenate([x_filtered_data, x_modified], axis=1)

        print("Pairs: ",        n_pairs)
        print("Non-pairs: ",    n_non_pairs)
        print("% pairs: ",      str(n_pairs/(1. * (n_pairs+n_non_pairs)) * 100.))

        paired_ds = tf.data.Dataset.from_tensor_slices((x_pairs, label_ids))
        return paired_ds


    def fit(self, data_gen_l, data_gen_u):
        del data_gen_u

        if not ((not self.overwrite) and (os.path.exists(self.save_path))):
            # Train the VAE
            paired_ds_l = self._build_paired_dataset(data_gen_l)

            self.logged_fit(self.vae, paired_ds_l, data_gen_l, n_epochs=self.n_epochs, frequency=5)


    def logged_fit(self, model, training_gen, eval_gen, n_epochs=60, frequency=10):

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.save_path, verbose=True,
                                                         save_best_only=True, monitor='loss',
                                                         mode='auto', save_freq='epoch')
        callbacks = [cp_callback]

        all_c_accs   = []

        eval_gen_xs = remove_ds_el(remove_ds_el(eval_gen))

        if not self.do_logged_fit: frequency = n_epochs

        for i in range(0, n_epochs, frequency):
            model.fit(training_gen.batch(self.batch_size), epochs=frequency, callbacks=callbacks)

            # Train the concept predictor models
            c_data = np.array([elem[1].numpy() for elem in eval_gen])
            z_data = model.predict(eval_gen_xs.batch(self.batch_size))

            c_accs = []

            for j in range(self.n_concepts):
                # Train concept label predictors, and evaluate their predictive accuracy
                z_train, z_test, c_train, c_test = train_test_split(z_data, c_data[:, j], test_size=0.15)

                if self.n_train_predictor is not None:
                    z_train, c_train = z_train[:self.n_train_predictor], c_train[:self.n_train_predictor]
                # Note: here we assume sklearn .fit() re-initializes all previous params
                self.concept_predictors[j].fit(z_train, c_train)

                accuracy = accuracy_score(c_test, self.concept_predictors[j].predict(z_test))
                print("Accuracy of concept ", str(j), " : ", str(accuracy))
                c_accs.append(accuracy)

            all_c_accs.append(c_accs)
            print("\n"*3)
            print(f"Ran {i+frequency}/{n_epochs} epochs...")

        all_c_accs = np.array(all_c_accs)
        fpath = os.path.join(self.log_path, "freq_accuracies.txt")
        np.savetxt(fpath, all_c_accs, fmt='%2.4f')



    def predict(self, data_gen_u):
        data_x    = remove_ds_el(data_gen_u)
        z_data    = self.vae.predict(data_x.batch(self.batch_size))
        c_data    = np.stack([self.concept_predictors[i].predict(z_data) for i in range(self.n_concepts)], axis=-1)
        return c_data



from methods.VAE.betaVAE import BetaVAE

class SSCC_BetaVAE(SSCClassifier):
    '''
    Beta-VAE Model wrapper
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_concept_vals     = kwargs["n_concept_vals"]
        self.n_concepts         = len(self.n_concept_vals)
        self.batch_size         = kwargs.get("batch_size", 256)
        self.n_epochs           = kwargs.get("epochs", 10)
        beta                    = kwargs.get("beta", 1)
        loss_fn                 = kwargs["loss_fn"]
        encoder_fn              = kwargs["encoder_fn"]
        decoder_fn              = kwargs["decoder_fn"]
        latent_dim              = kwargs["latent_dim"]
        input_shape             = kwargs["input_shape"]
        self.model_path         = kwargs.get("save_path", None)
        self.retrain            = kwargs.get("overwrite", False)
        self.n_train_predictor  = kwargs.get("n_train_predictor", 2000)
        self.optimizer = kwargs.get("optimizer", tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-8, learning_rate=0.001))
        self.vae                = BetaVAE(latent_dim, encoder_fn, decoder_fn, loss_fn, input_shape, beta)
        self.vae.compile(optimizer=self.optimizer)
        # Here, we rely on an ensemble of models (one per concept) for predicting concepts from latent factors
        # self.concept_predictors = [LogisticRegression(max_iter=200) for _ in range(self.n_concepts)]
        self.concept_predictors = [GradientBoostingClassifier() for _ in range(self.n_concepts)]

        # Load concept extractor, if the model already exists
        if (self.model_path is not None) and (os.path.exists(self.model_path)):
            self.vae(np.zeros([1]+input_shape))   # Need to call model before loading, in used version of Tf
            self.vae.load_weights(self.model_path)


    def fit(self, data_gen_l, data_gen_u):
        del data_gen_u

        if not ((self.model_path is not None) and (os.path.exists(self.model_path)) and (not self.retrain)):

            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_path, verbose=True,
                                                             save_best_only=True, monitor='loss',
                                                             mode='auto', save_freq='epoch')
            callbacks = [cp_callback]
            self.vae.fit(data_gen_l.batch(self.batch_size), epochs=self.n_epochs, callbacks=callbacks)

        # Train the concept predictor models
        c_data = np.array([elem[1].numpy() for elem in data_gen_l])
        z_data = self.vae.predict(remove_ds_el(remove_ds_el(data_gen_l)).batch(self.batch_size))

        for i in range(self.n_concepts):
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            z_train, z_test, c_train, c_test = train_test_split(z_data, c_data[:, i], test_size=0.15)

            if self.n_train_predictor is not None:
                z_train, c_train = z_train[:self.n_train_predictor], c_train[:self.n_train_predictor]

            self.concept_predictors[i].fit(z_train, c_train)
            print("Accuracy of concept ", str(i), " : ",
                  str(accuracy_score(c_test, self.concept_predictors[i].predict(z_test))))


    def predict(self, data_gen_u):
        data_x    = remove_ds_el(data_gen_u)
        z_data    = self.vae.predict(data_x.batch(self.batch_size))
        c_data    = np.stack([self.concept_predictors[i].predict(z_data) for i in range(self.n_concepts)], axis=-1)
        return c_data


def remove_ds_el(data_gen):
    '''
    Utility function for removing the last dimension data from a generator.
    :param data_gen: tf.data generator
    :return: data generator without the last tuple element, for every item in data_gen
    '''
    new_data_gen = data_gen.map(lambda *args: tuple([args[i] for i in range(len(args)-1)]))
    return new_data_gen

