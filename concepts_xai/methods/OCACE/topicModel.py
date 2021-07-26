import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

'''
Re-implementation of the "On Completeness-aware Concept-Based Explanations in Deep Neural Networks":

1) See https://arxiv.org/abs/1910.07969 for the original paper

2)  See https://github.com/chihkuanyeh/concept_exp for the original paper implementation:

Note: this code has not been tested yes!

'''


class TopicModel(tf.keras.Model):

    """Abstract base class of a topic model."""
    def __init__(self, predict_fn, n_channels, n_concepts, threshold=0.5, rec_param=500,
                 loss_fn=tf.keras.losses.sparse_categorical_crossentropy, top_k=32, lambda1=0.1, lambda2=0.1, **kwargs):

        super(TopicModel, self).__init__(**kwargs)

        initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)

        # Initialise trainable model variables
        self.topic_vector   = tf.Variable(initializer(shape=(n_channels, n_concepts)), trainable=True)
        self.rec_vector_1   = tf.Variable(initializer(shape=(n_concepts, rec_param)), trainable=True)
        self.rec_vector_2   = tf.Variable(initializer(shape=(rec_param, n_channels)), trainable=True)
        # Set the concept-to-label predictor model
        self.pred_fn        = predict_fn
        # Set remaining model hyperparams
        self.threshold      = threshold
        self.n_concepts     = n_concepts
        self.loss_fn        = loss_fn
        self.top_k          = top_k
        self.lambda1        = lambda1
        self.lambda2        = lambda2
        self.metric_names   = ["loss", "mean_sim", "accuracy"]
        self.metrics_dict   = {name: tf.keras.metrics.Mean(name=name) for name in self.metric_names}
        self.var_list       = [self.topic_vector, self.rec_vector_1, self.rec_vector_2]


    @property
    def metrics(self):
        return [self.metrics_dict[name] for name in self.metric_names]


    def update_metrics(self, losses):
        for (loss_name, loss) in losses: self.metrics_dict[loss_name].update_state(loss)


    def _compute_loss(self, x, y_true, is_training, with_losses=True):

        # Compute the concept representation
        f_input             = x
        f_input_n           = K.l2_normalize(f_input, axis=(3))
        topic_vector_n      = K.l2_normalize(self.topic_vector, axis=0)
        topic_prob          = K.dot(f_input, topic_vector_n)
        topic_prob_n        = K.dot(f_input_n,topic_vector_n)
        topic_prob_mask     = K.cast(K.greater(topic_prob_n, self.threshold),'float32')
        topic_prob_am       = topic_prob*topic_prob_mask
        topic_prob_sum      = K.sum(topic_prob_am, axis=3, keepdims=True)+1e-3
        topic_prob_nn       = topic_prob_am/topic_prob_sum
        rec_layer_1         = K.relu(K.dot(topic_prob_nn, self.rec_vector_1))
        rec_layer_2         = K.dot(rec_layer_1, self.rec_vector_2)
        # Compute the task label from the concept representation
        y_pred              = self.pred_fn(rec_layer_2, training=is_training)

        if not with_losses:
            return y_pred

        # Compute the task loss
        log_prob_loss   = tf.reduce_mean(self.loss_fn(y_true, y_pred))

        # Compute the regularization loss terms
        reg_loss_1      = tf.reduce_mean(tf.nn.top_k(K.transpose(K.reshape(topic_prob_n, (-1, self.n_concepts))),
                                            k=self.top_k, sorted=True).values)

        reg_loss_2      = tf.reduce_mean(K.dot(K.transpose(topic_vector_n), topic_vector_n) - np.eye(self.n_concepts))

        total_loss      = log_prob_loss - self.lambda1*reg_loss_1 + self.lambda2*reg_loss_2


        # Compute the regularization similarity metric to track
        sim_metric = tf.reduce_mean(tf.nn.top_k(K.transpose(K.reshape(topic_prob_n,(-1, self.n_concepts))),
                                                  k=self.top_k,sorted=True).values)

        # Compute the accuracy metric to track
        # TODO: load the accuracy corresponding to the loss_fn
        acc = tf.keras.metrics.SparseCategoricalAccuracy()
        acc.update_state(y_true, y_pred)
        accuracy_metric = acc.result()

        return total_loss, sim_metric, accuracy_metric, y_pred


    def train_step(self, data):
        x, y = data
        is_training=True

        with tf.GradientTape() as tape:
            loss, mean_sim, acc, _ = self._compute_loss(x, y, is_training)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.update_metrics([("loss", loss), ("mean_sim", mean_sim), ("accuracy", acc)])

        return {name: self.metrics_dict[name].result() for name in self.metric_names}


    def test_step(self, data):
        x, y = data
        is_training=False
        loss, mean_sim, acc, _ = self._compute_loss(x, y, is_training)
        self.update_metrics([("loss", loss), ("mean_sim", mean_sim), ("accuracy", acc)])

        return {name: self.metrics_dict[name].result() for name in self.metric_names}


    def call(self, x, **kwargs):
        return self._compute_loss(x, None, is_training=False, with_losses=False)










