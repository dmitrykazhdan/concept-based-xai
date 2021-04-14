import tensorflow as tf
import math

'''
Re-implementation of the Concept Whitening approach.

1)  See the IterNormRotation class in for the original paper implementation:
    https://github.com/zhiCHEN96/ConceptWhitening/blob/final_version/MODELS/iterative_normalization.py

2)  See Concept Whitening for Interpretable Image Recognition (https://arxiv.org/abs/2002.01650) for the original paper

Note: this code has not been tested yes!

'''

class ConceptWhiteningLayer(tf.keras.layers.Layer):

    def __init__(self, num_features):

        super(ConceptWhiteningLayer, self).__init__()

        self.T                  = 10
        self.eps                = 1e-5
        self.momentum           = 0.05
        self.num_features       = num_features  # Total number of channels.
        self.affine             = False
        self.dim                = 4             # Note: currently works for 3D images only
        self.mode               = -1            # Used to indicate whether to update the G matrix
        self.activation_mode    = 'pool_max'    # Methods for aggregating the maxpool

        # Current implementation works with 1 group only (does not support group whitening)
        self.num_groups     = 1
        self.num_channels   = num_features
        shape               = [1] * self.dim
        shape[1]            = self.num_features
        broadcast_shape     = [self.num_groups, self.num_channels, self.num_channels]

        # Running mean
        self.running_mean  = tf.zeros((self.num_groups, self.num_channels, 1))
        # Running whitened matrix
        self.running_wm    = tf.broadcast_to(tf.eye(self.num_channels), broadcast_shape)
        # Running rotation matrix
        self.running_rot   = tf.broadcast_to(tf.eye(self.num_channels), broadcast_shape)
        # Sum of Gradient matrix
        self.sum_G         = tf.zeros((self.num_groups, self.num_channels, self.num_channels))
        # Number of gradient for each concept
        self.counter       = tf.ones((self.num_channels))*0.001


    def build(self, input_shape):
        super(ConceptWhiteningLayer, self).build(input_shape)


    def _update_rotation_matrix(self):
        """
        Update the rotation matrix R using the accumulated gradient G.
        """

        size_R = self.running_rot.shape
        counter_exp = tf.expand_dims(self.counter, axis=-1)
        G = self.sum_G / counter_exp
        R = tf.identity(self.running_rot)

        for i in range(2):
            tau     = 1000  # learning rate in Cayley transform
            alpha   = 0
            beta    = 100000000
            c1      = 1e-4
            c2      = 0.9

            A = tf.einsum('gin,gjn->gij', G, R) - tf.einsum('gin,gjn->gij', R, G)   # GR^T - RG^T
            I = tf.broadcast_to(tf.eye(size_R[2]), shape=size_R)
            dF_0 = -0.5 * tf.math.reduce_sum(A ** 2)

            # Computing tau using Algorithm 1 in https://link.springer.com/article/10.1007/s10107-012-0584-1
            # Binary search for appropriate learning rate
            cnt = 0
            while True:
                Q = tf.linalg.matmul(tf.linalg.inv(I + 0.5 * tau * A), I - 0.5 * tau * A)
                Y_tau = tf.linalg.matmul(Q, R)
                F_X = tf.math.reduce_sum(G[:, :, :] * R[:, :, :])
                F_Y_tau = tf.math.reduce_sum(G[:, :, :] * Y_tau[:, :, :])
                dF_tau = tf.linalg.matmul(tf.einsum('gni,gnj->gij', G, tf.linalg.inv(I + 0.5 * tau * A)),
                                                 tf.linalg.matmul(A, 0.5 * (R + Y_tau)))
                dF_tau = -1.0 * tf.linalg.trace(dF_tau[0, :, :])

                if F_Y_tau > F_X + c1 * tau * dF_0 + 1e-18:
                    beta = tau
                    tau = (beta + alpha) / 2
                elif dF_tau + 1e-18 < c2 * dF_0:
                    alpha = tau
                    tau = (beta + alpha) / 2
                else:
                    break
                cnt += 1

                if cnt > 500:
                    print("--------------------update fail------------------------")
                    print(F_Y_tau, F_X + c1 * tau * dF_0)
                    print(dF_tau, c2 * dF_0)
                    print("-------------------------------------------------------")
                    break

            # Using the un-numbered equation in the Concept Whitening paper
            Q = tf.linalg.matmul(tf.linalg.inv(I + 0.5 * tau * A), I - 0.5 * tau * A)
            R = tf.linalg.matmul(Q, R)

        self.running_rot = R
        self.counter = tf.ones((size_R[-1])) * 0.001


    def call(self, X, training=False):

        X_hat = self._compute_whitened_activations(X, training)

        size_X = X_hat.shape
        size_R = self.running_rot.shape

        # Switch from (N, C, H, W) shape to (N, G, C, H, W)
        ngchw_shape = [size_X[0], size_R[0], size_R[2]]+size_X[2:]
        X_hat = tf.reshape(X_hat, ngchw_shape)

        # Updating the gradient matrix, using the concept datasets
        # "mode" specifies the concept index, which specifies which gradient matrix column to update
        if self.mode >= 0:
            if self.activation_mode == 'mean':

                self.sum_G = tf.Variable(self.sum_G)
                self.sum_G[:, self.mode, :].assign( self.momentum * -tf.math.reduce_mean(X_hat, axis=[0, 3, 4]) + \
                                                    (1. - self.momentum) * self.sum_G[:, self.mode, :])
                self.sum_G = tf.convert_to_tensor(self.sum_G)

                self.counter = tf.Variable(self.counter)
                self.counter[self.mode].assign(self.counter[self.mode]+1)
                self.counter = tf.convert_to_tensor(self.counter)

            elif self.activation_mode == 'pool_max':
                X_test = tf.linalg.einsum('bgchw,gdc->bgdhw', X_hat, self.running_rot)
                X_test_nchw = tf.reshape(X_test, size_X)
                X_test_nhwc = tf.transpose(X_test_nchw, perm=[0, 2, 3, 1])

                # Note: tensorflow does not currently support the 2D-unpool op available in PyTorch
                # Hence, the implementation is a little different here
                maxpool_value, max_indices = tf.nn.max_pool_with_argmax(X_test_nhwc, ksize=3, strides=3,
                                                                        padding='SAME', data_format='NHWC')
                mean_maxpool = tf.math.reduce_mean(maxpool_value, axis=[1, 2])
                mean_maxpool = tf.reshape(mean_maxpool, [mean_maxpool.shape[0], size_R[0], size_R[2]])
                grad = -1. * tf.math.reduce_mean(mean_maxpool, axis=0)
                self.sum_G[:, self.mode, :] = self.momentum * grad + (1. - self.momentum) * self.sum_G[:, self.mode, :]
                self.counter[self.mode] += 1

            else:
                raise NotImplementedError("Currently supporting only the pool_max option")

        X_hat = tf.linalg.einsum('bgchw,gdc->bgdhw', X_hat, self.running_rot)
        X_hat = tf.reshape(X_hat, size_X)

        return X_hat


    def _compute_whitened_activations(self, X, training):
        '''
        Implements Algorithm 1 from https://arxiv.org/pdf/1904.03441.pdf
        Note, however, that it implements the grouped variant
        Also, updates the running mean and whitening matrices using a moving average

        For the idea of "groups" of activations, see - https://arxiv.org/pdf/1804.08450.pdf

        Assumes the input X is in the format of (N, C, H, W)
        '''

        nc          = self.num_channels
        momentum    = self.momentum
        eps         = self.eps
        T           = self.T

        # Calculate the number of groups
        assert (X.shape[1] % nc == 0)
        g = X.shape[1] // nc
        assert g == self.num_groups

        # Flip first two dimensions in order to obtain a (C, N, H, W) tensor
        N, C, H, W = X.shape
        x = tf.transpose(X, perm=[1, 0, 2, 3])

        # Change (C, N, H, W) to (G, D, NxHxW)
        m = N * H * W
        x = tf.reshape(x, [g, nc, m])

        saved = []
        if training:
            # Calculate mini-batch mean
            mean = tf.reduce_mean(x, axis=-1, keepdims=True)

            # Calculate centered activation
            xc = x - mean
            saved.append(xc)

            # Calculate covariance matrix
            I = tf.broadcast_to(tf.eye(nc), shape=[g, nc, nc])
            Sigma = eps*I + 1./m * tf.linalg.matmul(xc, tf.transpose(xc, perm=[0, 2, 1]))

            # Calculate trace-normalized covariance matrix using eqn. (4) in the paper
            Sigma_Tr_rec = tf.math.reciprocal(tf.linalg.trace(Sigma))
            # Keep shape (g, 1, 1)
            Sigma_Tr_rec = tf.expand_dims(tf.expand_dims(Sigma_Tr_rec, axis=-1), axis=-1)
            Sigma_N = Sigma * Sigma_Tr_rec
            saved.append(Sigma_Tr_rec)
            saved.append(Sigma_N)

            # Calculate whitening matrix
            P = [tf.identity(I) for _ in range(T+2)]
            for k in range(T):
                P_k_cubed = tf.linalg.matmul(tf.linalg.matmul(P[k], P[k]), P[k])
                P[k+1] = 1.5*P[k] - 0.5*tf.linalg.matmul(P_k_cubed, Sigma_N)

            saved.extend(P)
            wm = tf.math.multiply(P[T], tf.math.sqrt(Sigma_Tr_rec))
            self.running_mean   = momentum * mean + (1. - momentum) * self.running_mean
            self.running_wm     = momentum * wm   + (1. - momentum) * self.running_wm

        else:
            xc = x - self.running_mean
            wm = self.running_wm

        # Calculate whitening output
        xn = tf.linalg.matmul(wm, xc)

        # Transform back to original shape of (N, C, H, W)
        xn = tf.reshape(xn, [C, N, H, W])
        Xn = tf.transpose(xn, perm=[1, 0, 2, 3])

        return Xn
