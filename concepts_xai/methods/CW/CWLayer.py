'''
Re-implementation of the Concept Whitening module proposed by Chen et al..

1)  See the IterNormRotation class in for the original paper implementation:
    https://github.com/zhiCHEN96/ConceptWhitening/blob/final_version/MODELS/iterative_normalization.py

2)  See Concept Whitening for Interpretable Image Recognition
    (https://arxiv.org/abs/2002.01650) for the original paper
'''

import tensorflow as tf

################################################################################
## Helper classes taken from tensorflow-addons to avoid import and extend
## functions.
## Full code can be found in
## https://github.com/tensorflow/addons/blob/v0.13.0/tensorflow_addons/layers/max_unpooling_2d.py#L88-L147
################################################################################


def normalize_tuple(value, n, name):
    """Transforms an integer or iterable of integers into an integer tuple.
    A copy of tensorflow.python.keras.util.
    Args:
      value: The value to validate and convert. Could an int, or any iterable
        of ints.
      n: The size of the tuple to be returned.
      name: The name of the argument being validated, e.g. "strides" or
        "kernel_size". This is only used to format error messages.
    Returns:
      A tuple of n integers.
    Raises:
      ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise TypeError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(n)
                + " integers. Received: "
                + str(value)
            )
        if len(value_tuple) != n:
            raise ValueError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(n)
                + " integers. Received: "
                + str(value)
            )
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError(
                    "The `"
                    + name
                    + "` argument must be a tuple of "
                    + str(n)
                    + " integers. Received: "
                    + str(value)
                    + " "
                    "including element "
                    + str(single_value)
                    + " of type"
                    + " "
                    + str(type(single_value))
                )
        return value_tuple


def _calculate_output_shape(input_shape, pool_size, strides, padding):
    """Calculates the shape of the unpooled output."""
    if padding == "VALID":
        output_shape = (
            input_shape[0],
            (input_shape[1] - 1) * strides[0] + pool_size[0],
            (input_shape[2] - 1) * strides[1] + pool_size[1],
            input_shape[3],
        )
    elif padding == "SAME":
        output_shape = (
            input_shape[0],
            input_shape[1] * strides[0],
            input_shape[2] * strides[1],
            input_shape[3],
        )
    else:
        raise ValueError('Padding must be a string from: "SAME", "VALID"')
    return output_shape


def _max_unpooling_2d(
    updates,
    mask,
    pool_size=(2, 2),
    strides=(2, 2),
    padding="SAME",
):
    """Unpool the outputs of a maximum pooling operation."""
    mask = tf.cast(mask, "int32")
    pool_size = normalize_tuple(pool_size, 2, "pool_size")
    strides = normalize_tuple(strides, 2, "strides")
    input_shape = tf.shape(updates, out_type="int32")
    input_shape = [updates.shape[i] or input_shape[i] for i in range(4)]
    output_shape = _calculate_output_shape(
        input_shape,
        pool_size,
        strides,
        padding,
    )

    # Calculates indices for batch, height, width and feature maps.
    one_like_mask = tf.ones_like(mask, dtype="int32")
    batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], axis=0)
    batch_range = tf.reshape(
        tf.range(output_shape[0], dtype="int32"), shape=batch_shape
    )
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = (mask // output_shape[3]) % output_shape[2]
    feature_range = tf.range(output_shape[3], dtype="int32")
    f = one_like_mask * feature_range

    # Transposes indices & reshape update values to one dimension.
    updates_size = tf.size(updates)
    indices = tf.transpose(
        tf.reshape(tf.stack([b, y, x, f]), [4, updates_size])
    )
    values = tf.reshape(updates, [updates_size])
    return tf.scatter_nd(indices, values, output_shape)


################################################################################
## Concept Whitening Layer Class
################################################################################


class ConceptWhiteningLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        T=5,
        eps=1e-5,
        momentum=0.9,
        activation_mode='max_pool_mean',
        c1=1e-4,
        c2=0.9,
        max_tau_iterations=500,
        initial_tau=1000.0,  # Original CW code: 1000
        data_format="channels_first",
        initial_beta=1e8,  # Original CW code: 1e8
        initial_alpha=0,  # Original CW code: 0
        **kwargs
    ):
        super(ConceptWhiteningLayer, self).__init__(**kwargs)
        assert data_format in ["channels_first", "channels_last"], (
            f'Expected data format to be either "channels_first" or '
            f'"channels_last" but got {data_format} instead.'
        )

        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.c1 = c1
        self.c2 = c2
        self.max_tau_iterations = max_tau_iterations
        self.data_format = data_format
        self.initial_tau = initial_tau
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta

        # Methods for aggregating a feature map into an score
        self.activation_mode = activation_mode

    def compute_output_shape(self, input_shape):
        return input_shape

    def concept_scores(
        self,
        inputs,
        aggregator='max_pool_mean',
        concept_indices=None,
    ):
        outputs = self(inputs, training=False)
        if len(tf.shape(outputs)) == 2:
            # Then the scores are already computed by our forward pass
            scores = outputs
        else:
            if self.data_format == "channels_last":
                # Then we will transpose to make things simpler so that
                # downstream we can always assume it is channels first
                # NHWC -> NCHW
                outputs = tf.transpose(
                    outputs,
                    perm=[0, 3, 1, 2],
                )

            # Else, we need to do some aggregation
            if aggregator == 'mean':
                # Compute the mean over all channels
                scores = tf.math.reduce_mean(outputs, axis=[2, 3])
            elif aggregator == 'max_pool_mean':
                # First downsample using a max pool and then continue with
                # a mean
                window_size = min(
                    2,
                    outputs.shape[-1],
                    outputs.shape[-2],
                )
                scores = tf.nn.max_pool(
                    outputs,
                    ksize=window_size,
                    strides=window_size,
                    padding="SAME",
                    data_format="NCHW",
                )
                scores = tf.math.reduce_mean(scores, axis=[2, 3])
            elif aggregator == 'max':
                # Simply select the maximum value across a given channel
                scores = tf.math.reduce_max(outputs, axis=[2, 3])
            else:
                raise ValueError(f'Unsupported aggregator {aggregator}.')

        if concept_indices is not None:
            return scores[:, concept_indices]
        return scores

    def build(self, input_shape):
        super(ConceptWhiteningLayer, self).build(input_shape)

        # And use the shape to construct all our running variables
        assert len(input_shape) in [2, 4], (
            f'Expected input to CW layer to be a rank-2 or rank-4 matrix but '
            f'instead got a tensor with shape {input_shape}.'
        )

        # We assume channels-first data format
        if self.data_format == "channels_first":
            self.num_features = input_shape[1]
        else:
            self.num_features = input_shape[-1]

        # Running mean
        self.running_mean = self.add_weight(
            name="running_mean",
            shape=(self.num_features,),
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
            # No gradient flow is expected to come into this variable
            trainable=False,
        )

        # Running whitened matrix
        self.running_wm = self.add_weight(
            name="running_wm",
            shape=(self.num_features, self.num_features),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Identity(),
            # No gradient flow is expected to come into this variable
            trainable=False,
        )

        # Running rotation matrix
        self.running_rot = self.add_weight(
            name="running_rot",
            shape=(self.num_features, self.num_features),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Identity(),
            # No gradient flow is expected to come into this variable
            trainable=False,
        )

        # Sum of Gradient matrix
        self.sum_G = self.add_weight(
            name="sum_G",
            shape=(self.num_features, self.num_features),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Zeros(),
            # No gradient flow is expected to come into this variable
            trainable=False,
        )

        # Counter of gradients for each feature
        self.counter = self.add_weight(
            name="counter",
            shape=(self.num_features,),
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.001),
            # No gradient flow is expected to come into this variable
            trainable=False,
        )

    def update_rotation_matrix(self, concept_groups, index_map=lambda x: x):
        """
        Update the rotation matrix R using the accumulated gradient G.
        """

        # Updating the gradient matrix, using the concept datasets and their
        # aligned maps
        for i, concept_samples in enumerate(concept_groups):

            samples_shape = tf.shape(concept_samples)
            if len(samples_shape) == 2:
                # Add trivial dimensions to make it 4D so that it works as
                # it does with image data. We will undo this at the end
                concept_samples = tf.reshape(
                    concept_samples,
                    tf.concat(
                        [samples_shape[0:1], samples_shape[1:], [1, 1]],
                        axis=0,
                    ),
                )
            if (
                (self.data_format == "channels_last") and
                (len(samples_shape) != 2)
            ):
                # Then we will transpose to make things simpler so that
                # downstream we can always assume it is channels first
                # NHWC -> NCHW
                concept_samples = tf.transpose(
                    concept_samples,
                    perm=[0, 3, 1, 2],
                )

            # Produce the whitened only activations of this concept group
            X_hat = self._compute_whitened_activations(
                concept_samples,
                rotate=False,
                training=False,
            )

            # Determine which feature map to use for this concept group
            feature_idx = index_map(i)

            # And update the gradient by performing an accumulation using the
            # requested activation mode
            if self.activation_mode == 'mean':
                grad_col = -tf.math.reduce_mean(
                    tf.math.reduce_mean(X_hat, axis=[2, 3]),
                    axis=0,
                )

                self.sum_G[feature_idx, :].assign(
                    (
                        (grad_col * self.momentum)
                    ) + (1. - self.momentum) * self.sum_G[feature_idx, :]
                )

            elif self.activation_mode == 'max_pool_mean':
                X_test_nchw = tf.linalg.einsum(
                    'bchw,dc->bdhw',
                    X_hat,
                    self.running_rot,
                )

                # Move to NHWC as tf.nn.max_pool_with_argmax only supports
                # channels last format
                X_test_nhwc = tf.transpose(X_test_nchw, perm=[0, 2, 3, 1])

                window_size = min(
                    2,
                    X_hat.shape[-1],
                    X_hat.shape[-2],
                )
                maxpool_value, max_indices = tf.nn.max_pool_with_argmax(
                    X_test_nhwc,
                    ksize=window_size,
                    strides=window_size,
                    padding='SAME',
                    data_format='NHWC',
                )
                X_test_unpool = _max_unpooling_2d(
                    maxpool_value,
                    max_indices,
                    pool_size=window_size,
                    strides=window_size,
                    padding="SAME",
                )

                # And reshape to original NCHW format from NHWC format
                X_test_unpool = tf.transpose(X_test_unpool, perm=[0, 3, 1, 2])

                # Finally, compute the actual gradient and update or running
                # matrix
                maxpool_mask = tf.cast(
                    tf.math.equal(X_test_nchw, X_test_unpool),
                    tf.float32,
                )
                # Average only over those elements selected by the max pool
                # operator.
                grad = (
                    tf.reduce_sum(X_hat * maxpool_mask, axis=(2, 3)) /
                    tf.reduce_sum(maxpool_mask, axis=(2, 3))
                )

                # And average over all samples
                grad = -tf.reduce_mean(grad, axis=0)
                self.sum_G[feature_idx, :].assign(
                    self.momentum * grad +
                    (1. - self.momentum) * self.sum_G[feature_idx, :]
                )

            else:
                raise NotImplementedError(
                    "Currently supporting only the max_pool_mean and mean "
                    "options"
                )

            # And increase the counter
            self.counter[feature_idx].assign(self.counter[feature_idx] + 1)

        # Time to update our rotation matrix
        # Original CW paper does this counter division so keeping it for now
        # for backwards compatability
        G = self.sum_G / tf.expand_dims(self.counter, axis=-1)

        # Original CW code uses range(2) for some bizzare reason
        for _ in range(2):
            tau = self.initial_tau  # learning rate in Cayley transform
            alpha = self.initial_alpha
            beta = self.initial_beta

            # Compute: GR^T - RG^T
            A = tf.einsum('in,jn->ij', G, self.running_rot) - tf.einsum(
                'in,jn->ij',
                self.running_rot,
                G,
            )
            I = tf.eye(self.num_features)
            dF_0 = -0.5 * tf.math.reduce_sum(A * A)

            # Computing tau using Algorithm 1 in
            # https://link.springer.com/article/10.1007/s10107-012-0584-1
            # Binary search for appropriate learning rate
            count = 0
            while count < self.max_tau_iterations:
                Q = tf.linalg.matmul(
                    tf.linalg.inv(I + 0.5 * tau * A),
                    I - 0.5 * tau * A,
                )
                Y_tau = tf.linalg.matmul(Q, self.running_rot)
                F_X = tf.math.reduce_sum(G * self.running_rot)
                F_Y_tau = tf.math.reduce_sum(G * Y_tau)
                dF_tau = tf.linalg.matmul(
                    tf.einsum(
                        'ni,nj->ij',
                        G,
                        tf.linalg.inv(I + 0.5 * tau * A),
                    ),
                    tf.linalg.matmul(A, 0.5 * (self.running_rot + Y_tau)),
                )
                dF_tau = -tf.linalg.trace(dF_tau)

                if F_Y_tau > F_X + self.c1 * tau * dF_0 + 1e-18:
                    beta = tau
                    tau = (beta + alpha) / 2
                elif dF_tau + 1e-18 < self.c2 * dF_0:
                    alpha = tau
                    tau = (beta + alpha) / 2
                else:
                    break
                count += 1

                if count > self.max_tau_iterations:
                    print("------------------update fail----------------------")
                    print(F_Y_tau, F_X + self.c1 * tau * dF_0)
                    print(dF_tau, self.c2 * dF_0)
                    print("---------------------------------------------------")
                    break

            # Using the un-numbered equation in the Concept Whitening paper
            # Lines 12-13 of Algorithm 2 in CW paper
            Q = tf.linalg.matmul(
                tf.linalg.matmul(
                    tf.linalg.inv(I + 0.5 * tau * A),
                    I - 0.5 * tau * A,
                ),
                self.running_rot,
            )
            # And update the rotation matrix as well as reset the counters
            self.running_rot.assign(Q)

        self.counter.assign(tf.ones((self.num_features,)) * 0.001)

    def call(self, inputs, training=False):
        input_shape = tf.shape(inputs)
        static_imputs_shape = inputs.shape
        if len(static_imputs_shape) == 2:
            # Add trivial dimensions to make it 4D so that it works as
            # it does with image data. We will undo this at the end
            inputs = tf.reshape(
                inputs,
                tf.concat(
                    [input_shape[0:1], input_shape[1:], [1, 1]],
                    axis=0,
                ),
            )
        if (self.data_format == "channels_last") and (
            len(static_imputs_shape) != 2
        ):
            # Then we will transpose to make things simpler so that downstream
            # we can always assume it is channels first
            # NHWC -> NCHW
            inputs = tf.transpose(
                inputs,
                perm=[0, 3, 1, 2],
            )

        result = tf.linalg.einsum(
            'bchw,dc->bdhw',
            self._compute_whitened_activations(inputs, training),
            self.running_rot,
        )

        if len(static_imputs_shape) == 2:
            # Then let's get it back to its original shape
            result = tf.reshape(result, input_shape)
        if (self.data_format == "channels_last") and (
            len(static_imputs_shape) != 2
        ):
            # Then let's move it back to channels last
            # NCHW -> NHWC
            result = tf.transpose(
                result,
                perm=[0, 2, 3, 1],
            )
        return result

    def _compute_whitened_activations(self, X, training, rotate=False):
        '''
        Implements Algorithm 1 from https://arxiv.org/pdf/1904.03441.pdf
        Also, updates the running mean and whitening matrices using a moving
        average

        Assumes the input X is in the format of (N, C, H, W)
        '''

        input_shape = tf.shape(X)

        # Flip first two dimensions in order to obtain a (C, N, H, W) tensor
        x = tf.transpose(X, perm=[1, 0, 2, 3])

        # Change (C, N, H, W) to (D, NxHxW)
        cnhw_shape = tf.shape(x)
        x = tf.reshape(x, [self.num_features, -1])
        m = tf.shape(x)[-1]

        if training:
            # Calculate mini-batch mean
            # Line 4 of Algorithm 1
            mean = tf.reduce_mean(x, axis=-1, keepdims=True)

            # Calculate centered activation
            # Line 5 of Algorithm 1
            xc = x - mean

            # Calculate covariance matrix
            # Line 6 of Algorithm 1
            I = tf.eye(self.num_features)
            sigma = self.eps * I
            sigma += 1./tf.cast(m, tf.float32) * tf.linalg.matmul(
                xc,
                tf.transpose(xc)
            )
            # Calculate trace-normalized covariance matrix using eqn. (4) in
            # the paper
            # Line 7 of Algorithm 1
            sigma_tr_rec = tf.expand_dims(
                tf.math.reciprocal(tf.linalg.trace(sigma)),
                axis=-1,
            )
            sigma_N = sigma * sigma_tr_rec

            # Original CW code: (they do not use the actual trace and this can
            # leas to instability during training)
            # sigma_tr_rec = tf.reduce_sum(sigma, axis=(0, 1), keepdims=True)
            # sigma_N = sigma * sigma_tr_rec

            # Calculate whitening matrix
            # Lines 8-11 of Algorithm 1
            P = tf.eye(self.num_features)
            for _ in range(self.T):
                P_cubed = tf.linalg.matmul(
                    tf.linalg.matmul(P, P),
                    P,
                )
                P = 1.5 * P - (
                    0.5 * tf.linalg.matmul(P_cubed, sigma_N)
                )

            # Line 12 of Algorithm 1
            wm = tf.math.multiply(P, tf.math.sqrt(sigma_tr_rec))

            # Update the running mean and whitening matrix
            self.running_mean.assign(
                self.momentum * tf.squeeze(mean, axis=-1) +
                (1. - self.momentum) * self.running_mean
            )
            self.running_wm.assign(
                self.momentum * wm +
                (1. - self.momentum) * self.running_wm
            )

        else:
            xc = x - tf.expand_dims(self.running_mean, axis=-1)
            wm = self.running_wm

        # Calculate whitening output
        xn = tf.linalg.matmul(wm, xc)

        # And, if requested, apply the rotation while it is in this format
        if rotate:
            xn = tf.linalg.einsum(
                'bchw,dc->bdhw',
                xn,
                self.running_rot,
            )

        # Transform back to original shape of (N, C, H, W)
        return tf.transpose(tf.reshape(xn, cnhw_shape), perm=[1, 0, 2, 3])

    def get_config(self):
        """
        Serialization function.
        """
        result = super(ConceptWhiteningLayer, self).get_config()
        result.update(dict(
            T=self.T,
            eps=self.eps,
            momentum=self.momentum,
            activation_mode=self.activation_mode,
            c1=self.c1,
            c2=self.c2,
            max_tau_iterations=self.max_tau_iterations,
        ))
        return result
