import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf

def bernoulli_loss(
    true_images,
    reconstructed_images,
    activation,
    subtract_true_image_entropy=False,
):

    """Computes the Bernoulli loss."""
    flattened_dim = np.prod(true_images.get_shape().as_list()[1:])
    reconstructed_images = tf.reshape(
        reconstructed_images,
        shape=[-1, flattened_dim]
    )
    true_images = tf.reshape(true_images, shape=[-1, flattened_dim])

    # Because true images are not binary, the lower bound in the xent is not
    # zero: the lower bound in the xent is the entropy of the true images.
    if subtract_true_image_entropy:
        dist = tfp.distributions.Bernoulli(
            probs=tf.clip_by_value(true_images, 1e-6, 1 - 1e-6)
        )
        loss_lower_bound = tf.reduce_sum(dist.entropy(), axis=1)
    else:
        loss_lower_bound = 0

    if activation == "logits":
        loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=reconstructed_images,
                labels=true_images
            ),
            axis=1,
        )
    elif activation == "tanh":
        reconstructed_images = tf.clip_by_value(
            tf.nn.tanh(reconstructed_images) / 2 + 0.5, 1e-6, 1 - 1e-6
        )
        loss = -tf.reduce_sum(
            (
                true_images * tf.math.log(reconstructed_images) +
                (1 - true_images) * tf.math.log(1 - reconstructed_images)
            ),
            axis=1,
        )
    else:
        raise NotImplementedError("Activation not supported.")

    return loss - loss_lower_bound


def l2_loss(true_images, reconstructed_images, activation):
    """Computes the l2 loss."""
    if activation == "logits":
        return tf.reduce_sum(
            tf.square(true_images - tf.nn.sigmoid(reconstructed_images)),
            [1, 2, 3]
        )
    elif activation == "tanh":
        reconstructed_images = tf.nn.tanh(reconstructed_images) / 2 + 0.5
        return tf.reduce_sum(
            tf.square(true_images - reconstructed_images),
            [1, 2, 3],
        )
    else:
        raise NotImplementedError("Activation not supported.")


def bernoulli_fn_wrapper(
    activation="logits",
    subtract_true_image_entropy=False,
):

    def loss_fn(true_images, reconstructed_images):
        return bernoulli_loss(
            true_images,
            reconstructed_images,
            activation,
            subtract_true_image_entropy,
        )
    return loss_fn


def l2_loss_wrapper(activation="logits"):
    def loss_fn(true_images, reconstructed_images):
        return l2_loss(true_images, reconstructed_images, activation)

    return loss_fn

