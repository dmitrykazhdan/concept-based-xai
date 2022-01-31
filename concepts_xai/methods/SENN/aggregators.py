"""
Library of some default aggregator functions that one can use in SENN.
"""

import tensorflow as tf


def multiclass_additive_aggregator(thetas, concepts):
    # Returns output shape (batch, n_outputs)
    return tf.squeeze(
        # (batch, n_outputs, 1)
        tf.linalg.matmul(
            # (batch, n_outputs, n_concepts)
            thetas,
            # (batch, n_concepts, 1)
            tf.expand_dims(concepts, axis=-1),
        ),
        axis=-1
    )


def scalar_additive_aggregator(thetas, concepts):
    # Returns output shape (batch)
    return tf.squeeze(
        multiclass_additive_aggregator(thetas=thetas, concepts=concepts),
        axis=-1,
    )


def softmax_additive_aggregator(thetas, concepts):
    # Returns output shape (batch, n_outputs)
    return tf.nn.softmax(
        multiclass_additive_aggregator(thetas=thetas, concepts=concepts),
        axis=-1,
    )
