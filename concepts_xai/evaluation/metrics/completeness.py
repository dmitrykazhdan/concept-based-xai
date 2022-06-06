'''
Implementation of metrics used for computing the completeness/explicitness
of a given set of concepts.
These metrics are inspired by Yeh et al. "On Completeness-aware Concept-Based
Explanations in Deep Neural Networks" seen in https://arxiv.org/abs/1910.07969v5
'''

import numpy as np
import sklearn
import tensorflow as tf

from sklearn.model_selection import train_test_split


################################################################################
## Helper Functions
################################################################################

def _get_default_model(num_concepts, num_hidden_acts, end_activation=None):
    """
    Helper function that returns a simple 3-layer ReLU MLP model with a hidden
    layer with 500 activation in it.

    Used as the default estimator for computing the completeness score.

    :param int num_concepts: The number concept vectors we have in our setup.
    :param int num_hidden_acts: The dimensionality of the concept vectors.

    :returns tf.keras.Model: The default model used for reconstructing a set of
        intermediate hidden activations from a set of concepts scores.
    """

    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            500,
            input_dim=num_concepts,
            activation='relu'
        ),
        tf.keras.layers.Dense(
            num_hidden_acts,
            activation=end_activation,
        ),
    ])


################################################################################
## Concept Score Functions
################################################################################


def dot_prod_concept_score(
    features,
    concept_vectors,
    epsilon=1e-5,
    beta=1e-5,
    channels_axis=-1,
):
    """
    Returns a vector of concept scores for the given features using a normalized
    dot product similarity as in Yeh et al.

    :param np.ndarray features: A 2D matrix of samples with shape
        (n_samples, ..., n_features, ...) with dimension at axis channels_axis
        havin `n_features` in it.
    :param np.ndarray concept_vectors: A 2D array of shape
        (num_concepts, n_features) where each column represents a given
        meaningful concept direction.
    :param float beta: A value used for zeroing dot products that are
        considered to be irrelevant. If a dot product is less than beta, then
        its score will be zero.
    :param float epsilon: A value used for numerical stability to avoid
        division by zero.
    :param int channels_axis: the channels axis in the input features. If not
        given, then we assume it is the last dimension always.

    :returns np.ndarray: A 2D matrix with shape (n_samples, n_concepts) where
        the (i, j)-th entry represents the score that the j-th concept assigned
        the i-th sample in `features`.
    """
    # First check that all the dimensions make sense
    assert features.shape[channels_axis] == concept_vectors.shape[-1], (
        f'Expected input to have {concept_vectors.shape[-1]} elements in its '
        f'channels axis (defined as axis {channels_axis}). '
        f'Instead, we found the input to have shape {features.shape}.'
    )
    # First normalize all concepts across their channels dimension
    concept_vectors_norm = np.linalg.norm(
        concept_vectors,
        axis=-1,
        keepdims=True,
    )
    concept_vectors = concept_vectors / (concept_vectors_norm + epsilon)

    # For simplicity, we will always move the channels dimension to the end
    if channels_axis not in [-1, len(features.shape) - 1]:
        # Then perform a transpose in here
        perm = list(range(len(features.shape)))
        perm[channels_axis] = len(features.shape) - 1
        perm[len(features.shape) - 1] = channels_axis
        features = np.transpose(features, perm)

    # And similarly, do the same for the input features
    x_norm = np.linalg.norm(
        features,
        axis=-1,
        keepdims=True,
    )

    x_norm = features / (x_norm + epsilon)

    # Compute the concept probabilities accordingly
    concept_prob = np.dot(features, concept_vectors.transpose())
    concept_prob_norm = np.dot(x_norm, concept_vectors.transpose())

    # Threshold scores accordingly using the normalized scores
    if beta is not None:
        concept_prob = concept_prob * (concept_prob_norm > beta)

    # And end by normalizing them
    norm = np.sum(concept_prob, axis=-1, keepdims=True)
    result = concept_prob / (norm + epsilon)

    # Finally, restore the shape of the output tensor if a transpose was
    # done at the beginning
    if channels_axis not in [-1, len(features.shape) - 1]:
        perm = list(range(len(features.shape)))
        perm[channels_axis] = len(features.shape) - 1
        perm[len(features.shape) - 1] = channels_axis
        result = np.transpose(result, perm)

    # And that's all folks
    return result


################################################################################
## Completeness Score Computation
################################################################################

def completeness_score(
    X,
    y,
    features_to_concepts_fn,
    concepts_to_labels_model,
    concept_vectors,
    task_loss,
    g_model=None,
    test_size=0.2,
    concept_score_fn=dot_prod_concept_score,
    predictor_train_kwags=None,
    g_optimizer='adam',
    acc_fn=sklearn.metrics.accuracy_score,
    channels_axis=-1,
):
    """
    Returns the completeness score for the given set of concept vectors
    `concept_vectors` using testing data `X` with labels `y`. This score
    is computed using Yeh et al.'s definition of a concept completeness
    score based on a model `features_to_concepts_fn`, which maps input features
    in the test data to a M-dimensional space, and a model
    `concepts_to_labels_model` which maps M-dimensional vectors to some
    probability distribution over classes in `y`.

    :param np.ndarray X: A tensor of testing samples that are in the domain of
        given function `features_to_concepts_fn` where the first dimension
        represents the number of test samples (which we call `n_samples`).
    :param np.ndarray y: A tensor of testing labels corresponding to matrix
        X whose first dimension must also be `n_samples`.
    :param Function[(np.ndarray), np.ndarray] features_to_concepts_fn: A
        function mapping batches of samples with the same dimensionality as
        X into some (n_samples, ..., M, ...)-dimensional vector space
        corresponding to the same vector space as that used for the given
        concept vectors. In this case, M is the channels dimension at location
        channels_axis.
    :param tf.keras.Model concepts_to_labels_model: An arbitrary Keras model
        which maps M-dimensional vectors (as those produced by calling the
        `features_to_concepts_fn` function on a batch of samples) into a
        probability distribution over labels in `y`.
    :param np.ndarray concept_vectors: A 2D array of shape (num_concepts, M)
        where each column represents a given meaningful concept direction.
    :param tf.keras.losses.Loss task_loss: The loss function one intends to
        minimize when mapping instances in `X` to labels in `y`.
    :param tf.keras.Model g_model: The model `g` we will train for mapping
        concept scores to the same M-dimensional space when computing
        the concept completeness score. If not given, then we will use a
        3-layered ReLU MLP with 500 hidden activations.
    :param float test_size: A value between 0 and 1 representing what percent
        of the (X, y) data will be used for testing the accuracy of our g_model
        (and the original model) when computing the completeness score. The
        rest of the data will be used for training our g_model.
    :param Function[(np.ndarray,List[np.ndarray]), np.ndarray] concept_score_fn:
        A function taking as an input a matrix of shape (n_samples, M),
        representing outputs produced by the `features_to_concepts_fn` function,
        and a list of`n_concepts` M-dimensional vectors, representing unit
        directions of meaningful concepts, and returning a vector with
        n_concepts concept scores. By default we use the normalized dot product
        scores.
    :param Dict[Any, Any] predictor_train_kwags: An optional set of parameters
        to pass to the g_model when trained for reconstructing the M-dimensional
        activations from their corresponding concept scores.
    :param tf.keras.optimizers.Optimizer g_optimizer: The optimizer used for
        training the g model for the reconstruction. By default we will use an
        ADAM optimizer.
    :param Function[(np.ndarray, np.ndarray), float] acc_fn: An accuracy
        function taking (true_labels, predicted_labels) and returning an
        accuracy value between 0 and 1.
    :param int channels_axis: The channels dimension axis of the output of the
        features_to_concepts function. If not given, then it is assumed to be
        the last dimension.

    :returns Tuple[float, tf.keras.Model]: A tuple (score, g_model) containing
        the computed completeness score together with the resulting trained
        g_model.
    """
    # Let's first start by splitting our data into a training and a testing
    # set
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
    )

    # Let's take a look at the intermediate activations we will be using
    phi_train = features_to_concepts_fn(X_train)
    scores_train = concept_score_fn(phi_train, concept_vectors)
    num_labels = len(set(y))

    # Compute some useful variables while also handling the default case for
    # the model we will optimize over
    num_concepts = concept_vectors.shape[0]
    num_hidden_acts = phi_train.shape[channels_axis]
    n_samples = X_train.shape[0]
    g_model = g_model or _get_default_model(
        num_concepts=num_concepts,
        num_hidden_acts=num_hidden_acts,
    )
    predictor_train_kwags = predictor_train_kwags or {
        'epochs': 50,
        'batch_size': min(16, n_samples),
        'verbose': 0,
    }

    # Construct a model that we can use for optimizing our g function
    # For this, we will first need to make sure that we set our concepts
    # to labels model so that we do not optimize over its parameters
    prev_trainable = concepts_to_labels_model.trainable
    concepts_to_labels_model.trainable = False
    f_prime_input = tf.keras.layers.Input(
        shape=scores_train.shape[1:],
        dtype=scores_train.dtype,
    )
    f_prime_output = concepts_to_labels_model(g_model(f_prime_input))
    f_prime_optimized = tf.keras.Model(
        f_prime_input,
        f_prime_output,
    )

    # Time to optimize it!
    f_prime_optimized.compile(
        optimizer=g_optimizer,
        loss=task_loss,
    )
    f_prime_optimized.fit(
        scores_train,
        y_train,
        **predictor_train_kwags,
    )

    # Don't forget to reconstruct the state of the concept to labels model
    concepts_to_labels_model.trainable = prev_trainable

    # Finally, compute the actual score by computing the accuracy of the
    # original concept-composable model
    phi_test = features_to_concepts_fn(X_test)
    random_pred_acc = 1 / num_labels
    f_preds = concepts_to_labels_model.predict(
        phi_test
    )
    f_acc = acc_fn(
        y_test,
        f_preds,
    )

    # And the accuracy of the model using the reconstruction from the
    # concept scores
    f_prime_preds = f_prime_optimized.predict(
        concept_score_fn(phi_test, concept_vectors)
    )
    f_prime_acc = acc_fn(
        y_test,
        f_prime_preds,
    )

    # That gives us everything we need
    if f_prime_acc == random_pred_acc:
        return 0, g_model
    completeness = (f_prime_acc - random_pred_acc) / (f_acc - random_pred_acc)
    return completeness, g_model


def direct_completeness_score(
    X,
    y,
    features_to_concepts_fn,
    concept_vectors,
    task_loss,
    g_model=None,
    test_size=0.2,
    concept_score_fn=dot_prod_concept_score,
    predictor_train_kwags=None,
    g_optimizer='adam',
    acc_fn=sklearn.metrics.accuracy_score,
    channels_axis=-1,
):
    """
    Returns the completeness score for the given set of concept vectors
    `concept_vectors` using testing data `X` with labels `y`. This score
    is computed as the predictive accuracy of a model trained to predict
    the labels using the concepts scores alone. It differs from the method
    above in that it does not require a pre-trained concept_to_labels map.

    :param np.ndarray X: A tensor of testing samples that are in the domain of
        given function `features_to_concepts_fn` where the first dimension
        represents the number of test samples (which we call `n_samples`).
    :param np.ndarray y: A tensor of testing labels corresponding to matrix
        X whose first dimension must also be `n_samples`.
    :param Function[(np.ndarray), np.ndarray] features_to_concepts_fn: A
        function mapping batches of samples with the same dimensionality as
        X into some (n_samples, ..., M, ...)-dimensional vector space
        corresponding to the same vector space as that used for the given
        concept vectors. In this case, M is the channels dimension at location
        channels_axis.
    :param np.ndarray concept_vectors: A 2D array of shape (num_concepts, M)
        where each column represents a given meaningful concept direction.
    :param tf.keras.losses.Loss task_loss: The loss function one intends to
        minimize when mapping instances in `X` to labels in `y`.
    :param tf.keras.Model g_model: The model `g` we will train for mapping
        concept scores to the space of labels when computing the concept
        completeness score. If not given, then we will use a 3-layered ReLU MLP
        with 500 hidden activations.
    :param float test_size: A value between 0 and 1 representing what percent
        of the (X, y) data will be used for testing the accuracy of our g_model
        (and the original model) when computing the completeness score. The
        rest of the data will be used for training our g_model.
    :param Function[(np.ndarray,List[np.ndarray]), np.ndarray] concept_score_fn:
        A function taking as an input a matrix of shape (n_samples, M),
        representing outputs produced by the `features_to_concepts_fn` function,
        and a list of`n_concepts` M-dimensional vectors, representing unit
        directions of meaningful concepts, and returning a vector with
        n_concepts concept scores. By default we use the normalized dot product
        scores.
    :param Dict[Any, Any] predictor_train_kwags: An optional set of parameters
        to pass to the g_model when trained for reconstructing the M-dimensional
        activations from their corresponding concept scores.
    :param tf.keras.optimizers.Optimizer g_optimizer: The optimizer used for
        training the g model for the reconstruction. By default we will use an
        ADAM optimizer.
    :param Function[(np.ndarray, np.ndarray), float] acc_fn: An accuracy
        function taking (true_labels, predicted_labels) and returning an
        accuracy value between 0 and 1.
    :param int channels_axis: The channels dimension axis of the output of the
        features_to_concepts function. If not given, then it is assumed to be
        the last dimension.

    :returns Tuple[float, tf.keras.Model]: A tuple (score, g_model) containing
        the computed completeness score together with the resulting trained
        g_model.
    """
    # Let's first start by splitting our data into a training and a testing
    # set
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
    )

    # Let's take a look at the intermediate activations we will be using
    phi_train = features_to_concepts_fn(X_train)
    scores_train = concept_score_fn(
        phi_train,
        concept_vectors,
    )
    num_labels = len(set(y))

    # Compute some useful variables while also handling the default case for
    # the model we will optimize over
    num_concepts = concept_vectors.shape[0]
    num_hidden_acts = phi_train.shape[channels_axis]
    n_samples = X_train.shape[0]
    g_model = g_model or _get_default_model(
        num_concepts=num_concepts,
        num_hidden_acts=num_labels if num_labels > 2 else 1,
    )
    predictor_train_kwags = predictor_train_kwags or {
        'epochs': 50,
        'batch_size': min(16, n_samples),
        'verbose': 0,
    }

    # Time to optimize it!
    g_model.compile(
        optimizer=g_optimizer,
        loss=task_loss,
    )
    g_model.fit(
        scores_train,
        y_train,
        **predictor_train_kwags,
    )

    # Finally, compute the actual score by computing the accuracy of predicting
    # the output labels using only the concept scores
    phi_test = features_to_concepts_fn(X_test)
    from_concepts_preds = g_model.predict(
        concept_score_fn(phi_test, concept_vectors)
    )
    from_concepts_acc = acc_fn(
        y_test,
        from_concepts_preds,
    )

    # That gives us everything we need
    return from_concepts_acc, g_model
