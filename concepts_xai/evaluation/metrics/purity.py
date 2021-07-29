'''
Metrics to measure concept purity based on the definition of purity by
Mahinpei et al.'s "Promises and Pitfalls of Black-Box Concept Learning Models"
(found at https://arxiv.org/abs/2106.13314).
'''

import numpy as np
import sklearn
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.engine import data_adapter


################################################################################
## Purity Matrix Computation
################################################################################

def concept_purity_matrix(
    c_soft,
    c_true,
    predictor_model_fn=None,
    predictor_train_kwags=None,
    test_size=0.2,
):
    """
    Computes a concept purity matrix where the (i,j)-th entry represents the
    predictive accuracy of a classifier trained to use the i-th concept's soft
    labels (as given by c_soft_train) to predict the ground truth value of the
    j-th concept.

    This process is informally defined only for binary concepts by Mahinpei et
    al.'s in "Promises and Pitfalls of Black-Box Concept Learning Models".
    Nevertheless, this method supports both binary concepts (given as a 2D
    matrix in c_soft) or categorical concepts (given by a list of 2D matrices
    in argument c_soft).

    :param Or[np.ndarray, List[np.ndarray]] c_soft: Predicted set of "soft"
        concepts by a concept encoder model applied to the testing data. This
        argument must be an np.ndarray with shape (n_samples, n_concepts) if
        all concepts are binary. Otherwise, it is expected to be a list of
        np.ndarray where the i-th element in the list is an array with shape
        (n_samples, <cardinality of set of labels for i-th concept>)
        representing a distribution over all possible labels for the i-th
        concept.
    :param np.ndarray c_true: Ground truth concept values in one-to-one
        correspondence with concepts in c_soft. Shape must be
        (n_samples, n_concepts).
    :param Function[(int, int), sklearn-like Estimator] predictor_model_fn: A
        function generator that takes as an argument two values, the number of
        classes for the input concept and the number of classes for the output
        target concept, respectively, and produces an sklearn-like Estimator
        which one can train for predicting a concept given another concept's
        soft concept values. If not given then we will use a 3-layer ReLU MLP
        as our predictor.
    :param Dict[Any, Any] predictor_train_kwags: optional arguments to pass
        the estimator being when calling its `fit` method.
    :param float test_size: A value in [0, 1] indicating the fraction of the
        given data that will be used to evaluate the trained concept-based
        classifier. The rest of the data will be used for training said
        classifier.

    :return np.ndarray: a matrix with shape (n_concepts, n_concepts)
        where the (i,j)-th entry specifies the testing AUC of using the i-th
        concept soft labels to predict the j-th concept.
    """
    # Start by handling default arguments
    predictor_train_kwags = predictor_train_kwags or {}

    # Check that their rank is the expected one
    assert len(c_true.shape) == 2, (
        f'Expected testing concept predictions to be a matrix with shape '
        f'(n_samples, n_concepts) but instead got a matrix with shape '
        f'{c_true.shape}'
    )

    # Construct a list concept_label_cardinality that maps a concept to the
    # cardinality of its label set as specified by the testing data
    (n_samples, n_concepts) = c_true.shape

    if isinstance(c_soft, np.ndarray):
        # Then, all concepts must be binary
        assert c_soft.shape == c_true.shape, (
            f'Expected a many test soft-concepts as ground truth test '
            f'concepts. Instead got {c_soft.shape} soft-concepts '
            f'and {c_true.shape} ground truth test concepts.'
        )
        concept_label_cardinality = [2 for _ in range(n_concepts)]
        # And for simplicity and consistency, we will rewrite c_soft as a
        # list such that i-th entry contains an array with shape (n_samples, 1)
        # indicating the probability that concept i is activated for all
        # test samples.
        new_c_soft = [None for _ in range(n_concepts)]
        for i in range(n_concepts):
            new_c_soft[i] = np.expand_dims(c_soft[:, i], axis=-1)
        c_soft = new_c_soft
    else:
        # Else, time to infer these values from the given list of soft
        # labels
        assert isinstance(c_soft, list), (
            f'c_soft must be passed as either a list or a np.ndarray. '
            f'Instead we got an instance of "{type(c_soft).__name__}".'
        )
        concept_label_cardinality = [None for _ in range(n_concepts)]
        for i, soft_labels in enumerate(c_soft):
            concept_label_cardinality[i] = max(soft_labels.shape[-1], 2)
            assert soft_labels.shape[0] == c_true.shape[0], (
                f"For concept {i}'s soft labels, we expected "
                f"{c_true.shape[0]} samples as we were given that many "
                f"in the ground-truth array. Instead we found "
                f"{soft_labels.shape[0]} samples."
            )

    # Handle the default parameters for both the generating function and
    # the concept label cardinality
    if predictor_model_fn is None:
        # Then by default we will use a simple MLP classifier with one hidden
        # ReLU layer with 32 units in it
        def predictor_model_fn(
            input_concept_classes=2,
            output_concept_classes=2,
        ):
            estimator = tf.keras.models.Sequential()
            estimator.add(tf.keras.layers.Dense(
                32,
                input_dim=(
                    input_concept_classes if input_concept_classes > 2 else 1
                ),
                activation='relu'
            ))
            estimator.add(tf.keras.layers.Dense(
                output_concept_classes if output_concept_classes > 2 else 1,
                # We will merge the activation into the loss for numerical
                # stability
                activation=None,
            ))
            estimator.compile(
                # Use ADAM optimizer by default
                optimizer='adam',
                # Note: we assume labels come without a one-hot-encoding in the
                #       case when the concepts are categorical.
                loss=(
                    tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True
                    ) if output_concept_classes > 2 else
                    tf.keras.losses.BinaryCrossentropy(
                        from_logits=True,
                    )
                ),
            )
            return estimator
        predictor_train_kwags = predictor_train_kwags or {
            'epochs': 25,
            'batch_size': min(16, n_samples),
            'verbose': 0,
        }

    # Time to start formulating our resulting matrix
    result = np.zeros((n_concepts, n_concepts), dtype=np.float32)

    # Split our test data into two subsets as we will need to train
    # a classifier and then use that trained classifier in the remainder of the
    # data for computing our scores
    train_indexes, test_indexes = train_test_split(
        list(range(n_samples)),
        test_size=test_size,
    )

    for concept_i in range(n_concepts):
        # Construct a test and training set of features for this concept
        concept_soft_train_x = c_soft[concept_i][train_indexes, :]
        concept_soft_test_x = c_soft[concept_i][test_indexes, :]

        for concept_j in range(n_concepts):
            # Let's populate the (i,j)-th entry of our matrix by first training
            # a classifier to predict the ground truth value of concept j using
            # the soft-concept labels for concept i.

            # Construct a new estimator for performing this prediction
            estimator = predictor_model_fn(
                concept_label_cardinality[concept_i],
                concept_label_cardinality[concept_j]
            )
            # Train it
            estimator.fit(
                concept_soft_train_x,
                c_true[train_indexes, concept_j:(concept_j + 1)],
                **predictor_train_kwags,
            )

            # Compute the AUC of this classifier on the test data
            auc = sklearn.metrics.roc_auc_score(
                c_true[test_indexes, concept_j],
                estimator.predict(concept_soft_test_x),
                multi_class='ovo',
            )

            # Finally, time to populate the actual entry of our resulting
            # matrix
            result[concept_i, concept_j] = auc

    # And that's all folks
    return result


def encoder_concept_purity_matrix(
    encoder_model,
    features,
    concepts,
    predictor_model_fn=None,
    predictor_train_kwags=None,
    test_size=0.2,
):
    """
    Computes a concept purity matrix where the (i,j)-th entry represents the
    predictive accuracy of a classifier trained to use the i-th concept's soft
    labels (as given by the encoder model) to predict the ground truth value of
    the j-th concept.

    This process is informally defined only for binary concepts by Mahinpei et
    al.'s in "Promises and Pitfalls of Black-Box Concept Learning Models".
    Nevertheless, this method supports both binary concepts (given as a 2D
    matrix output when using the encoder's predict method) or categorical
    concepts (given as a list of 2D matrices when using the encoder's predict
    method).

    :param skelearn-like Estimator encoder_model: An encoder estimator capable
        of extracting concepts from a set of features. This estimator may
        produce a vector of binary concept probabilities for each sample (i.e.,
        in the case of all concepts being binary) or a list of vectors
        representing probability distributions over the labels for each concept
        (i.e., in the case of one or more concepts being categorical).
    :param np.ndarray features: An array of testing samples with shape
        (n_samples, n_features) used to compute the purity matrix.
    :param np.ndarray concepts: Ground truth concept values in one-to-one
        correspondence with samples in features. Shape must be
        (n_samples, n_concepts).
    :param Function[(int, int), sklearn-like Estimator] predictor_model_fn: A
        function generator that takes as an argument two values, the number of
        classes for the input concept and the number of classes for the output
        target concept, respectively, and produces an sklearn-like Estimator
        which one can train for predicting a concept given another concept's
        soft concept values. If not given then we will use a 3-layer ReLU MLP
        as our predictor.
    :param Dict[Any, Any] predictor_train_kwags: optional arguments to pass
        the estimator when calling its `fit` method.
    :param float test_size: A value in [0, 1] indicating the fraction of the
        given data that will be used to evaluate the trained concept-based
        classifier. The rest of the data will be used for training said
        classifier.

    :return np.ndarray: a matrix with shape (n_concepts, n_concepts)
        where the (i,j)-th entry specifies the testing AUC of using the i-th
        concept soft labels to predict the j-th concept.
    """
    # Simply use the concept purity matrix computation defined above when given
    # soft concepts as computed by the encoder model
    return concept_purity_matrix(
        c_soft=encoder_model.predict(features),
        c_true=concepts,
        predictor_model_fn=predictor_model_fn,
        predictor_train_kwags=predictor_train_kwags,
        test_size=test_size,
    )


def oracle_purity_matrix(
    concepts,
    concept_label_cardinality=None,
    predictor_model_fn=None,
    predictor_train_kwags=None,
    test_size=0.2,
):
    """
    Computes an oracle's concept purity matrix where the (i,j)-th entry
    represents the predictive accuracy of a classifier trained to use the i-th
    concept (ground truth) to predict the ground truth value of the j-th
    concept.

    :param np.ndarray concepts: Ground truth concept values. Shape must be
        (n_samples, n_concepts).
    :param List[int] concept_label_cardinality: If given, then this is a list
        of integers such that its i-th index contains the number of classes
        that the it-th concept may take. If not given, then we will assume that
        all concepts are binary (i.e., concept_label_cardinality[i] = 2 for all
        i).
    :param Function[(int, int), sklearn-like Estimator] predictor_model_fn: A
        function generator that takes as an argument two values, the number of
        classes for the input concept and the number of classes for the output
        target concept, respectively, and produces an sklearn-like Estimator
        which one can train for predicting a concept given another concept's
        concept's value. If not given then we will use a 3-layer ReLU MLP
        as our predictor.
    :param Dict[Any, Any] predictor_train_kwags: optional arguments to pass
        the estimator when calling its `fit` method.
    :param float test_size: A value in [0, 1] indicating the fraction of the
        given data that will be used to evaluate the trained concept-based
        classifier. The rest of the data will be used for training said
        classifier.

    :return np.ndarray: a matrix with shape (n_concepts, n_concepts)
        where the (i,j)-th entry specifies the testing AUC of using the i-th
        concept label to predict the j-th concept.
    """
    # Check that their rank is the expected one
    assert len(concepts.shape) == 2, (
        f'Expected testing concept predictions to be a matrix with shape '
        f'(n_samples, n_concepts) but instead got a matrix with shape '
        f'{concepts.shape}'
    )

    (n_samples, n_concepts) = concepts.shape

    # Handle the default parameters for both the generating function and
    # the concept label cardinality
    concept_label_cardinality = concept_label_cardinality or [
        2 for _ in range(n_concepts)
    ]
    if predictor_model_fn is None:
        # Then by default we will use a simple MLP classifier with one hidden
        # ReLU layer with 32 units in it
        def predictor_model_fn(
            input_concept_classes=2,
            output_concept_classes=2,
        ):
            estimator = tf.keras.models.Sequential()
            estimator.add(tf.keras.layers.Dense(
                32,
                input_dim=(
                    input_concept_classes if input_concept_classes > 2 else 1
                ),
                activation='relu'
            ))
            estimator.add(tf.keras.layers.Dense(
                output_concept_classes if output_concept_classes > 2 else 1,
                # We will merge the activation into the loss for numerical
                # stability
                activation=None,
            ))
            estimator.compile(
                # Use ADAM optimizer by default
                optimizer='adam',
                # Note: we assume labels come without a one-hot-encoding in the
                #       case when the concepts are categorical.
                loss=(
                    tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True
                    ) if output_concept_classes > 2 else
                    tf.keras.losses.BinaryCrossentropy(
                        from_logits=True,
                    )
                ),
            )
            return estimator
        predictor_train_kwags = predictor_train_kwags or {
            'epochs': 25,
            'batch_size': min(16, n_samples),
            'verbose': 0,
        }

    # Time to start formulating our resulting matrix
    result = np.zeros((n_concepts, n_concepts), dtype=np.float32)

    # Split our test data into two subsets as we will need to train
    # a classifier and then use that trained classifier in the remainder of the
    # data for computing our scores
    train_indexes, test_indexes = train_test_split(
        list(range(n_samples)),
        test_size=test_size,
    )

    for concept_i in range(n_concepts):
        # Construct a test and training set of features for this concept
        concept_train_x = concepts[train_indexes, concept_i:(concept_i + 1)]
        concept_test_x = concepts[test_indexes, concept_i:(concept_i + 1)]

        for concept_j in range(n_concepts):
            # Let's populate the (i,j)-th entry of our matrix by first training
            # a classifier to predict the ground truth value of concept j using
            # the ground truth labels for concept i.

            # Construct a new estimator for performing this prediction
            estimator = predictor_model_fn(
                concept_label_cardinality[concept_i],
                concept_label_cardinality[concept_j]
            )
            # Train it
            estimator.fit(
                concept_train_x,
                concepts[train_indexes, concept_j:(concept_j + 1)],
                **predictor_train_kwags,
            )

            # Compute the AUC of this classifier on the test data
            auc = sklearn.metrics.roc_auc_score(
                concepts[test_indexes, concept_j],
                estimator.predict(concept_test_x),
                multi_class='ovo',
            )

            # Finally, time to populate the actual entry of our resulting
            # matrix
            result[concept_i, concept_j] = auc

    # And that's all folks
    return result


################################################################################
## Purity Metrics
################################################################################

def norm_purity_score(
    c_soft,
    c_true,
    predictor_model_fn=None,
    predictor_train_kwags=None,
    test_size=0.2,
    norm_fn=lambda x: np.linalg.norm(x, ord='fro'),
):
    """
    Returns the purity score of the given soft concept representations `c_soft`
    with respect to their corresponding ground truth concepts `c_true`. This
    value is higher if concepts encode unnecessary information from other
    concepts in their soft representation and lower otherwise. If zero, then
    all soft concept labels are considered to be "pure".

    We compute this metric by calculating the norm of the absolute difference
    between the purity matrix derived from the soft concepts and the purity
    matrix derived from an oracle model. This oracle model is trained using
    the ground truth labels instead of the soft labels and may capture trivial
    relationships between different concept labels.

    :param Or[np.ndarray, List[np.ndarray]] c_soft: Predicted set of "soft"
        concepts by a concept encoder model applied to the testing data. This
        argument must be an np.ndarray with shape (n_samples, n_concepts) if
        all concepts are binary. Otherwise, it is expected to be a list of
        np.ndarray where the i-th element in the list is an array with shape
        (n_samples, <cardinality of set of labels for i-th concept>)
        representing a distribution over all possible labels for the i-th
        concept.
    :param np.ndarray c_true: Ground truth concept values in one-to-one
        correspondence with concepts in c_soft. Shape must be
        (n_samples, n_concepts).
    :param Function[(int, int), sklearn-like Estimator] predictor_model_fn: A
        function generator that takes as an argument two values, the number of
        classes for the input concept and the number of classes for the output
        target concept, respectively, and produces an sklearn-like Estimator
        which one can train for predicting a concept given another concept's
        soft concept values. If not given then we will use a 3-layer ReLU MLP
        as our predictor.
    :param Dict[Any, Any] predictor_train_kwags: optional arguments to pass
        the estimator being when calling its `fit` method.
    :param float test_size: A value in [0, 1] indicating the fraction of the
        given data that will be used to evaluate the trained concept-based
        classifier. The rest of the data will be used for training said
        classifier.
    :param Function[(np.ndarray), float] norm_fn: A norm function applicable to
        a 2D numpy matrix representing the absolute difference between the
        oracle purity score matrix and the predicted purity score matrix. If not
        given then we will use the 2D Frobenius norm.

    :returns float: A non-negative float representing the degree to which
        individual concepts in the given bottleneck encode unnecessary
        information for other concepts. Higher values mean more impurity and
        the concepts are considered to be pure if the returned value is 0.
    """
    # First compute the predictor soft-concept purity matrix
    pred_matrix = concept_purity_matrix(
        c_soft=c_soft,
        c_true=c_true,
        predictor_model_fn=predictor_model_fn,
        predictor_train_kwags=predictor_train_kwags,
        test_size=test_size,
    )

    # Now infer the concept_label_cardinality vector from the given soft labels
    (n_samples, n_concepts) = c_true.shape
    if isinstance(c_soft, np.ndarray):
        # Then, all concepts must be binary
        concept_label_cardinality = [2 for _ in range(n_concepts)]
    else:
        # Else, time to infer these values from the given list of soft
        # labels
        concept_label_cardinality = [None for _ in range(n_concepts)]
        for i, soft_labels in enumerate(c_soft):
            concept_label_cardinality[i] = max(soft_labels.shape[-1], 2)

    # Compute the oracle's purity matrix
    oracle_matrix = oracle_purity_matrix(
        concepts=c_true,
        concept_label_cardinality=concept_label_cardinality,
        predictor_model_fn=predictor_model_fn,
        predictor_train_kwags=predictor_train_kwags,
        test_size=test_size,
    )

    # Finally, compute the norm of the absolute difference between the two
    # matrices
    return norm_fn(np.abs(oracle_matrix - pred_matrix))


def encoder_norm_purity_score(
    encoder_model,
    features,
    concepts,
    predictor_model_fn=None,
    predictor_train_kwags=None,
    test_size=0.2,
    norm_fn=lambda x: np.linalg.norm(x, ord='fro'),
):
    """
    Returns the purity score of the concept representations generated by
    `encoder_model` when given `features` with respect to their corresponding
    ground truth concepts `concepts`. This value is higher if concepts encode
    unnecessary information from other concepts in their soft representation and
    lower otherwise. If zero, then all soft concept labels are considered to be
    "pure".

    We compute this metric by calculating the norm of the absolute difference
    between the purity matrix derived from the soft concepts and the purity
    matrix derived from an oracle model. This oracle model is trained using
    the ground truth labels instead of the soft labels and may capture trivial
    relationships between different concept labels.

    :param skelearn-like Estimator encoder_model: An encoder estimator capable
        of extracting concepts from a set of features. This estimator may
        produce a vector of binary concept probabilities for each sample (i.e.,
        in the case of all concepts being binary) or a list of vectors
        representing probability distributions over the labels for each concept
        (i.e., in the case of one or more concepts being categorical).
    :param np.ndarray features: An array of testing samples with shape
        (n_samples, n_features) used to compute the purity matrix.
    :param np.ndarray concepts: Ground truth concept values in one-to-one
        correspondence with samples in features. Shape must be
        (n_samples, n_concepts).
    :param Function[(int, int), sklearn-like Estimator] predictor_model_fn: A
        function generator that takes as an argument two values, the number of
        classes for the input concept and the number of classes for the output
        target concept, respectively, and produces an sklearn-like Estimator
        which one can train for predicting a concept given another concept's
        soft concept values. If not given then we will use a 3-layer ReLU MLP
        as our predictor.
    :param Dict[Any, Any] predictor_train_kwags: optional arguments to pass
        the estimator when calling its `fit` method.
    :param float test_size: A value in [0, 1] indicating the fraction of the
        given data that will be used to evaluate the trained concept-based
        classifier. The rest of the data will be used for training said
        classifier.
    :param Function[(np.ndarray), float] norm_fn: A norm function applicable to
        a 2D numpy matrix representing the absolute difference between the
        oracle purity score matrix and the predicted purity score matrix. If not
        given then we will use the 2D Frobenius norm.


    :returns float: A non-negative float representing the degree to which
        individual concepts in the given bottleneck encode unnecessary
        information for other concepts. Higher values mean more impurity and
        the concepts are considered to be pure if the returned value is 0.
    """
    # Simply use the concept purity metric defined above when given
    # soft concepts as computed by the encoder model
    return norm_purity_score(
        c_soft=encoder_model.predict(features),
        c_true=concepts,
        predictor_model_fn=predictor_model_fn,
        predictor_train_kwags=predictor_train_kwags,
        test_size=test_size,
        norm_fn=norm_fn,
    )
