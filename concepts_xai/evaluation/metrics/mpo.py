import numpy as np

def total_mispredictions_fn(c_true, c_pred):
    '''
    Count the total number of mispredictions
    :param c_true: ground-truth concept values
    :param c_pred: predicted concept values
    :return: Number of elements i where c_true[i] != c_pred[i]
    '''

    n_errs = np.sum(c_true != c_pred)
    return n_errs


def compute_MPO(c_true, c_pred, err_fn=total_mispredictions_fn):
    '''
    Implementation of the (M)is-(P)rediction (O)verlap (MPO) metric from the
    CME paper https://arxiv.org/pdf/2010.13233.pdf
    Given a set of predicted concept values, MPO computes the fraction of samples in the test set,
    that have at least m relevant concepts predicted incorrectly

    :param c_true: ground truth concept data, numpy array of shape (n_samples, n_concepts)
    :param c_pred: predicted concept data, numpy array of shape (n_samples, n_concepts)
    :param err_fn: function for computing the error on one single sample of c_test and c_pred (e.g. if you want
                   to ignore certain concepts, or weight concept errors differently).
                   Should be a function taking two arguments of shape (n_concepts), and returning a scalar value.
                   Defaults to computing the total number of mispredictions.
    :return: MPO metric values computed from c_test and c_pred using err_fn,
             with m ranging from 0 to n_concepts
    '''

    # Apply error function over all samples
    err_vals = np.array([err_fn(c_true[i], c_pred[i]) for i in range(c_true.shape[0])])

    # Compute MPO values for m ranging from 0 to n_concepts
    n_concepts = c_true.shape[1]
    mpo_vals = []
    for i in range(n_concepts):
        # Compute number of samples with at least i incorrect concept predictions
        n_incorrect = (err_vals >= i).astype(np.int)

        # Compute % of these samples from total
        metric = (np.sum(n_incorrect) / c_true.shape[0])
        mpo_vals.append(metric)

    return np.array(mpo_vals)


