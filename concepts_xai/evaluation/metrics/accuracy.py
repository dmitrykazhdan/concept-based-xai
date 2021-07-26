from sklearn.metrics import accuracy_score, f1_score


def compute_accuracies(c_true, c_pred):
    '''
    Compute the accuracy scores per concept
    :param c_true: Numpy array of (n_samples, n_concepts) of ground-truth concept values
    :param c_pred: Numpy array of (n_samples, n_concepts) of predicted concept values
    :return: Accuracies for all samples, per concept
    '''
    n_concepts = c_true.shape[1]
    accuracies = [accuracy_score(c_true[:, i], c_pred[:, i]) for i in range(n_concepts)]
    return accuracies



