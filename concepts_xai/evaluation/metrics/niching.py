import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def niche_completeness(c_pred, y_true, predictor_model):
    '''
    Computes the niche completeness score for the downstream task
    :param c_pred: Concept data predictions, numpy array of shape (n_samples, n_concepts)
    :param y_true: Ground-truth task label data, numpy array of shape (n_samples, n_tasks)
    :param predictor_model: sklearn model to use for predicting the task labels from the concept data
    :return: Accuracy of predictor_model, evaluated on niches obtained from the provided concept and label data
    '''
    n_tasks = y_true.shape[1]

    # train the model
    c_train, c_test, y_train, y_test = train_test_split(c_pred, y_true)
    predictor_model.fit(c_train, y_train)

    # find concept niches
    niches = _niche_finding(c_train, y_train)

    # compute niche completeness for each task
    niche_completeness_list = []
    for task in range(n_tasks):
        # find niche
        niche = np.zeros_like(c_test)
        niche[:, niches[:, task] > 0] = niche[:, niches[:, task] > 0]

        # compute task predictions
        y_pred_niche = predictor_model.predict(niche)

        # compute niche accuracy
        # the higher the better (high predictive power of the niche)
        accuracy_niche = accuracy_score(y_test[:, task], y_pred_niche[:, task])
        niche_completeness_list.append(accuracy_niche)

    return np.mean(niche_completeness_list), niche_completeness_list


def niche_completeness_ratio(c_pred, y_true, predictor_model):
    '''
    Computes the niche completeness ratio for the downstream task
    :param c_pred: Concept data predictions, numpy array of shape (n_samples, n_concepts)
    :param y_true: Ground-truth task label data, numpy array of shape (n_samples, n_tasks)
    :param predictor_model: sklearn model to use for predicting the task labels from the concept data
    :return: Accuracy ratio between the accuracy of predictor_model evaluated on niches and
             the accuracy of predictor_model evaluated on all concepts
    '''
    n_tasks = y_true.shape[1]

    # train the model
    c_train, c_test, y_train, y_test = train_test_split(c_pred, y_true)
    predictor_model.fit(c_train, y_train)
    y_pred_test = predictor_model.predict(c_test)

    # find concept niches
    niches = _niche_finding(c_train, y_train)

    # compute niche completeness for each task
    niche_completeness_list = []
    for task in range(n_tasks):
        # find niche
        niche = np.zeros_like(c_test)
        niche[:, niches[:, task] > 0] = niche[:, niches[:, task] > 0]

        # compute task predictions
        y_pred_niche = predictor_model.predict(niche)

        # compute accuracies
        accuracy_base = accuracy_score(y_test[:, task], y_pred_test[:, task])
        accuracy_niche = accuracy_score(y_test[:, task], y_pred_niche[:, task])

        # compute the accuracy ratio of the niche w.r.t. the baseline (full concept bottleneck)
        # the higher the better (high predictive power of the niche)
        niche_completeness = accuracy_niche / accuracy_base
        niche_completeness_list.append(niche_completeness)

    return np.mean(niche_completeness_list), niche_completeness_list


def niche_purity(c_pred, y_true, predictor_model):
    '''
    Computes the niche purity score for the downstream task
    :param c_pred: Concept data predictions, numpy array of shape (n_samples, n_concepts)
    :param y_true: Ground-truth task label data, numpy array of shape (n_samples, n_tasks)
    :param predictor_model: sklearn model to use for predicting the task labels from the concept data
    :return: Accuracy ratio between the accuracy of predictor_model evaluated on concepts outside niches and
             the accuracy of predictor_model evaluated on concepts inside niches
    '''
    n_tasks = y_true.shape[1]

    # train the model
    c_train, c_test, y_train, y_test = train_test_split(c_pred, y_true)
    predictor_model.fit(c_train, y_train)

    # find concept niches
    niches = _niche_finding(c_train, y_train)

    # compute niche completeness for each task
    niche_purity_list = []
    for task in range(n_tasks):
        # find niche
        niche = np.zeros_like(c_test)
        niche[:, niches[:, task] > 0] = niche[:, niches[:, task] > 0]

        # find concepts outside the niche
        niche_out = np.zeros_like(c_test)
        niche_out[:, niches[:, task] <= 0] = niche_out[:, niches[:, task] <= 0]

        # compute task predictions
        y_pred_niche = predictor_model.predict(niche)
        y_pred_niche_out = predictor_model.predict(niche_out)

        # compute accuracies
        accuracy_niche = accuracy_score(y_test[:, task], y_pred_niche[:, task])
        accuracy_niche_out = accuracy_score(y_test[:, task], y_pred_niche_out[:, task])

        # compute the ratio of:
        # - the task accuracy obtained using concepts OUTSIDE the niche w.r.t.
        # - the task accuracy obtained using concepts INSIDE the niche
        # the lower the better (less information leakage among niches)
        niche_purity = accuracy_niche_out / accuracy_niche
        niche_purity_list.append(niche_purity)

    return np.mean(niche_purity_list), niche_purity_list


def _niche_finding(c, y, method = 'corr'):
    n_concepts = c.shape[1]
    if method == 'corr':
        niches = (np.corrcoef(np.hstack([c, y]).T) > 0)[:n_concepts, n_concepts:]
    else:
        niches = (np.corrcoef(np.hstack([c, y]).T) > 0)[:n_concepts, n_concepts:]
    return niches
