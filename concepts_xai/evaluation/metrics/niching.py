import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif


def niche_completeness(c_pred, y_true, predictor_model, niches):
    '''
    Computes the niche completeness score for the downstream task
    :param c_pred: Concept data predictions, numpy array of shape (n_samples, n_concepts)
    :param y_true: Ground-truth task label data, numpy array of shape (n_samples, n_tasks)
    :param predictor_model: trained decoder model to use for predicting the task labels from the concept data
    :return: Accuracy of predictor_model, evaluated on niches obtained from the provided concept and label data
    '''
    n_tasks = y_true.shape[1]

    # compute niche completeness for each task
    niche_completeness_list, y_pred_list = [], []
    for task in range(n_tasks):
        # find niche
        niche = np.zeros_like(c_pred)
        niche[:, niches[:, task] > 0] = c_pred[:, niches[:, task] > 0]

        # compute task predictions
        y_pred_niche = predictor_model.predict(niche)
        if predictor_model.__class__.__name__ == 'Sequential':
            # get class labels from logits
            y_pred_niche = y_pred_niche > 0
        elif len(y_pred_niche.shape) == 1:
            y_pred_niche = y_pred_niche[:, np.newaxis]

        # compute niche accuracy
        # the higher the better (high predictive power of the niche)
        accuracy_niche = accuracy_score(y_true[:, task], y_pred_niche[:, task])
        niche_completeness_list.append(accuracy_niche)

        y_pred_list.append(y_pred_niche[:, task])

    y_preds = np.vstack(y_pred_list)
    auc = roc_auc_score(y_true, y_preds, multi_class='ovo')

    result = {
        'auc_purity': auc,
        'niche_completeness_mean': np.mean(niche_completeness_list),
        'niche_completeness': niche_completeness_list,
    }
    return result


def niche_completeness_ratio(c_pred, y_true, predictor_model, niches):
    '''
    Computes the niche completeness ratio for the downstream task
    :param c_pred: Concept data predictions, numpy array of shape (n_samples, n_concepts)
    :param y_true: Ground-truth task label data, numpy array of shape (n_samples, n_tasks)
    :param predictor_model: sklearn model to use for predicting the task labels from the concept data
    :return: Accuracy ratio between the accuracy of predictor_model evaluated on niches and
             the accuracy of predictor_model evaluated on all concepts
    '''
    n_tasks = y_true.shape[1]

    y_pred_test = predictor_model.predict(c_pred)
    if predictor_model.__class__.__name__ == 'Sequential':
        # get class labels from logits
        y_pred_test = y_pred_test > 0
    elif len(y_pred_test.shape) == 1:
        y_pred_test = y_pred_test[:, np.newaxis]

    # compute niche completeness for each task
    niche_completeness_list = []
    for task in range(n_tasks):
        # find niche
        niche = np.zeros_like(c_pred)
        niche[:, niches[:, task] > 0] = c_pred[:, niches[:, task] > 0]

        # compute task predictions
        y_pred_niche = predictor_model.predict(niche)
        if predictor_model.__class__.__name__ == 'Sequential':
            # get class labels from logits
            y_pred_niche = y_pred_niche > 0
        elif len(y_pred_niche.shape) == 1:
            y_pred_niche = y_pred_niche[:, np.newaxis]

        # compute accuracies
        accuracy_base = accuracy_score(y_true[:, task], y_pred_test[:, task])
        accuracy_niche = accuracy_score(y_true[:, task], y_pred_niche[:, task])

        # compute the accuracy ratio of the niche w.r.t. the baseline (full concept bottleneck)
        # the higher the better (high predictive power of the niche)
        niche_completeness = accuracy_niche / accuracy_base
        niche_completeness_list.append(niche_completeness)

    result = {
        'niche_completeness_ratio_mean': np.mean(niche_completeness_list),
        'niche_completeness_ratio': niche_completeness_list,
    }
    return result


def niche_purity(c_pred, y_true, predictor_model, niches):
    '''
    Computes the niche purity score for the downstream task
    :param c_pred: Concept data predictions, numpy array of shape (n_samples, n_concepts)
    :param y_true: Ground-truth task label data, numpy array of shape (n_samples, n_tasks)
    :param predictor_model: sklearn model to use for predicting the task labels from the concept data
    :return: Accuracy ratio between the accuracy of predictor_model evaluated on concepts outside niches and
             the accuracy of predictor_model evaluated on concepts inside niches
    '''
    n_tasks = y_true.shape[1]

    # compute niche completeness for each task
    niche_purity_list = []
    niche_impurity_list = []
    y_pred_list = []
    for task in range(n_tasks):
        # find niche
        niche = np.zeros_like(c_pred)
        niche[:, niches[:, task] > 0] = c_pred[:, niches[:, task] > 0]

        # find concepts outside the niche
        niche_out = np.zeros_like(c_pred)
        niche_out[:, niches[:, task] <= 0] = c_pred[:, niches[:, task] <= 0]

        # compute task predictions
        y_pred_niche = predictor_model.predict(niche)
        y_pred_niche_out = predictor_model.predict(niche_out)
        if predictor_model.__class__.__name__ == 'Sequential':
            # get class labels from logits
            y_pred_niche = y_pred_niche > 0
            y_pred_niche_out = y_pred_niche_out > 0
        elif len(y_pred_niche.shape) == 1:
            y_pred_niche = y_pred_niche[:, np.newaxis]
            y_pred_niche_out = y_pred_niche_out[:, np.newaxis]

        # compute purity scores
        # niche purity: the task accuracy obtained using concepts INSIDE the niche
        niche_purity = accuracy_score(y_true[:, task], y_pred_niche[:, task])
        # niche impurity: the task accuracy obtained using concepts OUTSIDE the niche w.r.t.
        niche_impurity = accuracy_score(y_true[:, task], y_pred_niche_out[:, task])

        niche_purity_list.append(niche_purity)
        niche_impurity_list.append(niche_impurity)

        y_pred_list.append(y_pred_niche_out[:, task])

    y_preds = np.vstack(y_pred_list)
    auc = roc_auc_score(y_true, y_preds, multi_class='ovo')

    result = {
        'auc_impurity': auc,
        'niche_purity_mean': np.mean(niche_purity_list),
        'niche_impurity_mean': np.mean(niche_impurity_list),
        'niche_purity': niche_purity_list,
        'niche_impurity': niche_impurity_list,
    }
    return result


def niche_finding(c, y, mode='mi', threshold=0.5):
    n_concepts = c.shape[1]
    if mode == 'corr':
        corrm = np.corrcoef(np.hstack([c, y]).T)
        niching_matrix = corrm[:n_concepts, n_concepts:]
        niches = np.abs(niching_matrix) > threshold
    elif mode == 'mi':
        nm = []
        for yj in y.T:
            mi = mutual_info_classif(c, yj)
            nm.append(mi)
        nm = np.vstack(nm).T
        niching_matrix = nm / np.max(nm)
        niches = niching_matrix > threshold
    else:
        return None, None, None

    return niches, niching_matrix
