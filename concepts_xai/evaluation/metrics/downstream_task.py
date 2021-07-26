from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def compute_downstream_task(c_pred, y_true, predictor_model):
    '''
    Computes the accuracy score for the downstream task
    :param c_pred: Concept data predictions, numpy array of shape (n_samples, n_concepts)
    :param y_true: Ground-truth task label data, numpy array of shape (n_samples
    :param predictor_model: sklearn model to use for predicting the task labels from the concept data
    :return: Accuracy of predictor_model, trained and evaluated on the provided concept and label data
    '''
    c_train, c_test, y_train, y_test = train_test_split(c_pred, y_true)
    predictor_model.fit(c_train, y_train)
    y_pred = predictor_model.predict(y_test)
    acc = accuracy_score(y_test, y_pred)

    return acc