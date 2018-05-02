from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from util import results


def logistic_training(x_train, y_train):
    """
    :param x_train: the x-values we want to train on (2D numpy array)
    :param y_train: the y-values that correspond to x_train (1D numpy array)
    :return: sklearn LogisticRegression object that can now be used for predictions
    """
    prob = LogisticRegression()
    prob.fit(x_train, y_train)
    return prob


def logistic_probability(prob, x_test):
    """
    :param prob: trained sklearn LogisticRegression object
    :param x_test: the x-values we want to get predictions on (2D numpy array)
    :return: a 2D numpy array containing the probabilities for both classes
    """
    return prob.predict_proba(x_test)


def logisitic_classification(prob, x_test):
    """
    :param prob: trained sklearn LogisticRegression object
    :param x_test: the x-values we want to get predictions on (2D numpy array)
    :return: a 1D numpy array containing the predictions
    """
    return prob.predict(x_test)


def logistic_pipeline(x_train, y_train, x_test, y_test):
    """
    :param x_train: the x-values we want to train on (2D numpy array)
    :param y_train: the y-values that correspond to x_train (1D numpy array)
    :param x_test:  the x-values we want to test on (2D numpy array)
    :param y_test:  the y-values that correspond to x_test (1D numpy array)
    :return: the number of correct predictions, incorrect predictions, the percent correct, and the loss
    """
    prob = logistic_training(x_train, y_train)
    y_pred = logistic_probability(prob, x_test)
    y_pred_class = logisitic_classification(prob, x_test)
    return results(y_pred_class, y_test), log_loss(y_test, y_pred)
