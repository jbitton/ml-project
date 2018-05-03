import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from util import results, roc_results


def logistic_training(x_train, y_train):
    """
    :param x_train: the x-values we want to train on (2D numpy array)
    :param y_train: the y-values that correspond to x_train (1D numpy array)
    :return: sklearn LogisticRegression object that can now be used for predictions
    """
    prob = LogisticRegression(C=1.2412060301507537, penalty='l2', solver='liblinear', random_state=10)
    prob.fit(x_train, y_train)
    return prob


def logistic_probability(prob, x_test):
    """
    :param prob: trained sklearn LogisticRegression object
    :param x_test: the x-values we want to get predictions on (2D numpy array)
    :return: a 2D numpy array containing the probabilities for both classes
    """
    return prob.predict_proba(x_test)


def logisitic_classification(y_pred, threshold=0.5):
    """
    :param y_pred: a 1D numpy array containing the probabilities of x belonging to the positive class
    :param threshold: determines which class a probability estimate belongs to
    :return: a 1D numpy array containing the predictions
    """
    for i in range(y_pred.shape[0]):
        y_pred[i] = 1 if y_pred[i] > threshold else -1
    return y_pred


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
    y_pred_class = logisitic_classification(np.copy(y_pred[:, 1]), threshold=0.5657625157625158)
    roc_results(y_pred[:, 1], y_test, 'Logistic Regression')
    return results(y_pred_class, y_test), log_loss(y_test, y_pred)
