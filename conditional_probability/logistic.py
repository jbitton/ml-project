import numpy as np
from sklearn.linear_model import LogisticRegression
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


def probability_loss(y_pred, y_test):
    """
    :param y_pred: the predicted y-values
    :param y_test: the actual y-values
    :return: the logistic loss
    """
    total_log_loss = 0
    for i in range(len(y_test)):
        if y_test[i] == 1:
            total_log_loss += -np.log(y_pred[i][0])
        else:
            total_log_loss += -np.log(1 - y_pred[i][0])
    return total_log_loss


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
    return results(y_pred_class, y_test), probability_loss(y_pred, y_test)
