from sklearn.ensemble import RandomForestClassifier
from util import results


def rf_training(x_train, y_train):
    """
    :param x_train: the x-values we want to train on (2D numpy array)
    :param y_train: the y-values that correspond to x_train (1D numpy array)
    :return: sklearn random forest classifier object that can now be used for predictions
    """
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    return clf


def rf_classification(clf, x_test):
    """
    :param clf: trained sklearn random forest classifier object
    :param x_test: the x-values we want to get predictions on (2D numpy array)
    :return: a 1D numpy array containing the predictions
    """
    return clf.predict(x_test)


def rf_pipeline(x_train, y_train, x_test, y_test):
    """
    :param x_train: the x-values we want to train on (2D numpy array)
    :param x_test: the y-values that correspond to x_train (1D numpy array)
    :param y_train: the x-values we want to test on (2D numpy array)
    :param y_test: the y-values that correspond to x_test (1D numpy array)
    :return: the number of correct predictions, incorrect predictions, and the percent correct
    """
    clf = rf_training(x_train, y_train)
    y_pred = rf_classification(clf, x_test)
    return results(y_pred, y_test)