from sklearn import tree
from sklearn.metrics import roc_auc_score
from util import roc_results


def cart_training(x_train, y_train):
    """
    :param x_train: the x-values we want to train on (2D numpy array)
    :param y_train: the y-values that correspond to x_train (1D numpy array)
    :return: sklearn CART Classifier object that can now be used for predictions
    """
    clf = tree.DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)
    return clf


def cart_classification(clf, x_test):
    """
    :param clf: trained sklearn CART Classifier object
    :param x_test: the x-values we want to get predictions on (2D numpy array)
    :return: a 1D numpy array containing the predictions
    """
    return clf.predict(x_test)


def cart_pipeline(x_train, y_train, x_test, y_test):
    """
    :param x_train: the x-values we want to train on (2D numpy array)
    :param y_train: the y-values that correspond to x_train (1D numpy array)
    :param x_test:  the x-values we want to test on (2D numpy array)
    :param y_test:  the y-values that correspond to x_test (1D numpy array)
    :return: the roc auc score
    """
    clf = cart_training(x_train, y_train)
    y_pred = cart_classification(clf, x_test)
    roc_results(y_pred[:, 1], y_test, 'CART')
    return roc_auc_score(y_pred, y_test)
