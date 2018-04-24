from sklearn import svm


def svm_training(x_train, y_train):
    """
    :param x_train: the x-values we want to train on (2D numpy array)
    :param y_train: the y-values that correspond to x_train (1D numpy array)
    :return: sklearn SVM Classifier object that can now be used for predictions
    """
    clf = svm.SVC(kernel="rbf", C=1.089655, gamma=0.018571)
    clf.fit(x_train, y_train)
    return clf


def svm_classification(clf, x_test):
    """
    :param clf: trained sklearn SVM Classifier object
    :param x_test: the x-values we want to get predictions on (2D numpy array)
    :return: a 1D numpy array containing the predictions
    """
    return clf.predict(x_test)


def results(y_pred, y_test):
    """
    :param y_pred: the predicted y-values
    :param y_test: the actual y-values
    :return: the number of correct predictions, incorrect predictions, and the percent correct
    """
    num_right = 0
    num_wrong = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            num_right += 1
        else:
            num_wrong += 1
    return num_right, num_wrong, (num_right/(num_right + num_wrong))


def svm_pipeline(x_train, y_train, x_test, y_test):
    """
    :param x_train: the x-values we want to train on (2D numpy array)
    :param y_train: the y-values that correspond to x_train (1D numpy array)
    :param x_test:  the x-values we want to test on (2D numpy array)
    :param y_test:  the y-values that correspond to x_test (1D numpy array)
    :return: the number of correct predictions, incorrect predictions, and the percent correct
    """
    clf = svm_training(x_train, y_train)
    y_pred = svm_classification(clf, x_test)
    return results(y_pred, y_test)
