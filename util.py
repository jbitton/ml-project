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