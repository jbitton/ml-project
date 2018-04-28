import pandas as pd
import numpy as np
import plot
from sklearn.model_selection import train_test_split
from binary_classification.svm import svm_pipeline
from binary_classification.cart import cart_pipeline
from conditional_probability.logistic import logistic_pipeline
np.set_printoptions(threshold=np.nan)


def add_extra_features(x_values):
    interaction_features = np.zeros((x_values.shape[0], 21))

    for row_idx in range(x_values.shape[0]):
        col_idx = 0
        for i in range(36, 42):
            for j in range(i, 42):
                interaction_features[row_idx][col_idx] = x_values[row_idx][i] * x_values[row_idx][j]
                col_idx += 1

    return np.concatenate((x_values, interaction_features), axis=1)


def add_total_genes(x_values):
    total_gene_values = np.zeros((x_values.shape[0], 1))
    for i in range(x_values.shape[0]):
        for j in range(36):
            total_gene_values[i] += x_values[i][j]

    return np.concatenate((x_values, total_gene_values), axis=1)


def load_data(add_features=False):
    """
    :return: a tuple of featurized x and y values
    """
    aml_data = pd.read_csv('data.csv', index_col=0)
    del aml_data['DrawID']
    y_values = aml_data['caseflag'].values
    del aml_data['caseflag']
    for column in aml_data.columns:
        aml_data[column].fillna(aml_data[column].mean(), inplace=True)  # missing columns replaced with avg column value
        if aml_data[column].var() <= 0:
            del aml_data[column]
    x_values = add_total_genes(aml_data.values)
    if add_features:
        x_values = add_extra_features(x_values)
    featurizer = np.vectorize(lambda x: 1 if x == 'Yes' else -1)
    y_values = featurizer(y_values)
    return x_values, y_values


def get_train_and_test(x_values, y_values):
    """
    :param x_values: the featurized values of x
    :param y_values: the featurized values of y
    :return: a tuple containing x_train, x_test, y_train, and y_test
    """
    test_set_size = int(len(y_values) * 0.2)
    return train_test_split(x_values, y_values, test_size=test_set_size, random_state=10)


def pipeline():
    """
    This function acts as a pipeline and calls the needed functions before
    any actual machine learning occurs.
    """
    x_values, y_values = load_data()
    x_train, x_test, y_train, y_test = get_train_and_test(x_values, y_values)
    x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))
    x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))

    print(svm_pipeline(x_train, y_train, x_test, y_test))
    print(cart_pipeline(x_train, y_train, x_test, y_test))
    print(logistic_pipeline(x_train, y_train, x_test, y_test))
    plot.show_data()


pipeline()
