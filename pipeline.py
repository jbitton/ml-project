import plot
from binary_classification.svm import SVM
from binary_classification.cart import CART
from binary_classification.xgb import XGB
from binary_classification.rf import RF
from conditional_probability.lr import LR
from conditional_probability.nb import NB
from preprocessing import preprocessing, get_train_and_test, standardize_features
from best_params import no_interactions, with_interactions

import warnings
warnings.filterwarnings("ignore")


def pipeline(add_features=False):
    """
    This function acts as a pipeline and calls the needed functions before
    any actual machine learning occurs.
    """
    x_values, y_values = preprocessing(add_features)
    x_train, x_test, y_train, y_test = get_train_and_test(x_values, y_values)
    x_train, x_test = standardize_features(x_train, x_test)

    hyperparameters = with_interactions if add_features else no_interactions
    dataset = (x_train, y_train, x_test, y_test)

    print("SVM:  ", SVM(dataset, hyperparameters['svm']).pipeline())
    print("CART: ", CART(dataset, hyperparameters['cart']).pipeline())
    print("XGB:  ", XGB(dataset, hyperparameters['xgb']).pipeline())
    print("RF:   ", RF(dataset, hyperparameters['rf']).pipeline())
    print("LOG:  ", LR(dataset, hyperparameters['lr']).pipeline())
    print("NB:   ", NB(dataset, hyperparameters['nb']).pipeline())
    plot.show_data()


pipeline()
