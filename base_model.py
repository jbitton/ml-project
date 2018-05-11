from abc import abstractmethod


class BaseModel(object):
    def __init__(self, dataset, params):
        """
        :param dataset: contains a tuple with four items (in the following order:
            - x_train: the x-values we want to train on (2D numpy array)
            - y_train: the y-values that correspond to x_train (1D numpy array)
            - x_test:  the x-values we want to test on (2D numpy array)
            - y_test:  the y-values that correspond to x_test (1D numpy array)
        :param params:  the best hyperparameter values for the model (dictionary)
        """
        self.x_train = dataset[0]
        self.y_train = dataset[1]
        self.x_test = dataset[2]
        self.y_test = dataset[3]
        self.params = params
        self.model = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def pipeline(self):
        pass
