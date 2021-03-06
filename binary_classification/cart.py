from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from util import roc_results, results
from base_model import BaseModel


class CART(BaseModel):
    def __init__(self, dataset, params):
        """
        :param dataset: contains, x_train, y_train, x_test, y_test (in that order)
        :param params:  the best hyperparameter values for the model
        """
        super().__init__(dataset, params)

    def train(self):
        self.model = DecisionTreeClassifier(**self.params)
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        """
        :return: the predictions for self.x_test
        """
        return self.model.predict(self.x_test)

    def pipeline(self):
        """
        :return: the roc auc score and the accuracy
        """
        self.train()
        y_pred = self.predict()
        roc_results(y_pred, self.y_test, 'CART')
        return roc_auc_score(self.y_test, y_pred), results(y_pred, self.y_test)
