import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from util import roc_results, results
from base_model import BaseModel


class RF(BaseModel):
    def __init__(self, dataset, params):
        """
        :param dataset: contains, x_train, y_train, x_test, y_test (in that order)
        :param params:  the best hyperparameter values for the model
        """
        super().__init__(dataset, params)

    def train(self):
        self.model = RandomForestClassifier(**self.params)
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
        roc_results(y_pred, self.y_test, 'Random Forest')
        return roc_auc_score(self.y_test, y_pred), results(y_pred, self.y_test)

    def plot_feature_importance(self):
        """
        This function plots the feature importances in the random forest classifier

        ***NOTE*** THIS FUNCTION WILL ONLY RUN IF THERE ARE NO INTERACTION FEATURES
        """
        columns = np.array(['G1', 'G2', 'G3', 'G4', 'G7', 'G8', 'G10', 'G11', 'G12',
                            'G13', 'G15', 'G17', 'G22', 'G23', 'G24', 'G25', 'G26',
                            'G27', 'G28', 'G31', 'G34', 'G35', 'G36', 'G37', 'G39',
                            'G40', 'G42', 'G43', 'G44', 'G45', 'HCT', 'PLT', 'WBC',
                            'HGL', 'Age', 'TG', 'Bias'])
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        indices = [i for i in indices if importances[i] > 0]

        plt.figure()
        plt.title("Feature Importances: Random Forest")
        plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
        plt.xticks(range(len(indices)), columns[indices], rotation=90)
        plt.xlim([-1, len(indices)])
        plt.xlabel('Feature')
        plt.ylabel('Feature Importance')
        plt.show()
