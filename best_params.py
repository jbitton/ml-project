no_interactions = {
    'nb': {
        'priors': [0.5, 0.5]
    },
    'lr': {
        'penalty': 'l1',
        'C': 0.185,
        'solver': 'saga',
        'fit_intercept': True,
        'random_state': 0
    },
    'svm': {
        'kernel': 'sigmoid',
        'C': 87,
        'gamma': 0.0212,
        'random_state': 0
    },
    'rf': {
        'n_estimators': 100,
        'criterion': 'gini',
        'max_depth': 4,
        'max_features': 'log2',
        'min_samples_split': 25,
        'min_samples_leaf': 7,
        'bootstrap': True,
        'random_state': 0
    },
    'cart': {
        'criterion': 'entropy',
        'splitter': 'best',
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_impurity_decrease': 0.023,
        'random_state': 0
    },
    'xgb': {
        'learning_rate': 0.3,
        'max_depth': 6,
        'reg_lambda': 4,
        'reg_alpha': 0.0001,
        'gamma': 3,
        'n_estimators': 100,
        'random_state': 0,
        'seed': 0
    }
}

with_interactions = {
    'nb': {
        'priors': [0.5, 0.5]
    },
    'lr': {
        'penalty': 'l1',
        'C': 0.12,
        'solver': 'saga',
        'fit_intercept': True,
        'random_state': 0
    },
    'svm': {
        'kernel': 'sigmoid',
        'C': 72,
        'gamma': 0.0049,
        'random_state': 0
    },
    'rf': {
        'n_estimators': 100,
        'criterion': 'gini',
        'max_depth': 5,
        'max_features': 'log2',
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'bootstrap': True,
        'random_state': 0
    },
    'cart': {
        'criterion': 'entropy',
        'splitter': 'random',
        'max_depth': 13,
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'min_impurity_decrease': 0.02,
        'random_state': 0
    },
    'xgb': {
        'learning_rate': 0.4,
        'max_depth': 10,
        'reg_lambda': 65,
        'reg_alpha': 3,
        'gamma': 1,
        'n_estimators': 100,
        'random_state': 0,
        'seed': 0
    }
}
