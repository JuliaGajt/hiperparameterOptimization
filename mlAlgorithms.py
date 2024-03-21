from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real


def select_ml_algorithm(alg):

    if alg == 'CART':
        ml_alg = DecisionTreeRegressor(criterion='squared_error', random_state=42)
        param_dict = {"splitter": ["best", "random"],
                      "max_depth": [1, 3, 5, 7, 9, 11, 12],
                      "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4, 0.5],
                      # "max_features": [5, 10, 20, "log2", "sqrt", None],
                      # "max_leaf_nodes": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                      # 'min_samples_split': [2, 5, 10],
                      # 'min_impurity_decrease': [0.0, 0.1, 0.2],
                      # 'ccp_alpha': [0.0, 0.001, 0.01]
                      }
        search_space = {
            "splitter": Categorical(["best", "random"]),
            "max_depth": Integer(1, 12),
            "min_samples_leaf": Integer(1, 10),
            "min_weight_fraction_leaf": Real(0.1, 0.5),
            "max_features": Categorical([5, 10, 20, "log2", "sqrt", None]),
            "max_leaf_nodes": Categorical([None, 10, 20, 30, 40, 50, 60, 70, 80, 90]),
            'min_samples_split': Integer(2, 10),
            'min_impurity_decrease': Real(0.0, 0.2),
            'ccp_alpha': Real(0.0, 0.01)
        }

    if alg == 'GBM':
        ml_alg = GradientBoostingRegressor(loss='absolute_error', criterion='squared_error', random_state=42)
        param_dict = {'n_estimators': [50, 100, 150],
                      'learning_rate': [0.01, 0.1, 0.2],
                      'max_depth': [3, 5, 7, 9],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4],
                      'subsample': [0.5, 0.75, 1.0],
                      # 'loss': ['squared_error', 'absolute_error'],
                      'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
                      'min_impurity_decrease': [0.0, 0.1, 0.2],
                      # 'max_features': ['auto', 'sqrt', 'log2'],
                      }
        search_space = {
            'n_estimators': Integer(50, 150),
            'learning_rate': Real(0.01, 0.2),
            'max_depth': Integer(3, 9),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 4),
            'subsample': Real(0.5, 1.0),
            'min_weight_fraction_leaf': Real(0.0, 0.2),
            'min_impurity_decrease': Real(0.0, 0.2)
        }

    if alg == 'SVM':
        ml_alg = SVR()
        param_dict = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                      'degree': [2, 3, 4],
                      'gamma': ['scale', 'auto', 0.1, 1],
                      'coef0': [0.0, 0.1, 0.5],
                      'tol': [1e-3, 1e-4],
                      'C': [0.1, 1, 10],
                      'epsilon': [0.01, 0.1, 0.2],
                      'shrinking': [True, False],
                      'cache_size': [100, 200, 300]}
        search_space = {
            'kernel': Categorical(['linear', 'rbf', 'poly', 'sigmoid']),
            'degree': Integer(2, 4),
            'gamma': Categorical(['scale', 'auto', 0.1, 1]),
            'coef0': Real(0.0, 0.5),
            'tol': Real(1e-4, 1e-3),
            'C': Real(0.1, 10),
            'epsilon': Real(0.01, 0.2),
            'shrinking': Categorical([True, False]),
            'cache_size': Integer(100, 300)
        }

    if alg == 'KNN':
        ml_alg = KNeighborsRegressor(n_jobs=-1)
        param_dict = {'n_neighbors': [3, 5, 7, 9, 11],
                      'weights': ['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'leaf_size': [10, 20, 30, 40],
                      'p': [1, 2, 3],
                      'metric': ['euclidean', 'manhattan', 'minkowski']
                      }
        search_space = {
            'n_neighbors': Integer(3, 11),
            'weights': Categorical(['uniform', 'distance']),
            'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
            'leaf_size': Integer(10, 40),
            'p': Integer(1, 3),
            'metric': Categorical(['euclidean', 'manhattan', 'minkowski'])
        }

    if alg == 'PLS':
        ml_alg = PLSRegression()
        param_dict = {'n_components': [1, 2, 3, 4, 5],
                      'scale': [True, False],
                      'max_iter': [100, 200, 300, 400, 500],
                      'tol': [1e-3, 1e-4, 1e-5]
                      }
        search_space = {
            'n_components': Integer(1, 5),
            'scale': Categorical([True, False]),
            'max_iter': Integer(100, 500),
            'tol': Real(1e-5, 1e-3)
        }

    if alg == 'RF':
        ml_alg = RandomForestRegressor(criterion='squared_error', n_jobs=-1, random_state=42)
        param_dict = {'n_estimators': [50, 100, 150],
                      'max_depth': [None, 10, 20, 30],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4],
                      'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'bootstrap': [True, False],
                      'max_samples': [None, 0.5, 0.7, 0.9]
                      }
        search_space = {
            'n_estimators': Integer(50, 150),
            'max_depth': Categorical([None, 10, 20, 30]),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 4),
            'min_weight_fraction_leaf': Real(0.0, 0.2),
            'max_features': Categorical(['auto', 'sqrt', 'log2']),
            'bootstrap': Categorical([True, False]),
            'max_samples': Categorical([None, 0.5, 0.7, 0.9])
        }

    return ml_alg, param_dict, search_space


def select_hyper_algorithm(hyper_alg, ml_model, validation_method, param_grid=None, param_spaces=None):
    if param_spaces is None:
        param_spaces = {}
    if param_grid is None:
        param_grid = {}
    if hyper_alg == 'grid search':
        grid_search = GridSearchCV(estimator=ml_model, cv=5, scoring="neg_mean_squared_error",
                                   param_grid=param_grid,
                                   n_jobs=3)
        return grid_search

    if hyper_alg == 'random search':
        grid_search = RandomizedSearchCV(estimator=ml_model, cv=validation_method, scoring="neg_mean_squared_error",
                                         param_distributions=param_grid, n_iter=15,
                                         n_jobs=3, random_state=42)
        return grid_search

    if hyper_alg == 'bayes search':
        grid_search = BayesSearchCV(estimator=ml_model, cv=validation_method, scoring="neg_mean_squared_error",
                                    search_spaces=param_spaces,
                                    n_jobs=3, random_state=42)
        return grid_search
