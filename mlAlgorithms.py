from scipy.stats import uniform, loguniform
from scipy.stats import randint
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
        ml_alg = DecisionTreeRegressor(criterion='squared_error', splitter='best', random_state=42)
        param_dict = {
            # "splitter": ["best", "random"],
            "max_depth": [None, 1, 2, 5, 10],
            # "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4, 0.5],
            "max_features": [0.3, 0.7, 1, "log2", "sqrt", None],
            # "max_leaf_nodes": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            # 'min_samples_split': [2, 5, 10],
            # 'min_impurity_decrease': [0.0, 0.1, 0.2],
        }
        search_uniform = {
            # "splitter": ["best", "random"],
            "max_depth": randint(1, 10),
            # "min_samples_leaf": randint(1, 10),
            # "min_weight_fraction_leaf": uniform(0.1, 0.4),
            "max_features": uniform(0.3, 0.7),
            # "max_features": [uniform(0.3, 1), "log2", "sqrt", None],
            # "max_features": list(range(1, 20)) + list(float(x / 100) for x in range(1, 100)) + ["log2", "sqrt", None],
            # "max_leaf_nodes": randint(10, 200),
            # 'min_samples_split': randint(2, 10),
            # 'min_impurity_decrease': uniform(0.0, 0.2),
        }

        search_space = {
            # "splitter": Categorical(["best", "random"]),
            "max_depth": Integer(1, 10),
            # "min_samples_leaf": Integer(1, 10, prior="uniform"),
            # "min_weight_fraction_leaf": Real(0.1, 0.5, prior="uniform"),
            "max_features": Real(0.3, 0.7),
            # "max_leaf_nodes": Categorical([None, 10, 20, 30, 40, 50, 60, 70, 80, 90]),
            # 'min_samples_split': Integer(2, 10, prior="uniform"),
            # 'min_impurity_decrease': Real(0.0, 0.2, prior="uniform"),
        }

    if alg == 'GBM':
        ml_alg = GradientBoostingRegressor(loss='absolute_error', criterion='squared_error', random_state=42)
        param_dict = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            # 'min_samples_leaf': [1, 2, 4],
            'subsample': [0.3, 0.75, 1.0],
            # 'loss': ['squared_error', 'absolute_error'],
            # 'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
            # 'min_impurity_decrease': [0.0, 0.1, 0.2],
            'max_features': ['auto', 'sqrt', 'log2'],
        }
        search_space = {
            'n_estimators': Integer(50, 150, prior='uniform'),
            'learning_rate': Real(0.01, 0.2, prior='uniform'),
            'max_depth': Integer(3, 9, prior='uniform'),
            'min_samples_split': Integer(2, 10, prior='uniform'),
            # 'min_samples_leaf': Integer(1, 4, prior='uniform'),
            'subsample': Real(0.5, 1.0, prior='uniform'),
            # 'loss': ['squared_error', 'absolute_error'],
            # 'min_weight_fraction_leaf': Real(0.0, 0.2, prior='uniform'),
            # 'min_impurity_decrease': Real(0.0, 0.2, prior='uniform'),
            'max_features': ['auto', 'sqrt', 'log2'],
        }
        search_uniform = {
            'n_estimators': randint(20, 100),
            'learning_rate': loguniform(0.01, 0.1),
            'max_depth': randint(4, 10),
            'min_samples_split': randint(2, 10),
            # 'min_samples_leaf': Integer(1, 4, prior='uniform'),
            'subsample': uniform(0.5, 0.5),
            # 'loss': ['squared_error', 'absolute_error'],
            # 'min_weight_fraction_leaf': uniform(0.0, 0.2),
            # 'min_impurity_decrease': uniform(0.0, 0.2),
            'max_features': ['auto', 'sqrt', 'log2'],
        }

    if alg == 'SVM':
        ml_alg = SVR()
        param_dict = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                      'degree': [2, 3, 4],
                      # 'gamma': ['scale', 'auto', 0.1, 1],
                      # 'coef0': [0.0, 0.1, 0.5],
                      # 'tol': [1e-3, 1e-4],
                      'C': [0.01, 0.1, 1, 10],
                      'epsilon': [0.01, 0.1, 1],
                      # 'shrinking': [True, False],
                      # 'cache_size': [100, 200, 300]
                      }
        search_space = {
            'kernel': Categorical(['linear', 'rbf', 'poly', 'sigmoid']),
            'degree': Integer(2, 4),
            # 'gamma': Categorical(['scale', 'auto', 0.1, 1]),
            # 'coef0': Real(0.0, 0.5),
            # 'tol': Real(1e-4, 1e-3),
            'C': Real(0.01, 10),
            'epsilon': Real(0.01, 1),
            # 'shrinking': Categorical([True, False]),
            # 'cache_size': Integer(100, 300)
        }
        search_uniform = {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'degree': randint(2, 4),
            # 'gamma': ['scale', 'auto', 0.1, 1],
            # 'coef0': uniform(0.0, 0.5),
            # 'tol': loguniform(1e-4, 1e-3),
            'C': loguniform(0.01, 10),
            'epsilon': loguniform(0.01, 1),
            # 'shrinking': [True, False],
            # 'cache_size': randint(100, 300)
        }

    if alg == 'KNN':
        ml_alg = KNeighborsRegressor(n_jobs=-1)
        param_dict = {
            'n_neighbors': [3, 5, 7, 9, 11],
            # 'weights': ['uniform', 'distance'],
            # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 20, 30, 40],
            # 'p': [1, 2, 3],
            # 'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        search_space = {
            'n_neighbors': Integer(3, 11),
            # 'weights': Categorical(['uniform', 'distance']),
            # 'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
            'leaf_size': Integer(10, 40),
            # 'p': Integer(1, 3),
            # 'metric': Categorical(['euclidean', 'manhattan', 'minkowski'])
        }
        search_uniform = {
            'n_neighbors': randint(3, 11),
            # 'weights': ['uniform', 'distance'],
            # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': randint(10, 40),
            # 'p': randint(1, 3),
            # 'metric': ['euclidean', 'manhattan', 'minkowski']
        }

    if alg == 'PLS':
        ml_alg = PLSRegression()
        param_dict = {
            'n_components': [1, 2, 3, 4, 5],
            # 'scale': [True, False],
            'max_iter': [100, 200, 300, 400, 500],
            'tol': [1e-3, 1e-4, 1e-5]
        }
        search_space = {
            'n_components': Integer(1, 5),
            # 'scale': Categorical([True, False]),
            'max_iter': Integer(100, 500, prior="uniform", dtype=int),
            'tol': Real(1e-5, 1e-3, prior="uniform", dtype=float)
        }
        search_uniform = {
            'n_components': randint(1, 5),
            # 'scale': [True, False],
            'max_iter': randint(100, 500),
            'tol': loguniform(1e-5, 1e-3)
        }

    if alg == 'RF':
        ml_alg = RandomForestRegressor(criterion='squared_error', n_jobs=-1, random_state=42)
        param_dict = {
            'n_estimators': [20, 50, 100],
            'max_depth': [None, 1, 2, 5, 10],
            # 'min_samples_split': [2, 5, 10],
            # 'min_samples_leaf': [1, 2, 4],
            # 'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
            'max_features': [0.3, 0.7, 1],
            # 'bootstrap': [True, False],
            # 'max_samples': [None, 0.5, 0.7, 0.9]
        }
        search_space = {
            'n_estimators': Integer(20, 100),
            'max_depth': [[None], Integer(1, 10)],
            # 'min_samples_split': Integer(2, 10),
            # 'min_samples_leaf': Integer(1, 4),
            # 'min_weight_fraction_leaf': Real(0.0, 0.2),
            'max_features': Real(0.3, 1),
            # 'bootstrap': Categorical([True, False]),
            # 'max_samples': Categorical([None, 0.5, 0.7, 0.9])
        }
        search_uniform = {
            'n_estimators': randint(20, 100),
            'max_depth': [[None], randint(1, 10)],
            # 'min_samples_split': randint(2, 10),
            # 'min_samples_leaf': randint(1, 4),
            # 'min_weight_fraction_leaf': uniform(0.0, 0.2),
            'max_features': uniform(0.3, 1),
            # 'bootstrap': [True, False],
            # 'max_samples': [None, 0.5, 0.7, 0.9]
        }

    return ml_alg, param_dict, search_uniform, search_space


def select_hyper_algorithm(pipeline, hyper_alg, params, validation_method):
    cv_search = None

    if hyper_alg == 'grid search':
        cv_search = GridSearchCV(pipeline, scoring="neg_mean_squared_error",
                                 param_grid=params, cv=validation_method,
                                 n_jobs=3)

    if hyper_alg == 'random search':
        cv_search = RandomizedSearchCV(pipeline, cv=validation_method, scoring="neg_mean_squared_error",
                                       param_distributions=params, n_iter=60,
                                       n_jobs=-1, random_state=42, error_score='raise')

    if hyper_alg == 'bayes search':
        cv_search = BayesSearchCV(pipeline, cv=validation_method, n_iter=60, scoring="neg_mean_squared_error",
                                  search_spaces=params,
                                  n_jobs=-1, random_state=42, error_score='raise')

    return cv_search
