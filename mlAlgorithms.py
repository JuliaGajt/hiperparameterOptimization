import numpy as np
from scipy.stats import loguniform, uniform
from scipy.stats import randint
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real
import random

from TPEAlgorithm import TPEAlgorithm
from SimulatedAnnealingAlgorithm import SimulatedAnnealing


def _sample(params, param):
    distribution = params[param]
    if type(distribution) is not list:
        return distribution.rvs()
    else:
        return random.choice(distribution)


def select_ml_algorithm(alg):
    if alg == 'CART':
        ml_alg = DecisionTreeRegressor(criterion='squared_error', random_state=42)
        param_dict = {
            "max_depth": [None, 1, 2, 5, 10, 15, 20, 30],
            "max_features": [0.3, 0.7, 1, "log2", "sqrt", None],
        }
        search_uniform = {
            "max_depth": randint(1, 30),
            "max_features": uniform(0.3, 0.7),
        }

        search_space = {
            "max_depth": Integer(1, 30),
            "max_features": Real(0.3, 1),
        }

    if alg == 'GBM':
        ml_alg = GradientBoostingRegressor(criterion='squared_error', random_state=42)
        param_dict = {
            'n_estimators': [20, 35, 50, 70, 80, 100],
            'learning_rate': [0.01, 0.05, 0.07, 0.1],
            'subsample': [0.3, 0.5, 0.75, 0.85, 1.0],
        }
        search_space = {
            'n_estimators': Integer(20, 100, prior='uniform'),
            'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
            'subsample': Real(0.3, 1.0, prior='uniform'),
        }
        search_uniform = {
            'n_estimators': randint(20, 100),
            'learning_rate': loguniform(a=0.01, b=0.1),
            'subsample': uniform(loc=0.3, scale=0.7),
        }

    if alg == 'SVM':
        ml_alg = SVR()
        param_dict = {'kernel': ['linear', 'poly'],
                      'degree': [2, 3],
                      'C': [0.01, 0.1, 1, 10],
                      'epsilon': [0.01, 0.1, 1],
                      }
        search_space = {
            'kernel': Categorical(['linear', 'poly']),
            'degree': Integer(2, 3),
            'C': Real(0.01, 10),
            'epsilon': Real(0.01, 1),
        }
        search_uniform = {
            'kernel': ['linear', 'poly'],
            'degree': randint(2, 3),
            'C': loguniform(0.01, 9.99),
            'epsilon': loguniform(0.01, 0.99),
        }

    if alg == 'KNN':
        ml_alg = KNeighborsRegressor()
        param_dict = {
            'n_neighbors': np.arange(2, 14, 1),
        }
        search_space = {
            'n_neighbors': Integer(2, 14),
        }
        search_uniform = {
            'n_neighbors': randint(2, 14),
        }

    if alg == 'PLS':
        ml_alg = PLSRegression()
        param_dict = {
            'n_components': [1, 2, 3, 4, 5],
            'max_iter': [100, 200, 300, 400, 500],
            'tol': [1e-3, 1e-4, 1e-5]
        }
        search_space = {
            'n_components': Integer(1, 5),
            'max_iter': Integer(100, 500, prior="uniform", dtype=int),
            'tol': Real(1e-5, 1e-3, prior="uniform", dtype=float)
        }
        search_uniform = {
            'n_components': randint(1, 5),
            'max_iter': randint(100, 500),
            'tol': loguniform(1e-5, 1e-3)
        }

    if alg == 'RF':
        ml_alg = RandomForestRegressor(criterion='squared_error', random_state=42)
        param_dict = {
            'n_estimators': [20, 40, 60, 80, 100],
            'max_depth': [None, 1, 2, 5, 10],
            'max_features': [0.3, 0.6, 0.8, 1],
        }
        search_space = {
            'n_estimators': Integer(20, 100),
            'max_depth': Integer(1, 10),
            'max_features': Real(0.3, 1),
        }
        search_uniform = {
            'n_estimators': randint(20, 100),
            'max_depth': randint(1, 10),
            'max_features': uniform(0.3, 0.7),
        }

    if alg == 'EN':
        ml_alg = ElasticNet(random_state=42)
        param_dict = {
            'alpha': np.logspace(-3, 3, 7),
            'l1_ratio': [0, 0.25, 0.5, 0.75, 1]
        }
        search_space = {
            'alpha': Real(0.001, 1000, prior='log-uniform'),
            'l1_ratio': Real(0, 1)
        }
        search_uniform = {
            'alpha': loguniform(0.001, 999.999),
            'l1_ratio': uniform(0, 1)
        }

    return ml_alg, param_dict, search_uniform, search_space


def select_hyper_algorithm(pipeline, hyper_alg, params, validation_method, X, y):
    cv_search = None

    if hyper_alg == 'grid search':
        cv_search = GridSearchCV(pipeline, scoring="neg_mean_squared_error",
                                 param_grid=params, cv=validation_method,
                                 n_jobs=-1)

    if hyper_alg == 'random search':
        cv_search = RandomizedSearchCV(pipeline, cv=validation_method, scoring="neg_mean_squared_error",
                                       param_distributions=params, n_iter=50,
                                       n_jobs=-1, random_state=42, error_score='raise')

    if hyper_alg == 'bayes search':
        cv_search = BayesSearchCV(pipeline, cv=validation_method, n_iter=50, scoring="neg_mean_squared_error",
                                  search_spaces=params,
                                  n_jobs=-1, random_state=42, error_score='raise')

    if hyper_alg == 'SA':
        fresh_pipeline = pipeline

        new_params = {key: _sample(params, key) for key, param in params.items()}
        new_estimator = dict(pipeline['model'].get_params(), **new_params)

        regressor = (fresh_pipeline['model'])

        fresh_pipeline.steps.pop()
        fresh_pipeline.steps.insert(2, ('model', regressor.__class__(**new_estimator)))

        cv_search = SimulatedAnnealing(initial_solution=fresh_pipeline['model'].get_params(),
                                       cv=validation_method, pipeline=pipeline, params=params)

    if hyper_alg == 'OPTUNA':
        cv_search = TPEAlgorithm(pipeline=pipeline, cv=validation_method, params=params, X=X, y=y)

    return cv_search
