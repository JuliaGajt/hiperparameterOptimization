from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR


def select_ml_algorithm(alg, args):
    if alg == 'CART':
        ml_alg = DecisionTreeRegressor(criterion='squared_error', random_state=42)
        param_dict = {"splitter": ["best", "random"],
                      "max_depth": [1, 3, 5, 7, 9, 11, 12],
                      "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4, 0.5],
                      "max_features": [5, 10, 20, "log2", "sqrt", None],
                      "max_leaf_nodes": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                      'min_samples_split': [2, 5, 10],
                      'min_impurity_decrease': [0.0, 0.1, 0.2],
                      'ccp_alpha': [0.0, 0.001, 0.01]
                      }
        return ml_alg, param_dict

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
        return ml_alg, param_dict

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
        return ml_alg, param_dict

    if alg == 'KNN':
        ml_alg = KNeighborsRegressor(n_jobs=-1)
        param_dict = {'n_neighbors': [3, 5, 7, 9, 11],
                      'weights': ['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'leaf_size': [10, 20, 30, 40],
                      'p': [1, 2, 3],
                      'metric': ['euclidean', 'manhattan', 'minkowski']
                      }
        return ml_alg, param_dict

    if alg == 'PLS':
        ml_alg = PLSRegression()
        param_dict = {'n_components': [1, 2, 3, 4, 5],
                      'scale': [True, False],
                      'max_iter': [100, 200, 300, 400, 500],
                      'tol': [1e-3, 1e-4, 1e-5]
                      }
        return ml_alg, param_dict

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
        return ml_alg, param_dict


def select_hyper_algorithm(hyper_alg, ml_model, params):
    if hyper_alg == 'grid search':
        # cv_params = [1, 3, 5]
        grid_search = GridSearchCV(estimator=ml_model, cv=KFold(n_splits=3, shuffle=True), scoring="neg_mean_squared_error", param_grid=params,
                                   n_jobs=3)
        return grid_search

