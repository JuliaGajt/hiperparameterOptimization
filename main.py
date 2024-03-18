from datetime import datetime

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, ShuffleSplit, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import mlAlgorithms
import loadData
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    num_experiments = 3
    # datasets = ['maxwell', 'china', 'desharnais', 'kitchenham', 'miyazaki94', 'nasa93']
    datasets = ['maxwell']
    # regressors = ['CART', 'GBM']
    regressors = ['GBM']
    hyper_algorithms = ['grid search']
    # validation_methods = [
    #     ('KFold', KFold(n_splits=5)),
    #     ('RepeatedKFold', RepeatedKFold(n_splits=5, n_repeats=2)),
    #     ('ShuffleSplit', ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)),
    #     ('RepeatedStratifiedKFold', RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42))
    # ]

    validation_methods = ['KFold']

    #########################################################

    for dataset in tqdm(datasets):
        x, y = loadData.load_dataset(dataset)
        sns.heatmap(x[x.corr().index].corr(), annot=True)

        for alg in regressors:

            for i in range(num_experiments):

                ml_alg, params = mlAlgorithms.select_ml_algorithm(alg, [])
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                for hyper_algorithm in hyper_algorithms:

                    for validation_method in validation_methods:
                        ml_alg.fit(X_train, y_train)
                        prediction = ml_alg.predict(X_test)

                        to_be_displayed = (
                            f' \n ---------------------------------------------------------------------------\n'
                            f' \n Before tuning: \n'
                            f' Score for TRAIN dataset: {ml_alg.score(X_train, y_train)} \n'
                            f' Score for TEST dataset: {ml_alg.score(X_test, y_test)} \n'
                            f' MAE: {metrics.mean_absolute_error(y_test, prediction)} \n'
                            f' MSE: {metrics.mean_squared_error(y_test, prediction)} \n'
                            f' RMSE: {np.sqrt(metrics.mean_squared_error(y_test, prediction))} \n'
                            f' ML algorithm setup: {ml_alg} \n'
                            f' ......... \n')
                        print(to_be_displayed)

                        sns.histplot(y_test - prediction, kde=True, stat='density')
                        plt.tight_layout()
                        plt.show()
                        plt.scatter(y_test, prediction)
                        plt.show()

                        # Parameter tuning ----------------------------------------------------------------------------

                        hyper_alg = mlAlgorithms.select_hyper_algorithm(hyper_algorithm, ml_alg, params)

                        start = datetime.now()

                        hyper_alg.fit(X_train, y_train)

                        end = datetime.now()
                        print(f" Time of tuning : {str(end - start)}")

                        prediction = hyper_alg.best_estimator_.predict(X_test)

                        sns.histplot(y_test - prediction, kde=True, stat='density')
                        plt.tight_layout()
                        plt.show()
                        plt.scatter(y_test, prediction)
                        plt.show()

                        to_be_displayed = (
                            f' \n After tuning: \n'
                            f' Score for TRAIN dataset: {hyper_alg.best_estimator_.score(X_train, y_train)} \n'
                            f' Score for TEST dataset: {hyper_alg.best_estimator_.score(X_test, y_test)} \n'
                            f' MAE: {metrics.mean_absolute_error(y_test, prediction)} \n'
                            f' MSE: {metrics.mean_squared_error(y_test, prediction)} \n'
                            f' RMSE: {np.sqrt(metrics.mean_squared_error(y_test, prediction))} \n'
                            f' Hyper best estimator: {hyper_alg.best_estimator_} \n'
                            f' ---------------------------------------------------------------------------\n')
                        print(to_be_displayed)

                        print(f' Tuning with method: {hyper_algorithm} \n for ML algorithm: {alg} has finished.')
