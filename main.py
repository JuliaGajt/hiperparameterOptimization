import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, ShuffleSplit, \
    cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.metrics import r2_score

import mlAlgorithms
import loadData
import seaborn as sns
import matplotlib.pyplot as plt

from BootstrapCV import BootstrapCV


if __name__ == '__main__':

    # ramka z wynikami
    results = pd.DataFrame(columns=['Tuned', 'Dataset', 'Regressor', 'Hyper_param_method', 'Validation_method',
                                    'n_splits', 'score_train', 'score_test', 'MAE', 'MSE', 'RMSE', 'MedianAE', 'R2',
                                    'best_params',
                                    'best_score', ' cross_val_score', 'cross_val_score_valid', 'cross_val_score_mean', 'cross_val_score_vslid_mean', 'target_values', 'predicted_target_values',
                                    'time_tuning'])
    results_dataset = results.copy()

    # ustawienia metod walidacji
    num_experiments = 10
    n_splits = 10        # 3, 5, 10
    n_repeats = 3

    # tworzenie folderu dla wyników
    folder_name = f"ml-tuning-{datetime.now().strftime('%m-%d-%Y')}-splits-{n_splits}"
    folder = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder)

    # testowane kombinacje
    datasets = ['maxwell', 'desharnais', 'kitchenham', 'miyazaki94', 'nasa93']
    regressors = ['CART', 'GBM', 'SVM', 'KNN', 'RF', 'EN']
    hyper_algorithms = ['grid search', 'random search', 'bayes search', 'OPTUNA', 'SA']

    # meotdy walidacji
    validation_methods = [
        ('KFold', KFold(n_splits=n_splits)), # k-fold
        ('RepeatedKFold', RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)), # repeated k-fold
        ('HoldOut', ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)),  # hold-out
        ('RepeatedHoldOut', ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)), # repeated hold-out
        ('Bootstrap', BootstrapCV(n_bootstrap_samples=50)) # bootstrap
    ]

    for i in range(0, num_experiments):

        print(f"Experiment no. {i} \n")
        results = results.iloc[0:0]

        for dataset in tqdm(datasets):

            results_dataset = results_dataset.iloc[0:0]
            x, y = loadData.load_dataset(dataset)
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42 + i)

            plt.figure(figsize=(20, 10))
            sns.heatmap(x.corr(numeric_only=True), annot=True)
            plt.show()

            for regression_algorithm in regressors:

                mae_for_hyperalgorithm = []
                plt.figure(figsize=(18, 10))

                for hyper_algorithm in hyper_algorithms:

                    # uczenie modelu na zbiorze na bazowym modelu ML
                    # wybieranie modelu uczenia maszynowego
                    regressor, param_grid, param_uniform, param_spaces = mlAlgorithms.select_ml_algorithm(
                        regression_algorithm)

                    # tworzenie pipeline ze skalowaniem danyh i regresorem
                    pipe = Pipeline(steps=[("scaler", MinMaxScaler()),
                                           ("model", regressor)])

                    # trenowanie regresora
                    start = datetime.now()
                    pipe.fit(X_train, y_train)
                    end = datetime.now()

                    prediction = pipe.predict(X_test)
                    score = r2_score(y_test, prediction)
                    cross_val_results = cross_val_score(pipe, X_train, y_train, cv=n_splits,
                                                        scoring='neg_mean_squared_error')

                    # zapisywanie wszystkich wyników do ramki danych
                    results.loc[len(results.index)] = [0, dataset, regression_algorithm, hyper_algorithm,
                                                       None,
                                                       n_splits, pipe.score(X_train, y_train),
                                                       pipe.score(X_test, y_test),
                                                       metrics.mean_absolute_error(y_test, prediction),
                                                       metrics.mean_squared_error(y_test, prediction),
                                                       np.sqrt(metrics.mean_squared_error(y_test, prediction)),
                                                       metrics.median_absolute_error(y_test, prediction),
                                                       round(score, 2) * 100, pipe['model'].get_params(),
                                                       np.mean(cross_val_results),
                                                       cross_val_results.tolist(),
                                                       cross_val_results.tolist(),
                                                       np.mean(cross_val_results),
                                                       np.mean(cross_val_results),
                                                       y_test.tolist(),
                                                       prediction.tolist(), str(end - start)]
                    results_dataset.loc[len(results_dataset.index)] = results.iloc[len(results.index)-1]

                    MAE_not_tuned = metrics.mean_absolute_error(y_test, prediction)
                    mae_for_validation = [MAE_not_tuned]

                    # na podstawie algorytmu strojenia wybierz odpowiedni słownik z parametrami
                    params = {}
                    if hyper_algorithm == 'random search' or hyper_algorithm == 'GA':
                        params.update({f'model__{key}': values for key, values in param_uniform.items()})
                    if hyper_algorithm == 'grid search':
                        params.update({f'model__{key}': values for key, values in param_grid.items()})
                    if hyper_algorithm == 'bayes search':
                        params.update({f'model__{key}': values for key, values in param_spaces.items()})
                    if hyper_algorithm == 'GA' or hyper_algorithm == 'SA' or hyper_algorithm == 'OPTUNA':
                        params = param_uniform

                    validation_methods_list = ['None']

                    for validation_method in validation_methods:

                        # pobieranie nazwy i obiektu metody walidacji
                        val_met_name, val_met = validation_method
                        validation_methods_list.append(val_met_name)

                        """
                        Parameter tuning ----------------------------------------------------------------------------
                        """

                        # wybieranie nowego modelu uczenia maszynowego i słowników hiperparametrów
                        regressor, _, _, _ = mlAlgorithms.select_ml_algorithm(
                            regression_algorithm)

                        # tworzenie pipline ze skalowaniem danych i regresorem
                        pipe = Pipeline(steps=[("scaler", MinMaxScaler()),
                                               ("model", regressor)])


                        # przygotuj metodę strojenia
                        tuned_regressor = mlAlgorithms.select_hyper_algorithm(pipe, hyper_algorithm,
                                                                              params,
                                                                              val_met, X_train, y_train)

                        # dostrój i trenuj model
                        start = datetime.now()
                        tuned_regressor.fit(X_train, y_train)
                        end = datetime.now()

                        prediction = tuned_regressor.best_estimator_.predict(X_test)
                        regressor = tuned_regressor.best_estimator_
                        score = r2_score(y_test, prediction)
                        cross_val_results = cross_val_score(regressor, X_train, y_train, cv=n_splits,
                                                            scoring='neg_mean_squared_error')
                        cross_val_results_val = cross_val_score(regressor, X_train, y_train, cv=val_met,
                                                            scoring='neg_mean_squared_error')

                        # zapisywanie wszystkich wyników do ramki danych
                        results.loc[len(results.index)] = [1, dataset, regression_algorithm, hyper_algorithm,
                                                           val_met_name,
                                                           n_splits, regressor.score(X_train, y_train),
                                                           regressor.score(X_test, y_test),
                                                           metrics.mean_absolute_error(y_test, prediction),
                                                           metrics.mean_squared_error(y_test, prediction),
                                                           np.sqrt(metrics.mean_squared_error(y_test, prediction)),
                                                           metrics.median_absolute_error(y_test, prediction),
                                                           round(score, 2) * 100,
                                                           tuned_regressor.best_params_, tuned_regressor.best_score_,
                                                           cross_val_results.tolist(),
                                                           cross_val_results_val.tolist(),
                                                           np.mean(cross_val_results),
                                                           np.mean(cross_val_results_val),
                                                           y_test.tolist(),
                                                           prediction.tolist(), str(end - start)]
                        results_dataset.loc[len(results_dataset.index)] = results.iloc[len(results.index)-1]

                        MAE_tuned = metrics.mean_absolute_error(y_test, prediction)

            # results_dataset.to_csv(f'{folder_name}/model_performance_{dataset}-{datetime.now().strftime("%H-%M-%S")}.csv', index=False)
        results.to_csv(f'{folder_name}/model_performance_{i}-{datetime.now().strftime("%H-%M-%S")}.csv', index=False)
