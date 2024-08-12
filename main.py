import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, ShuffleSplit, \
    cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.metrics import r2_score

import mlAlgorithms
import loadData
import seaborn as sns
import matplotlib.pyplot as plt

from BootstrapCV import BootstrapCV


def display_results(pipe_disp, X_train_disp, y_train_disp, X_test_disp, y_test_disp, prediction_disp, score_disp):
    to_be_displayed = (
        f' \n ---------------------------------------------------------------------------\n'
        f' \n Before tuning: \n'
        f' Score for TRAIN dataset: {pipe_disp.score(X_train_disp, y_train_disp)} \n'
        f' Score for TEST dataset: {pipe_disp.score(X_test_disp, y_test_disp)} \n'
        f' MAE: {metrics.mean_absolute_error(y_test_disp, prediction_disp)} \n'
        f' MSE: {metrics.mean_squared_error(y_test_disp, prediction_disp)} \n'
        f' RMSE: {np.sqrt(metrics.mean_squared_error(y_test_disp, prediction_disp))} \n'
        f' The accuracy of our model is {round(score_disp, 2) * 100}.'
        f' ML algorithm setup: {pipe_disp["model"]} \n'
        f' ......... \n')

    print(to_be_displayed)

    # sns.histplot(y_test_disp - prediction_disp, kde=True, stat='density')
    # plt.tight_layout()
    # plt.show()
    # plt.scatter(y_test_disp, prediction_disp)
    # plt.show()
    #
    # explainer_disp = shap.Explainer(pipe_disp.predict, X_test_disp)  # model is a trained sklearn model where
    # shap_values_disp = explainer_disp(X_test_disp)  # X_test is our test data
    # shap.plots.beeswarm(shap_values_disp)
    # plt.show()


if __name__ == '__main__':

    # dataframe with all results
    results = pd.DataFrame(columns=['Tuned', 'Dataset', 'Regressor', 'Hyper_param_method', 'Validation_method',
                                    'n_splits', 'score_train', 'score_test', 'MAE', 'MSE', 'RMSE', 'MedianAE', 'R2',
                                    'best_params',
                                    'best_score', ' cross_val_score', 'cross_val_score_valid', 'cross_val_score_mean', 'cross_val_score_vslid_mean', 'target_values', 'predicted_target_values',
                                    'time_tuning'])
    results_dataset = results.copy()

    # creating folder for results
    # folder_name = f"ml-tuning-08-07-2024-splits-3"
    folder_name = f"ml-tuning-{datetime.now().strftime('%m-%d-%Y')}-splits-5"
    folder = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder)

    # tested settings
    datasets = ['maxwell', 'desharnais', 'kitchenham', 'miyazaki94', 'nasa93'] # china - too long
    regressors = ['CART', 'GBM', 'SVM', 'KNN', 'RF', 'EN']
    hyper_algorithms = ['grid search', 'random search', 'bayes search', 'OPTUNA', 'SA'] # GA - too long

    # settings
    num_experiments = 10
    n_splits = 5        # 3, 5, 10
    n_repeats = 3

    # validation methods
    validation_methods = [
        ('KFold', KFold(n_splits=n_splits)), # k-fold
        ('RepeatedKFold', RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)), # repeated k-fold
        ('HoldOut', ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)),  # hold-out
        # Monte Carlo Cross Validation
        # It is also called Repeated Random Sub-sampling Validation. Just like KFold, we split our data into folds again
        # However, folds are not predefined. The data is randomly split into training and test sets for each iteration.
        ('RepeatedHoldOut', ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)), # repeated hold-out
        # ('LeaveOneOut', LeaveOneOut()), # leave one out
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

                    """
                    Calculation on rough model --------------------------------------------------------------------
                    """

                    # selecting machine learning model
                    regressor, param_grid, param_uniform, param_spaces = mlAlgorithms.select_ml_algorithm(
                        regression_algorithm)

                    # creating pipeline with data scaling, feature reduction and regressor
                    pipe = Pipeline(steps=[("scaler", MinMaxScaler()),
                                           ("model", regressor)])

                    # if len(x.columns) > 9:
                    #     pipe = Pipeline(steps=[("scaler", MinMaxScaler()),
                    #                            ('pca', PCA(n_components=4)),
                    #                            ("model", regressor)])

                    # training regressor
                    start = datetime.now()
                    pipe.fit(X_train, y_train)
                    end = datetime.now()
                    # print(f" Time of tuning : {str(end - start)}")

                    # getting predictions, R2 score and cross validation scores from trained model
                    prediction = pipe.predict(X_test)
                    score = r2_score(y_test, prediction)
                    cross_val_results = cross_val_score(pipe, X_train, y_train, cv=n_splits,
                                                        scoring='neg_mean_squared_error')

                    # saving ALL results to dataframe "results"
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

                    # displaying all results in cmd and plots
                    # display_results(pipe, X_train, y_train, X_test, y_test, prediction, score)

                    MAE_not_tuned = metrics.mean_absolute_error(y_test, prediction)
                    mae_for_validation = [MAE_not_tuned]

                    # based on tuning algorithm select proper dictionary with params
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

                        # getting validation method name and object
                        val_met_name, val_met = validation_method
                        validation_methods_list.append(val_met_name)

                        """
                        Parameter tuning ----------------------------------------------------------------------------
                        """

                        # selecting fresh machine learning model and hyper-parameters dictonaries
                        regressor, _, _, _ = mlAlgorithms.select_ml_algorithm(
                            regression_algorithm)

                        # creating pipeline with data scaling, feature reduction and regressor
                        pipe = Pipeline(steps=[("scaler", MinMaxScaler()),
                                               ("model", regressor)])

                        # if len(x.columns) > 9:
                        #     pipe = Pipeline(steps=[("scaler", MinMaxScaler()),
                        #                            ('pca', PCA(n_components=4)),
                        #                            ("model", regressor)])

                        # get ready tuning method
                        tuned_regressor = mlAlgorithms.select_hyper_algorithm(pipe, hyper_algorithm,
                                                                              params,
                                                                              val_met, X_train, y_train)

                        # tune and train model
                        start = datetime.now()
                        tuned_regressor.fit(X_train, y_train)
                        end = datetime.now()
                        # print(f" Time of tuning : {str(end - start)}")

                        # getting predictions, R2 score and cross validation scores from tuned and trained model
                        prediction = tuned_regressor.best_estimator_.predict(X_test)
                        regressor = tuned_regressor.best_estimator_
                        score = r2_score(y_test, prediction)
                        cross_val_results = cross_val_score(regressor, X_train, y_train, cv=n_splits,
                                                            scoring='neg_mean_squared_error')
                        cross_val_results_val = cross_val_score(regressor, X_train, y_train, cv=val_met,
                                                            scoring='neg_mean_squared_error')

                        # saving ALL results to dataframe "results"
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

                        # displaying all results in cmd and plots
                        # display_results(regressor, X_train, y_train, X_test, y_test, prediction, score)

                        # print(f'''
                        #     Tuning of dataset: {dataset},
                        #     with ML method: {regressor['model']},
                        #     with tuning method: {hyper_algorithm},
                        #     with validation method: {val_met_name}, \n
                        #     AND IT IS {("BETTER!!! :DDD" if MAE_tuned < MAE_not_tuned else "WORSE :((((")}
                        # ''')

                        # mae_for_validation.append(MAE_tuned)
                    #
                    # plt.plot(validation_methods_list, mae_for_validation, label=hyper_algorithm)
                #
                # plt.ylabel("MAE")
                # plt.xlabel("Validation method")
                # plt.legend()
                # plt.title(
                #     f'MAE based on hyper-algorithm and validation method \n for {regression_algorithm} and {dataset}')
                # plt.show()

            results_dataset.to_csv(f'{folder_name}/model_performance_{dataset}-{datetime.now().strftime("%H-%M-%S")}.csv', index=False)
        results.to_csv(f'{folder_name}/model_performance_{i}-{datetime.now().strftime("%H-%M-%S")}.csv', index=False)
