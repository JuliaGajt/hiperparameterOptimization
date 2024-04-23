from collections import Counter

import numpy as np
import pandas as pd
from scipy.io import arff


def check_for_null_values(df):
    if np.any(pd.isnull(df)):
        print(f'There are several null values: {np.where(pd.isnull(df))}')


def IQR_method(df, n, features):
    """
    Takes a dataframe and returns an index list corresponding to the observations
    containing more than n outliers according to the Tukey IQR method.
    """
    outlier_list = []

    for column in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[column], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[column], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # IQR = 0
        # outlier step
        outlier_step = 1.5 * IQR
        # Determining a list of indices of outliers
        outlier_list_column = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step)].index
        # appending the list of outliers
        outlier_list.extend(outlier_list_column)

    # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)
    multiple_outliers = list(k for k, v in outlier_list.items() if v > n)

    # Calculate the number of records below and above lower and above bound value respectively
    out1 = df[df[column] < Q1 - outlier_step]
    out2 = df[df[column] > Q3 + outlier_step]

    print('Total number of deleted outliers is:', out1.shape[0] + out2.shape[0])

    return multiple_outliers


def drop_outliers(df, effort_column_name):
    Outliers_IQR = IQR_method(df.drop(effort_column_name, axis=1), 1, df.drop(effort_column_name, axis=1).columns)
    return df.drop(Outliers_IQR, axis=0).reset_index(drop=True)


def load_dataset(dataset_name):
    """
    :type dataset_name: string
    :return X: data
    :return y: target
    """

    if dataset_name == "maxwell":
        data = arff.loadarff('datasets/maxwell.arff')
        train = pd.DataFrame(data[0])

        # too much data deleted, model overfiting
        # train = drop_outliers(train, 'Effort')

        effort = train["Effort"]
        features = train.drop('Effort', axis=1)

    if dataset_name == "china":
        data = arff.loadarff('datasets/china.arff')
        train = pd.DataFrame(data[0])
        train = drop_outliers(train, 'Effort')

        effort = train["Effort"]
        features = train.drop(['Effort', 'Dev.Type', 'ID'], axis=1)

    if dataset_name == "desharnais":
        data = arff.loadarff('datasets/desharnais.arff')
        train = pd.DataFrame(data[0])
        train['Language'] = train['Language'].astype(int)

        # too much data deleted, model overfiting
        # train = drop_outliers(train, 'Effort')

        effort = train["Effort"]
        features = train.drop('Effort', axis=1)

    if dataset_name == "kitchenham":
        data = pd.read_csv('datasets/kitchenham.csv')
        data = data.drop(['id', 'Project'], axis=1)

        columns_with_object_type = [col for col in data.columns if data[col].dtype == "O"]
        for column in columns_with_object_type:
            if 'date' in column:
                for row_id in data.index:
                    if data[column][row_id] == '?':
                        data = data.drop([row_id])

                data[column] = pd.to_datetime(data[column], format='%Y-%m-%d', errors='coerce')
                data[f'{column}_year_sin'] = np.sin(data[column].dt.year)
                data[f'{column}_year_cos'] = np.cos(data[column].dt.year)
                data[f'{column}_month_sin'] = np.sin(data[column].dt.month)
                data[f'{column}_month_cos'] = np.cos(data[column].dt.month)
                data[f'{column}_day_sin'] = np.sin(data[column].dt.day)
                data[f'{column}_day_cos'] = np.cos(data[column].dt.day)
                data = data.drop(column, axis=1)
            else:
                encodings = data.groupby([column])['Actual.effort'].mean()
                data[column] = data[column].map(encodings)

        # smaller errors. no MAE reduction :/ no overfitting (negative scores)
        data = drop_outliers(data, "Actual.effort")

        effort = data["Actual.effort"]
        features = data.drop('Actual.effort', axis=1)

    if dataset_name == "miyazaki94":
        data = arff.loadarff('datasets/miyazaki94.arff')
        train = pd.DataFrame(data[0])
        train = train.drop(['ID'], axis=1)

        # smaller errors (cross-validation), no overfiting, smaller MAE in general. no MAE reduction :/
        train = drop_outliers(train, "MM")

        effort = train['MM']
        features = train.drop('MM', axis=1)

    if dataset_name == "nasa93":
        data = arff.loadarff('datasets/nasa93.arff')
        train = pd.DataFrame(data[0])

        columns_with_object_type = [col for col in train.columns if train[col].dtype == "O"]
        for column in columns_with_object_type:
            encodings = train.groupby([column])['act_effort'].mean()
            train[column] = train[column].map(encodings)

        # smaller errors, better scores, no overfitting (negative scores), bigger MAE in general
        train = drop_outliers(train, "act_effort")
        effort = train['act_effort']
        features = train.drop('act_effort', axis=1)

    check_for_null_values(features)
    return features, effort

