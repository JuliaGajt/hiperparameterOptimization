import numpy as np
import pandas as pd
from scipy.io import arff


def check_for_null_values(df):
    if np.any(pd.isnull(df)):
        print(f'There are several null values: {np.where(pd.isnull(df))}')


def load_dataset(dataset_name):
    """
    :type dataset_name: string
    :return X: data
    :return y: target
    """

    if dataset_name == "maxwell":
        data = arff.loadarff('datasets/maxwell.arff')
        train = pd.DataFrame(data[0])

        effort = train["Effort"]
        features = train.drop('Effort', axis=1)

    if dataset_name == "desharnais":
        data = arff.loadarff('datasets/desharnais.arff')
        train = pd.DataFrame(data[0])
        train['Language'] = train['Language'].astype(int)

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

        effort = data["Actual.effort"]
        features = data.drop('Actual.effort', axis=1)

    if dataset_name == "miyazaki94":
        data = arff.loadarff('datasets/miyazaki94.arff')
        train = pd.DataFrame(data[0])
        train = train.drop(['ID'], axis=1)

        effort = train['MM']
        features = train.drop('MM', axis=1)

    if dataset_name == "nasa93":
        data = arff.loadarff('datasets/nasa93.arff')
        train = pd.DataFrame(data[0])
        train = train.drop(['recordnumber'], axis=1)

        columns_with_object_type = [col for col in train.columns if train[col].dtype == "O"]
        for column in columns_with_object_type:
            encodings = train.groupby([column])['act_effort'].mean()
            train[column] = train[column].map(encodings)

        effort = train['act_effort']
        features = train.drop('act_effort', axis=1)

    check_for_null_values(features)

    return features, effort

