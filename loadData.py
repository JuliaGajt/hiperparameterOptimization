import pandas as pd
from scipy.io import arff


def load_dataset(dataset_name):
    """
    :type dataset_name: string
    :return X: data
    :return y: target
    """

    global train

    if dataset_name == "maxwell":
        data = arff.loadarff('datasets/maxwell.arff')
        train = pd.DataFrame(data[0])
        effort = train["Effort"]
        return train.drop('Effort', axis=1), effort

    if dataset_name == "china":
        data = arff.loadarff('datasets/china.arff')
        train = pd.DataFrame(data[0])
        effort = train["Effort"]
        return train.drop('Effort', axis=1), effort

    if dataset_name == "desharnais":
        data = arff.loadarff('datasets/desharnais.arff')
        train = pd.DataFrame(data[0])
        effort = train["Effort"]
        return train.drop('Effort', axis=1), effort

    if dataset_name == "kitchenham":
        data = arff.loadarff('datasets/kitchenham.arff')
        train = pd.DataFrame(data[0])
        effort = train["Actual.effort"]
        return train.drop('Actual.effort', axis=1), effort

    if dataset_name == "miyazaki94":
        data = arff.loadarff('datasets/miyazaki94.arff')
        train = pd.DataFrame(data[0])
        effort = train['MM']
        return train.drop('MM', axis=1), effort

    if dataset_name == "nasa93":
        data = arff.loadarff('datasets/nasa93.arff')
        train = pd.DataFrame(data[0])
        effort = train['act_effort']
        return train.drop('act_effort', axis=1), effort
