import os, warnings, sys

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from urllib.parse import urlparse

import mlflow
import mlflow.sklearn

import logging
import logging

def eval_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    return rmse, mae, r2


def load_data():
    csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data = pd.read_csv(csv_url, sep=';')
    return data


def prepare_data():
    train, test = train_test_split(data)
    train_x = train.drop(['quality'], axis=1)
    test_x = test.drop(['quality'], axis=1)
    train_y = train[['quality']]
    test_y = test[['quality']]

    return train_x, train_y, test_x, test_y


def read_params():
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    return alpha, l1_ratio


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    data = load_data()
    train_x, train_y, test_x, test_y = prepare_data()
    alpha, l1_ratio = read_params()

    with mlflow.start_run():
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)

        predicted_y = model.predict(test_x)

        # evaluate the model
        (rmse, mae, r2) = eval_metrics(test_y, predicted_y)

        mlflow.log_params({'alpha': alpha, 'l1_ratio': l1_ratio})
        mlflow.log_metrics({'rmse': rmse, 'mae': mae, 'r2': r2})

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticNet Model")
        else:
            mlflow.sklearn.log_model(model, "model")
