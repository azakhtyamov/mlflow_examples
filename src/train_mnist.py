import os
import mlflow
import random
import hashlib
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from src.git_autocommit import autocommit

SEED = 0
TRACKING_URI = 'http://localhost:5000'
EXPERIMENT_NAME = 'mnist'
random.seed(SEED)
np.random.seed(SEED)


def train(cfg):
    os.system("conda env export > environment.yaml")
    autocommit(file_paths=['./'], message='Another version of random forest')
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    digits = datasets.load_digits()

    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=SEED)

    # Track hash of data & split
    data_hash = hashlib.md5()
    for df in [X_train, X_test, y_train, y_test]:
        data_hash.update(df)
    data_hash = data_hash.hexdigest()

    clf = RandomForestClassifier(**cfg, random_state=SEED)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    scores = classification_report(y_test, preds, output_dict=True)

    df = pd.json_normalize(scores, sep='_')
    df = df.to_dict(orient='records')[0]

    with mlflow.start_run():
        mlflow.log_params(cfg)
        mlflow.log_param('data_hash', data_hash)
        mlflow.log_metrics(df)
    print(df['macro avg_f1-score'])


if __name__ == '__main__':
    cfg = {'n_estimators': 500,
           'max_depth': 25,
           'min_samples_split': 2,
           'min_samples_leaf': 1,
           }
    train(cfg)
