import mlflow
import random
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

SEED = 0
TRACKING_URI = 'http://localhost:5003'
EXPERIMENT_NAME = 'first_try2'
random.seed(SEED)
np.random.seed(SEED)

def train(cfg):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    digits = datasets.load_digits()

    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=SEED)

    clf = RandomForestClassifier(**cfg, random_state=SEED)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    scores = classification_report(y_test, preds, output_dict = True)

    df = pd.json_normalize(scores, sep='_')
    df = df.to_dict(orient='records')[0]

    with mlflow.start_run():
        mlflow.log_params(cfg)
        mlflow.log_metrics(df)
    print(df)

if __name__ == '__main__':
    cfg = {'n_estimators': 200,
           'max_depth': 5,
           'min_samples_split': 2,
           'min_samples_leaf': 1,
           }
    train(cfg)
