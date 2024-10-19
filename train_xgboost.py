# XGBoost Training, Random Search Tuning
# lib.get_data에서 dataset 가져오지 않음, 독립적임

import pandas as pd
from lib.get_data import col_names, feature_cols
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
import scipy.stats as stats
import neptune

def train_xgboost():
    params = {'max_depth' : 3,
            'eta' : 0.1,
            'objective' : 'binary:logistic',
            'eval_metric' : 'logloss',
            'early_stoppings' : 100
            }

    param_dist = {
        'max_depth': stats.randint(3, 10),
        'learning_rate': stats.uniform(0.1, 0.2),
        'subsample': stats.uniform(0.5, 0.5),
        'n_estimators':stats.randint(50, 400)
    }

    dataset = pd.read_csv("dataset/dataset.csv", header=None, names=col_names)
    dataset = dataset.iloc[1:]
    dataset['is_solar'] = (dataset['energy'] == 'solar')

    for col in feature_cols:
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
    
    X = dataset[feature_cols]
    y = dataset.is_solar
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    xgb_model = XGBClassifier()
    random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy')
    random_search.fit(X_train, y_train)
    print("Best set of hyperparameters: ", random_search.best_params_)
    print("Best score: ", random_search.best_score_)

if __name__ == '__main__':
    train_xgboost()