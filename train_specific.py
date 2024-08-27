# 특정 지역 모델 적용

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

col_names = ['maxrad', 'minrad', 'meanrad', 'maxdur', 'mindur', 'meandur', 'maxwind', 'minwind', 'meanwind', 'elevation', 'energy']
feature_cols = ['maxrad', 'minrad', 'meanrad', 'maxdur', 'mindur', 'meandur', 'maxwind', 'minwind', 'meanwind', 'elevation']

def get_data():
    dataset = pd.read_csv("dataset/dataset.csv", header=0, names=col_names)
    X = dataset[feature_cols]
    y = dataset.energy
    return train_test_split(X, y, test_size=0.3, random_state=1) # 70% train, 30% test

def train():
    model = joblib.load('models/final.joblib')
    new_sample = [[22.56,7.62,15.0,43645.67,17414.33,28436.4,31.32,14.11,22.43,0.0]] 
    new_sample_df = pd.DataFrame(new_sample, columns=feature_cols)
    probabilities = model.predict_proba(new_sample_df)
    print(probabilities)


train()