# 데이터셋 불러오기

import pandas as pd
from sklearn.model_selection import train_test_split

col_names = ['maxrad', 'minrad', 'meanrad', 'maxdur', 'mindur', 'meandur', 'maxwind', 'minwind', 'meanwind', 'elevation', 'energy']
feature_cols = ['maxrad', 'minrad', 'meanrad', 'maxdur', 'mindur', 'meandur', 'maxwind', 'minwind', 'meanwind', 'elevation']

def get_data():
    dataset = pd.read_csv("dataset/dataset.csv", header=0, names=col_names)
    X = dataset[feature_cols]
    y = dataset.energy
    return train_test_split(X, y, test_size=0.3, random_state=1) # 70% train, 30% test