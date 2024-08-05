# 모델 Train, Evaluate

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pydotplus
import joblib


col_names = ['maxrad', 'minrad', 'meanrad', 'maxdur', 'mindur', 'meandur', 'maxwind', 'minwind', 'meanwind', 'elevation', 'energy']
dataset = pd.read_csv("dataset/dataset.csv", header=None, names=col_names)
dataset = dataset.iloc[1:]

feature_cols = ['maxrad', 'minrad', 'meanrad', 'maxdur', 'mindur', 'meandur', 'maxwind', 'minwind', 'meanwind', 'elevation']
X = dataset[feature_cols]
y = dataset.energy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% train, 30% test

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

joblib.dump(clf, 'models/second_model.joblib')

# For loading: clf = joblib.load('decision_tree_model.joblib')

dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=feature_cols,  
                           class_names=y.unique().astype(str),  # Convert class names to string
                           filled=True, rounded=True,  
                           special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('models/second_model.png')