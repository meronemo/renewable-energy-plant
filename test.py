# n_estimators에 따른 정확도 변화 그래프 그리기

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Define column names and feature columns
col_names = ['maxrad', 'minrad', 'meanrad', 'maxdur', 'mindur', 'meandur', 'maxwind', 'minwind', 'meanwind', 'elevation', 'energy']
feature_cols = ['maxrad', 'minrad', 'meanrad', 'maxdur', 'mindur', 'meandur', 'maxwind', 'minwind', 'meanwind', 'elevation']

def get_data():
    dataset = pd.read_csv("dataset/dataset.csv", header=0, names=col_names)
    X = dataset[feature_cols]
    y = dataset.energy
    return train_test_split(X, y, test_size=0.3, random_state=1) # 70% train, 30% test

def evaluate_n_estimators(n_estimators_list):
    accuracies = []
    
    for n_estimators in n_estimators_list:
        # Load data and split into train/test
        X_train, X_test, y_train, y_test = get_data()
        
        # Initialize and train the RandomForestClassifier
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
        model.fit(X_train, y_train)
        
        # Predict and calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        accuracies.append(accuracy)
        print(f'n_estimators={n_estimators}, Accuracy={accuracy:.4f}')
    
    return accuracies

# List of n_estimators values to evaluate
n_estimators_list = list(range(100, 1501, 100))

# Evaluate the model for different n_estimators
accuracies = evaluate_n_estimators(n_estimators_list)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_list, accuracies, marker='o')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Affect of n_estimators on Random Forest Performance')
plt.grid(True)
plt.savefig("graphs/n_estimators_and_accuracy.png")