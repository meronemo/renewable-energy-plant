# 모델 Train, Evaluate

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pydotplus
import joblib
import itertools
import json

col_names = ['maxrad', 'minrad', 'meanrad', 'maxdur', 'mindur', 'meandur', 'maxwind', 'minwind', 'meanwind', 'elevation', 'energy']
feature_cols = ['maxrad', 'minrad', 'meanrad', 'maxdur', 'mindur', 'meandur', 'maxwind', 'minwind', 'meanwind', 'elevation']

def get_data():
    dataset = pd.read_csv("dataset/dataset.csv", header=None, names=col_names)
    dataset = dataset.iloc[1:]
    X = dataset[feature_cols]
    y = dataset.energy
    return train_test_split(X, y, test_size=0.3, random_state=1) # 70% train, 30% test

def train():
    X_train, X_test, y_train, y_test = get_data()
    model = DecisionTreeClassifier(random_state=1)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix')
    print(conf_matrix)

    class_report = classification_report(y_test, y_pred)
    print('Classification Report')
    print(class_report)

    # joblib.dump(model, 'models/third_model.joblib')
    # model = joblib.load('decision_tree_model.joblib')

def experiment_1():
    X_train, X_test, y_train, y_test = get_data()
    max_depth_values = range(1, 51)
    train_accuracies = []
    test_accuracies = []
    
    for max_depth in max_depth_values:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        # print(f"max_depth: {max_depth}, train_accuracy: {train_accuracy:.4f}, test_accuracy: {test_accuracy:.4f}")

    plt.figure(figsize=(10, 6))
    # plt.plot(max_depth_values, train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(max_depth_values, test_accuracies, label='Testing Accuracy', marker='o')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Effect of max_depth on Decision Tree Performance')
    plt.legend()
    plt.grid(True)

    max_test_accuracy = max(test_accuracies)
    max_test_depth = max_depth_values[test_accuracies.index(max_test_accuracy)]
    plt.annotate(f'Max Accuracy: {max_test_accuracy:.4f}\n@ Max Depth: {max_test_depth}',
                 xy=(max_test_depth, max_test_accuracy), 
                 xytext=(max_test_depth, max_test_accuracy - 0.04),
                 arrowprops=dict(facecolor='black', linewidth=2, shrink=0.02))

    plt.annotate('Accuracy: 0.8877\n@ Max Depth: 24', xy=(24, 0.8877), xytext=(24, 0.8877-0.04), arrowprops=dict(facecolor='black', linewidth=2, shrink=0.02))

    plt.savefig('graphs/max_depth_and_accuracy.png')
    

def tuning():
    X_train, X_test, y_train, y_test = get_data()
    
    '''
    grid_search = {
        'max_depth': range(1, 50, 5),
        'min_samples_leaf': range(2, 50, 5),
        'min_samples_split': range(1, 50, 5),
        'max_features': ["None", "auto", "sqrt", "log2"]
    }
    '''

    grid_search = {
        'max_depth': [3,5,7,10],
        'n_estimators': [100, 200, 300, 400, 500],
        'max_features': [10, 20, 30, 40],
        'min_samples_leaf': [1, 2, 4]
    }
    
    clf = RandomForestClassifier(random_state=1)
    grid = GridSearchCV(clf, grid_search, cv=3)
    grid.fit(X_train, y_train)
    grid_pf = grid.best_estimator_.predict(X_test)
    grid_acc = accuracy_score(y_test, grid_pf)
    print(grid.best_params_, grid_acc)
    with open("tuning.txt", 'w') as file:
        file.write(str(grid.best_params_) + ' ' + str(grid_acc))

def visualize(model):
    dot_data = export_graphviz(model, out_file=None, 
                            feature_names=feature_cols,  
                            class_names=['solar', 'wind'],
                            filled=True, rounded=True,  
                            special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('models/visualized.png')

if __name__ == '__main__':
    tuning()