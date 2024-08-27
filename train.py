# 모델 Train, Evaluate

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pydotplus
import scipy.stats as stats

col_names = ['maxrad', 'minrad', 'meanrad', 'maxdur', 'mindur', 'meandur', 'maxwind', 'minwind', 'meanwind', 'elevation', 'energy']
feature_cols = ['maxrad', 'minrad', 'meanrad', 'maxdur', 'mindur', 'meandur', 'maxwind', 'minwind', 'meanwind', 'elevation']

def get_data():
    dataset = pd.read_csv("dataset/dataset.csv", header=0, names=col_names)
    X = dataset[feature_cols]
    y = dataset.energy
    return train_test_split(X, y, test_size=0.3, random_state=1) # 70% train, 30% test

def train():
    X_train, X_test, y_train, y_test = get_data()
    dataset = pd.read_csv("dataset/dataset.csv", header=0, names=col_names)
    X = dataset[feature_cols]
    y = dataset.energy
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')   
    cross_scores = cross_val_score(model, X, y, cv=10)
    print("Cross validation score: ", cross_scores.mean())

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig("cm.png")

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

    importances = model.feature_importances_
    feature_names = X_train.columns

    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.savefig('feature_importance.png')

    

    # joblib.dump(model, 'models/third_model.joblib')
    # model = joblib.load('decision_tree_model.joblib')

def experiment_1():
    X_train, X_test, y_train, y_test = get_data()
    max_depth_values = range(1, 41)
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
    plt.title('Affect of max_depth on Decision Tree Performance')
    plt.legend()
    plt.grid(True)

    max_test_accuracy = max(test_accuracies)
    max_test_depth = max_depth_values[test_accuracies.index(max_test_accuracy)]
    plt.annotate(f'Max Accuracy: {max_test_accuracy:.4f}\n@ Max Depth: {max_test_depth}',
                 xy=(max_test_depth, max_test_accuracy), 
                 xytext=(max_test_depth, max_test_accuracy - 0.04),
                 arrowprops=dict(facecolor='black', linewidth=2, shrink=0.02))
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

def tuning_random():
    X_train, X_test, y_train, y_test = get_data()
    '''
    param_distributions = {
        'n_estimators': range(10, 201),            # Number of trees in the forest
        'max_depth': range(3, 51),                 # Maximum depth of the tree
        'min_samples_split': range(2, 21),         # Minimum number of samples required to split an internal node
        'min_samples_leaf': range(1, 21),          # Minimum number of samples required to be at a leaf node
        'max_features': ['None', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],  # Number of features to consider for the best split
    }
    '''
    param_distributions = {'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
    
    clf = RandomForestClassifier(random_state=1)
    random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_distributions, n_iter=300, cv=5)
    random_search.fit(X_train, y_train)
    rand_pf = random_search.best_estimator_.predict(X_test)
    rand_acc = accuracy_score(y_test, rand_pf)
    print(random_search.best_params_, rand_acc)
    with open("tuning.txt", 'w') as file:
        file.write(str(random_search.best_params_) + ' ' + str(rand_acc))

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
    '''
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    '''
    

def visualize(model):
    dot_data = export_graphviz(model, out_file=None, 
                            feature_names=feature_cols,  
                            class_names=['solar', 'wind'],
                            filled=True, rounded=True,  
                            special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('models/visualized.png')

if __name__ == '__main__':
    experiment_1()