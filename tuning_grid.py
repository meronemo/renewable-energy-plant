# Hyperparameter Tuning Grid Search

from lib.get_data import get_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import neptune

def tuning():
    run = neptune.init_run(
        project="meronemo/renewable-energy-plant",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzNWE1YWYyZS1lN2ZmLTQ4YmYtOTUzMi1jNjY0ZGNiNzM3NDkifQ==",
        monitoring_namespace="monitoring",
        tags=["decision_tree", "grid_search", "test"]
    )

    X_train, X_test, y_train, y_test = get_data()
    
    '''
    grid_search = {
        'max_depth': range(1, 50, 5),
        'min_samples_leaf': range(2, 50, 5),
        'min_samples_split': range(1, 50, 5),
        'max_features': ["None", "auto", "sqrt", "log2"]
    }
    '''

    '''
    grid_search = {
        'max_depth': [3,5,7,10],
        'n_estimators': [100, 200, 300, 400, 500],
        'max_features': [10, 20, 30, 40],
        'min_samples_leaf': [1, 2, 4]
    }
    '''

    # Grid for Decision Tree Tuning
    grid_search = {
        'criterion': ['gini', 'entropy'],
        'max_depth': list(range(1, 31)),
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10]
    }
    
    clf = DecisionTreeClassifier(random_state=1)
    grid = GridSearchCV(clf, grid_search, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    # with open("tuning.txt", 'w') as file:
    #     file.write(str(grid.best_params_) + ' ' + str(accuracy))

    for i, params in enumerate(grid.cv_results_['params']):
        run[f'grid_search/{i}/params'] = params
        run[f'grid_search/{i}/mean_test_score'] = grid.cv_results_['mean_test_score'][i]
        run[f'grid_search/{i}/std_test_score'] = grid.cv_results_['std_test_score'][i]
    run["best_params"] = grid.best_params_
    run["best_score"] = grid.best_score_

    run.stop()

if __name__ == '__main__':
    tuning()