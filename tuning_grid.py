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
        tags=["random_forest", "grid_search"]
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
    accuracy = accuracy_score(y_test, grid_pf)
    precision = precision_score(y_test, grid_pf, average='weighted')
    recall = recall_score(y_test, grid_pf, average='weighted')
    f1 = f1_score(y_test, grid_pf, average='weighted')
    print(grid.best_params_, accuracy)

    # with open("tuning.txt", 'w') as file:
    #     file.write(str(grid.best_params_) + ' ' + str(accuracy))

    run["acc"].append(accuracy)
    run["precision"].append(precision)
    run["recall"].append(recall)
    run["f1"].append(f1)
    run.stop()

if __name__ == '__main__':
    tuning()