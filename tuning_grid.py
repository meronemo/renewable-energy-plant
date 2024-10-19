# Hyperparameter Tuning Grid Search

from lib.get_data import get_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import neptune

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