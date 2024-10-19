# Hyperparameter Tuning Random Search

from lib.get_data import get_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import neptune

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