# Random Forest n_estimators 실험

import matplotlib.pyplot as plt
from lib.get_data import get_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def exp(n_estimators_list):
    accuracies = []
    for n_estimators in n_estimators_list:
        X_train, X_test, y_train, y_test = get_data()
        
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        accuracies.append(accuracy)
        print(f'n_estimators={n_estimators}, Accuracy={accuracy:.4f}')
    return accuracies

if __name__ == '__main__':
    n_estimators_list = list(range(100, 1501, 100))
    accuracies = exp(n_estimators_list)

    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, accuracies, marker='o')
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    plt.title('Affect of n_estimators on Random Forest Performance')
    plt.grid(True)
    plt.savefig("graphs/n_estimators_and_accuracy.png")