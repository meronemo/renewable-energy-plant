# Random Forest n_estimators 실험

from lib.get_data import get_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import neptune

def exp():
    X_train, X_test, y_train, y_test = get_data()
    n_estimators_values = range(10, 1501, 50)
    accuracies = []
    
    for n_estimators in n_estimators_values:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        accuracies.append(round(scores.mean(), 4))

    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_values, accuracies, label='Accuracy', marker='o')
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    plt.title('Affect of n_estimators on Random Forest Performance')
    plt.legend(loc="lower right")
    plt.grid(True)

    max_accuracy = max(accuracies)
    max_n_estimators = n_estimators_values[accuracies.index(max_accuracy)]
    plt.axvline(x=max_n_estimators, color='red', linestyle='--', alpha=0.2)
    plt.axhline(y=max_accuracy, color='red', linestyle='--', alpha=0.2)
    plt.plot(max_n_estimators, max_accuracy, 'ro', markersize=4)

    ax = plt.gca()
    secax_y = ax.secondary_yaxis('left')
    secax_x = ax.secondary_xaxis('top')
    secax_y.set_yticks([max_accuracy], [f'{max_accuracy:.4f}'])
    secax_x.set_xticks([max_n_estimators], [max_n_estimators])
    plt.savefig('graphs/n_estimators_and_accuracy.png')


if __name__ == '__main__':
    exp()