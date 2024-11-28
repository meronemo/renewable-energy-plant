# Decision Tree Max Depth 실험

from lib.get_data import get_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import neptune

def exp():
    X_train, X_test, y_train, y_test = get_data()
    max_depth_values = range(1, 25)
    accuracies = []
    
    for max_depth in max_depth_values:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
        scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
        accuracies.append(round(scores.mean(), 4))

    plt.figure(figsize=(10, 6))
    plt.plot(max_depth_values, accuracies, label='Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Affect of max_depth on Decision Tree Performance')
    plt.legend(loc="lower right")
    plt.grid(True)
    print(accuracies)
    max_accuracy = max(accuracies)
    max_test_depth = max_depth_values[accuracies.index(max_accuracy)]
    plt.axvline(x=max_test_depth, color='red', linestyle='--', alpha=0.2)
    plt.axhline(y=max_accuracy, color='red', linestyle='--', alpha=0.2)
    plt.plot(max_test_depth, max_accuracy, 'ro', markersize=4)

    ax = plt.gca()
    secax_y = ax.secondary_yaxis('left')
    secax_x = ax.secondary_xaxis('top')
    secax_y.set_yticks([max_accuracy], [f'{max_accuracy:.4f}'])
    secax_x.set_xticks([max_test_depth], [max_test_depth])
    plt.savefig('graphs/max_depth_and_accuracy.png')
    
    '''
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    '''

if __name__ == '__main__':
    exp()