# Decision Tree Max Depth 실험

from lib.get_data import get_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import neptune

def exp():
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
    
    '''
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    '''

if __name__ == '__main__':
    exp()