# Raw 모델 Train, Evaluate

from lib.get_data import get_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import neptune

def train():
    run = neptune.init_run(
        project="meronemo/renewable-energy-plant",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzNWE1YWYyZS1lN2ZmLTQ4YmYtOTUzMi1jNjY0ZGNiNzM3NDkifQ==",
        monitoring_namespace="monitoring",
        tags=["dt"]
    )

    X_train, X_test, y_train, y_test = get_data()

    # Train
    model = DecisionTreeClassifier(random_state=1)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')   
    cross_scores = cross_val_score(model, X_test, y_test, cv=10)
    print("Cross validation score: ", cross_scores.mean())

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig("confusion_matrix.png")

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

    run["acc"].append(accuracy)
    run["precision"].append(precision)
    run["recall"].append(recall)
    run["f1"].append(f1)
    run.stop()

    # 모델 저장
    # joblib.dump(model, 'models/third_model.joblib')
    # model = joblib.load('decision_tree_model.joblib')

if __name__ == '__main__':
    train()