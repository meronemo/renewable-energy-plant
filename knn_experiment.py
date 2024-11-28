# Raw 모델 Train, Evaluate

from lib.get_data import get_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import neptune

def train():
    run = neptune.init_run(
        project="meronemo/renewable-energy-plant",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzNWE1YWYyZS1lN2ZmLTQ4YmYtOTUzMi1jNjY0ZGNiNzM3NDkifQ==",
        monitoring_namespace="monitoring",
        tags=["knn", "experiment"]
    )

    X_train, X_test, y_train, y_test = get_data()

    for n_neighbors in range(1, 31):
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')   
        cross_scores = cross_val_score(model, X_test, y_test, cv=10)
        run[f"acc"].append(cross_scores.mean())
        run[f"precision"].append(precision)
        run[f"recall"].append(recall)
        run[f"f1"].append(f1)
    
    run.stop()

if __name__ == '__main__':
    train()