# Raw 모델 Train, Evaluate

from lib.get_data import get_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import neptune

def train():
    # run = neptune.init_run(
    #    project="meronemo/renewable-energy-plant",
    #    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzNWE1YWYyZS1lN2ZmLTQ4YmYtOTUzMi1jNjY0ZGNiNzM3NDkifQ==",
    #    monitoring_namespace="monitoring",
    #    tags=["dt"]
    #)

    X_train, X_test, y_train, y_test = get_data()

    # Train
    model = DecisionTreeClassifier(random_state=1)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    print(model.get_depth())

    # Evaluation
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='binary', pos_label='wind'),
        'recall': make_scorer(recall_score, average='binary', pos_label='wind'),
        'f1': make_scorer(f1_score, average='binary', pos_label='wind'),
    }
    
    cv_results = cross_validate(model, X_train, y_train, cv=10, scoring=scoring)

    print("Cross-Validation Results")
    print(f"Mean Accuracy: {cv_results['test_accuracy'].mean():.4f}")
    print(f"Mean Precision: {cv_results['test_precision'].mean():.4f}")
    print(f"Mean Recall: {cv_results['test_recall'].mean():.4f}")
    print(f"Mean F1 Score: {cv_results['test_f1'].mean():.4f}")
    print(cv_results)

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_dt.png")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix')
    print(conf_matrix)

    class_report = classification_report(y_test, y_pred)
    print('Classification Report')
    print(class_report)

    #run["acc"].append(cross_scores.mean())
    #run["precision"].append(precision) 
    #run["recall"].append(recall)
    #run["f1"].append(f1)

    # ROC AUC
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_test = y_test.map({'solar': 0, 'wind': 1}).astype(int)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})", color="blue")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("roccurve_dt.png")

    #run.stop()

    # 모델 저장
    # joblib.dump(model, 'models/third_model.joblib')
    # model = joblib.load('decision_tree_model.joblib')

if __name__ == '__main__':
    train()