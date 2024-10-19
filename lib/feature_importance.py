# Feature Importance Plot 작성

import pandas as pd
import matplotlib.pyplot as plt
from lib.get_data import get_data

def feature_importance(model):
    X_train, X_test, y_train, y_test = get_data()
    importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.savefig('feature_importance.png')