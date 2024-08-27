# 1
scikit-learn DecisionTreeClassifier
Solar와 Wind 에너지에 대해 각각 300개 데이터, 학습 70% 테스트 30%로 나눔
데이터 전처리 X, Accuracy 97.2%
데이터셋 열: maxrad,minrad,meanrad,maxdur,mindur,meandur,maxwind,minwind,meanwind,elevation,energy

# 2
모델 종류, 데이터셋 열은 처음 모델과 동일
각각 2500개 데이터, 학습:테스트 7:3 분배
전처리 X, Accuracy 92.6%
처음 모델에 비해 매우 복잡함, 전처리 및 정확도 향상 작업 필요
Confusion Matrix
[[699  58]
 [ 51 692]]
Classification Report
              precision    recall  f1-score   support

       solar       0.93      0.92      0.93       757
        wind       0.92      0.93      0.93       743

    accuracy                           0.93      1500
   macro avg       0.93      0.93      0.93      1500
weighted avg       0.93      0.93      0.93      150

# 3
각각 5300개 데이터 수집, 이후 중복값 제거 (Solar 263개, Wind 248개)
각 에너지에서 5000개씩 샘플하여 학습 데이터셋 구성
학습 시 하이퍼파라미터 설정 안함
Accuracy 88.7%
Confusion Matrix
[[1341  160]
 [ 177 1322]]
Classification Report
              precision    recall  f1-score   support

       solar       0.88      0.89      0.89      1501
        wind       0.89      0.88      0.89      1499

    accuracy                           0.89      3000
   macro avg       0.89      0.89      0.89      3000
weighted avg       0.89      0.89      0.89      3000

# 4
Third Model에서 하이퍼파라미터 튜닝함
Grid Search 방법
grid_search = {'max_depth': range(1, 50, 2),
                'min_samples_leaf': range(2, 50, 2),
                'min_samples_split': range(1, 50, 2)}
최적 파라미터: {'max_depth': 13, 'min_samples_leaf': 2, 'min_samples_split': 3} Accuracy: 0.8796666666666667

# 5 (Random Forest)
Accuracy: 0.9257
Confusion Matrix
[[1395  106]
 [ 117 1382]]
Classification Report
              precision    recall  f1-score   support

       solar       0.92      0.93      0.93      1501
        wind       0.93      0.92      0.93      1499

    accuracy                           0.93      3000
   macro avg       0.93      0.93      0.93      3000
weighted avg       0.93      0.93      0.93      3000

# 6 (Random Forest + Hyperparameter Tuning)
grid_search = {
        'max_depth': [3,5,7,10],
        'n_estimators': [100, 200, 300, 400, 500],
        'max_features': [10, 20, 30, 40],
        'min_samples_leaf': [1, 2, 4]
    }
{'max_depth': 10, 'max_features': 10, 'min_samples_leaf': , 'n_estimators': 500} 0.9133333333333333


param_distributions = {
        'n_estimators': range(10, 201),
        'max_depth': range(3, 51),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'max_features': ['None', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
    }
{'n_estimators': 138, 'min_samples_split': 6, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 33, 'criterion': 'entropy'} 0.9243333333333333

param_distributions = {'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
{'n_estimators': 800, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False} 0.925

