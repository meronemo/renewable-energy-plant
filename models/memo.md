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




정확도 개선 위해 feature selection 고려
완성 후 feature importance 그래프 그려보는 것도 좋을듯