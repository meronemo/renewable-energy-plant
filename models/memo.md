# First Model
scikit-learn DecisionTreeClassifier
Solar와 Wind 에너지에 대해 각각 300개 데이터, 학습 70% 테스트 30%로 나눔
데이터 전처리 X, Accuracy 97.2%
데이터셋 열: maxrad,minrad,meanrad,maxdur,mindur,meandur,maxwind,minwind,meanwind,elevation,energy

# Second Model
모델 종류, 데이터셋 열은 처음 모델과 동일
각각 2500개 데이터, 학습:테스트 7:3 분배
전처리 X, Accuracy 92.6%
처음 모델에 비해 매우 복잡함, 전처리 및 정확도 향상 작업 필요
