# 실제 지역에 모델 Application

import pandas as pd
from lib.get_data import feature_cols
import joblib

def application():
    model = joblib.load('models/final.joblib')
    new_sample = [[22.56,7.62,15.0,43645.67,17414.33,28436.4,31.32,14.11,22.43,0.0]] 
    new_sample_df = pd.DataFrame(new_sample, columns=feature_cols)
    probabilities = model.predict_proba(new_sample_df)
    print(probabilities)

if __name__ == '__main__':
    application()