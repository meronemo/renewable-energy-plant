# 데이터 전처리

import pandas as pd
import numpy as np

# 결측값, 중복값 제거
def clean_data(df):
    rows_with_missing = df.isnull().any(axis=1).sum()
    print(f"Number of rows with missing values: {rows_with_missing}")

    duplicate_rows = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_rows}")
    print(df[df.duplicated(keep=False)])

    df_cleaned = df.dropna()

    df_cleaned = df_cleaned.drop_duplicates()

    total_rows_removed = len(df) - len(df_cleaned)
    print(f"Total number of rows removed: {total_rows_removed}")

    return df_cleaned

# 이상치 제거
def remove_outlier(df):
    # For each column (feature) in the DataFrame
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        # Calculate IQR (Interquartile Range)
        IQR = Q3 - Q1
        # Define the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter the DataFrame to remove outliers
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

if __name__ == "__main__":
    df = pd.read_csv('dataset/dataset.csv')
    # df_cleaned = clean_data(df)
    # df_cleaned = df_cleaned.to_csv('dataset/solar_dataset.csv', index=False)
    df_cleaned = remove_outlier(df)
    df_cleaned.to_csv('dataset/outlier_removed.csv', index=False)