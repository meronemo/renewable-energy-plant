# 데이터 전처리 (결측값, 중복값 삭제)

import pandas as pd

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

if __name__ == "__main__":
    df = pd.read_csv('dataset/solar_dataset.csv')
    df_cleaned = clean_data(df)
    df_cleaned = df_cleaned.to_csv('dataset/solar_dataset.csv', index=False)