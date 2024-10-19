# 각 특성별 이상치 데이터 포인트와 그의 클래스 확인

import pandas as pd
from lib.get_data import col_names
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset/dataset.csv", header=0, names=col_names)

# Function to get numerical columns
def get_numerical_columns(dataframe):
    return dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Get numerical columns (excluding 'energy' which is categorical)
numerical_columns = [col for col in get_numerical_columns(df) if col != 'energy']

# Function to identify outliers
def identify_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Analyze outliers for each numerical column
print("Outlier Analysis:")
for col in numerical_columns:
    outliers = identify_outliers(df, col)
    if not outliers.empty:
        print(f"\nOutliers in {col}:")
        print(outliers[['energy', col]])
        
        # Count outliers by energy class
        outlier_counts = outliers['energy'].value_counts()
        print("\nOutlier counts by energy class:")
        print(outlier_counts)
        
        # Visualize outliers by energy class
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='energy', y=col, data=df)
        plt.title(f'Distribution of {col} by Energy Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Overall distribution of energy classes
print("\nOverall distribution of energy classes:")
print(df['energy'].value_counts())

# Correlation between numerical features and energy (if energy is numeric)
if pd.api.types.is_numeric_dtype(df['energy']):
    correlation = df[numerical_columns + ['energy']].corr()['energy'].sort_values(ascending=False)
    print("\nCorrelation of numerical features with energy:")
    print(correlation)

# Print the first few rows of the dataframe to verify
print("\nFirst few rows of the dataframe:")
print(df.head())