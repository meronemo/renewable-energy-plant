# 이상치 탐지, 시각화

import pandas as pd
import matplotlib.pyplot as plt

numerical_columns = ['maxrad', 'minrad', 'meanrad', 'maxdur', 'mindur', 'meandur', 
                     'maxwind', 'minwind', 'meanwind', 'elevation']
df = pd.read_csv("dataset/dataset.csv")
print(df.head())

plt.figure(figsize=(12, 6))
df[numerical_columns].boxplot()

# Customize the plot
plt.title('Box Plot of Numerical Features')
plt.xlabel('Features')
plt.ylabel('Values')

# Show the plot
plt.tight_layout()
plt.show()

# Optional: Create individual box plots for each feature
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(numerical_columns):
    df.boxplot(column=col, ax=axes[i])
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
plt.savefig('outlier.png')