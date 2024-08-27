# csv 파일 처리

import pandas as pd

# csv 파일에 energy 라벨 붙이기
def label():
    df = pd.read_csv('dataset/extracted_others.csv')
    df['energy'] = 'others'
    df.to_csv('dataset/extracted_others.csv', index=False)

# 두 csv 하나로 합치기
def combine():
    df1 = pd.read_csv('dataset/wind_dataset_2.csv')
    df2 = pd.read_csv('dataset/wind_dataset.csv')
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df.to_csv('dataset/wind_dataset.csv', index=False)

# 두 csv에서 각각 cnt개 데이터 샘플해서 하나로 합치기
def sample(cnt):
    solar_data = pd.read_csv('dataset/solar_dataset.csv')
    wind_data = pd.read_csv('dataset/wind_dataset.csv')

    solar_sample = solar_data.sample(n=cnt, random_state=1)
    wind_sample = wind_data.sample(n=cnt, random_state=1)

    combined_data = pd.concat([solar_sample, wind_sample])
    combined_data.to_csv('dataset/100_dataset.csv', index=False)

# 한 csv에서 cnt개 샘플
def sample2(cnt):
    data = pd.read_csv('dataset/extracted_others.csv')
    data_sample = data.sample(n=cnt, random_state=1)
    data_sample.to_csv('dataset/extracted_others.csv', index=False)

if __name__ == '__main__':
    sample(100)