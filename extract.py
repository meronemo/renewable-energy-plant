# Global Power Plant Database 중 Solar, Wind 데이터만 추출

import pandas as pd
import numpy as np

def extract(df):
    required_columns = ['latitude', 'longitude', 'primary_fuel']
    solar_wind_df = df[df['primary_fuel'].isin(['Solar', 'Wind'])].reset_index()
    solar_wind_df = solar_wind_df[['index'] + required_columns]
    solar_wind_df['index'] = solar_wind_df['index'] + 1
    solar_wind_df_sorted = solar_wind_df.sort_values(by='primary_fuel')
    return solar_wind_df_sorted

# Main execution
if __name__ == "__main__":
    df = pd.read_csv("power_plant.csv")
    result_df = extract(df)
    print(result_df.head())
    result_df.to_csv('extracted.csv', index=False)