# 데이터셋 구성

import pandas as pd
import requests
from datetime import datetime, timedelta

def fetch(latitude, longitude, start_date, end_date):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_mean,shortwave_radiation_sum,sunshine_duration&hourly=windspeed_100m&timezone=auto"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data

def fetch_elevation(latitude, longitude):
    url = f"https://api.open-meteo.com/v1/elevation?latitude={latitude}&longitude={longitude}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data['elevation'][0]

def process_data(data):
    daily_df = pd.DataFrame({
        'date': pd.to_datetime(data['daily']['time']),
        'temperature': data['daily']['temperature_2m_mean'],
        'solar_radiation': data['daily']['shortwave_radiation_sum'],
        'sunshine_duration': data['daily']['sunshine_duration']
    })
    
    hourly_df = pd.DataFrame({
        'date': pd.to_datetime(data['hourly']['time']),
        'wind_speed': data['hourly']['windspeed_100m']
    })
    
    daily_wind_speed = hourly_df.set_index('date').resample('D').mean()
    
    merged_df = daily_df.merge(daily_wind_speed, left_on='date', right_index=True, how='left')
    
    monthly_data = merged_df.set_index('date').resample('ME').mean().reset_index()
    return monthly_data

data = pd.read_csv("dataset/wind.csv")

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

columns = ['maxrad', 'minrad', 'meanrad', 'maxdur', 'mindur', 'meandur', 'maxwind', 'minwind', 'meanwind', 'elevation']
completed = pd.DataFrame(columns=columns)


# Solar 1~3000번째까지 완료
# Wind 1~2892번째까지 완료

for i, row in enumerate(data.itertuples(), start=1):
    if i < 1127: continue
    if i == 3001: break
    lat = row.latitude
    long = row.longitude
    print(i, lat, long)

    try:
        weather = fetch(lat, long, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        weather = process_data(weather)
        summary = weather.agg(['max', 'min', 'mean'])
        elevation = fetch_elevation(lat, long)
        row = {'maxrad': summary.loc['max', 'solar_radiation'],
            'minrad': summary.loc['min', 'solar_radiation'],
            'meanrad': summary.loc['mean', 'solar_radiation'],
            'maxdur': summary.loc['max', 'sunshine_duration'],
            'mindur': summary.loc['min', 'sunshine_duration'],
            'meandur': summary.loc['mean', 'sunshine_duration'],
            'maxwind': summary.loc['max', 'wind_speed'],
            'minwind': summary.loc['min', 'wind_speed'],
            'meanwind': summary.loc['mean', 'wind_speed'],
            'elevation': elevation}
        row = {key: round(value, 2) for key, value in row.items()}
        completed.loc[len(completed)] = row
    except Exception as e:
        print(e)
        break

completed.to_csv('dataset/wind_dataset_2.csv', index=False)