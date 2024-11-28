import pandas as pd
import requests
from datetime import datetime, timedelta
from statistics import median

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
    return round(data['elevation'][0], 2)

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
    
    monthly_data = merged_df.set_index('date').resample('ME').agg({
        'temperature': ['max', 'min', 'mean', 'median'],
        'solar_radiation': ['max', 'min', 'mean', 'median'], 
        'sunshine_duration': ['max', 'min', 'mean', 'median'],
        'wind_speed': ['max', 'min', 'mean', 'median']
    }).reset_index()
    
    monthly_data.columns = ['_'.join(col).strip() for col in monthly_data.columns]
    return monthly_data

data = pd.read_csv("dataset/solar.csv")

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

columns = ['latitude', 'longitude', 'temperature_max', 'temperature_min', 'temperature_mean', 'temperature_median',
          'solar_radiation_max', 'solar_radiation_min', 'solar_radiation_mean', 'solar_radiation_median',
          'sunshine_duration_max', 'sunshine_duration_min', 'sunshine_duration_mean', 'sunshine_duration_median',
          'wind_speed_max', 'wind_speed_min', 'wind_speed_mean', 'wind_speed_median', 'elevation',
          'temperature_extreme_count', 'solar_radiation_extreme_count', 'sunshine_duration_extreme_count', 'wind_speed_extreme_count']
completed = pd.DataFrame(columns=columns)

for i, row in enumerate(data.itertuples(), start=1):
    if i > 500: break
    lat = round(row.latitude, 2)
    long = round(row.longitude, 2)
    print(i, lat, long)

    try:
        weather = fetch(lat, long, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        weather = process_data(weather)

        # Calculate extreme values count
        temperature_extreme_count = 0
        solar_radiation_extreme_count = 0
        sunshine_duration_extreme_count = 0
        wind_speed_extreme_count = 0

        for col in ['temperature', 'solar_radiation', 'sunshine_duration', 'wind_speed']:
            q1 = round(weather[f"{col}_median"] - 1.5 * (weather[f"{col}_median"] - weather[f"{col}_min"]), 2)
            q3 = round(weather[f"{col}_median"] + 1.5 * (weather[f"{col}_max"] - weather[f"{col}_median"]), 2)
            extreme_count = ((weather[f"{col}_median"] < q1) | (weather[f"{col}_median"] > q3)).sum()
            if col == 'temperature':
                temperature_extreme_count = extreme_count
            elif col == 'solar_radiation':
                solar_radiation_extreme_count = extreme_count
            elif col == 'sunshine_duration':
                sunshine_duration_extreme_count = extreme_count
            elif col == 'wind_speed':
                wind_speed_extreme_count = extreme_count

        elevation = fetch_elevation(lat, long)
        row = {
            'latitude': lat,
            'longitude': long,
            'temperature_max': round(weather['temperature_max'].max(), 2),
            'temperature_min': round(weather['temperature_min'].min(), 2),
            'temperature_mean': round(weather['temperature_mean'].mean(), 2),
            'temperature_median': round(weather['temperature_median'].median(), 2),
            'solar_radiation_max': round(weather['solar_radiation_max'].max(), 2),
            'solar_radiation_min': round(weather['solar_radiation_min'].min(), 2),
            'solar_radiation_mean': round(weather['solar_radiation_mean'].mean(), 2),
            'solar_radiation_median': round(weather['solar_radiation_median'].median(), 2),
            'sunshine_duration_max': round(weather['sunshine_duration_max'].max(), 2),
            'sunshine_duration_min': round(weather['sunshine_duration_min'].min(), 2),
            'sunshine_duration_mean': round(weather['sunshine_duration_mean'].mean(), 2),
            'sunshine_duration_median': round(weather['sunshine_duration_median'].median(), 2),
            'wind_speed_max': round(weather['wind_speed_max'].max(), 2),
            'wind_speed_min': round(weather['wind_speed_min'].min(), 2),
            'wind_speed_mean': round(weather['wind_speed_mean'].mean(), 2),
            'wind_speed_median': round(weather['wind_speed_median'].median(), 2),
            'elevation': elevation,
            'temperature_extreme_count': temperature_extreme_count,
            'solar_radiation_extreme_count': solar_radiation_extreme_count,
            'sunshine_duration_extreme_count': sunshine_duration_extreme_count,
            'wind_speed_extreme_count': wind_speed_extreme_count
        }
        completed.loc[len(completed)] = row
    except Exception as e:
        print(e)
        break

completed.to_csv('experiment.csv', index=False)