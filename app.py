import gradio as gr
import pandas as pd
import requests
from datetime import datetime, timedelta
import joblib

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


def predict(latitude, longitude):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    weather = fetch(latitude, longitude, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    weather = process_data(weather)

    summary = weather.agg(['max', 'min', 'mean'])
    elevation = fetch_elevation(latitude, longitude)
    data = {'maxrad': summary.loc['max', 'solar_radiation'],
        'minrad': summary.loc['min', 'solar_radiation'],
        'meanrad': summary.loc['mean', 'solar_radiation'],
        'maxdur': summary.loc['max', 'sunshine_duration'],
        'mindur': summary.loc['min', 'sunshine_duration'],
        'meandur': summary.loc['mean', 'sunshine_duration'],
        'maxwind': summary.loc['max', 'wind_speed'],
        'minwind': summary.loc['min', 'wind_speed'],
        'meanwind': summary.loc['mean', 'wind_speed'],
        'elevation': elevation}
    data = {key: round(value, 2) for key, value in data.items()}

    columns = ['maxrad', 'minrad', 'meanrad', 'maxdur', 'mindur', 'meandur', 'maxwind', 'minwind', 'meanwind', 'elevation']
    df = pd.DataFrame([data], columns=columns)

    model = joblib.load('model.pkl')
    probabilities = model.predict_proba(df)
    solar = round(probabilities[0][0], 2)
    wind = round(probabilities[0][1], 2)
    return solar, wind

demo = gr.Interface(
    fn=predict,
    inputs=["number", "number"],
    outputs=[gr.Label(label="solar"), gr.Label(label="wind")],
    description="Pick a location and copy the coordinates on [latlong.net](https://www.latlong.net/).",
    flagging_mode="never",
)

demo.launch()