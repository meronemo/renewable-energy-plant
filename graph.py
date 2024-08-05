# [테스트] 특정 위치 그래프 그리기

import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_weather_data(latitude, longitude, start_date, end_date):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_mean,shortwave_radiation_sum,sunshine_duration&hourly=windspeed_100m&timezone=auto"
    
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    
    data = response.json()
    return data

def process_data(data):
    # Process daily data
    daily_df = pd.DataFrame({
        'date': pd.to_datetime(data['daily']['time']),
        'temperature': data['daily']['temperature_2m_mean'],
        'solar_radiation': data['daily']['shortwave_radiation_sum'],
        'sunshine_duration': data['daily']['sunshine_duration']
    })
    
    # Process hourly wind speed data
    hourly_df = pd.DataFrame({
        'date': pd.to_datetime(data['hourly']['time']),
        'wind_speed': data['hourly']['windspeed_100m']
    })
    
    # Calculate daily average wind speed
    daily_wind_speed = hourly_df.set_index('date').resample('D').mean()
    
    # Merge daily data with daily average wind speed
    merged_df = daily_df.merge(daily_wind_speed, left_on='date', right_index=True, how='left')
    
    # Resample to monthly averages
    monthly_data = merged_df.set_index('date').resample('M').mean().reset_index()
    return monthly_data

def plot_weather_graphs(data, lat, long, eng):
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    
    # Temperature graph
    axs[0, 0].plot(data['date'], data['temperature'], marker='o', color='red')
    axs[0, 0].set_title('Average Monthly Temperature')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Temperature (°C)')
    axs[0, 0].grid(True)

    # Solar radiation graph
    axs[0, 1].plot(data['date'], data['solar_radiation'], marker='o', color='orange')
    axs[0, 1].set_title('Average Monthly Solar Radiation')
    axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel('Solar Radiation (MJ/m²)')
    axs[0, 1].grid(True)

    # Sunshine duration graph
    axs[1, 0].plot(data['date'], data['sunshine_duration'], marker='o', color='yellow')
    axs[1, 0].set_title('Average Monthly Sunshine Duration')
    axs[1, 0].set_xlabel('Date')
    axs[1, 0].set_ylabel('Sunshine Duration (hours)')
    axs[1, 0].grid(True)

    # Wind speed graph
    axs[1, 1].plot(data['date'], data['wind_speed'], marker='o', color='blue')
    axs[1, 1].set_title('Average Monthly Wind Speed')
    axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel('Wind Speed (m/s)')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.suptitle(f'Weather Data for the Past 3 Years ({lat},{long}) {eng} Energy', fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.savefig('graph.png')
    plt.close()

def main():
    latitude = 33.9131
    longitude = -116.5631
    energy = "Wind"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365) 
    
    try:
        print("Open-Meteo API에서 데이터를 가져오는 중...")
        data = fetch_weather_data(
            latitude, 
            longitude, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        
        print("데이터 처리 중...")
        monthly_data = process_data(data)
        
        print("그래프 그리는 중...")
        plot_weather_graphs(monthly_data, latitude, longitude, energy)
        
    except requests.exceptions.RequestException as e:
        print(f"데이터 가져오기 오류: {e}")
    except KeyError as e:
        print(f"데이터 처리 오류: {e}")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")

if __name__ == "__main__":
    main()