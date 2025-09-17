import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from config import *

class WeatherDataCollector:
    def __init__(self):
        self.base_url = OPEN_METEO_BASE_URL
        
    def fetch_city_data(self, lat, lon, start_date, end_date, city_name):
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'daily': ','.join(WEATHER_VARIABLES),
            'timezone': 'auto'
        }
        
        try:
            print(f"Fetching data for {city_name}")
            response = requests.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'daily' in data:
                    daily_data = data['daily']
                    
                    df = pd.DataFrame()
                    df['date'] = pd.to_datetime(daily_data['time'])
                    df['city'] = city_name
                    df['latitude'] = lat
                    df['longitude'] = lon
                    
                    for var in WEATHER_VARIABLES:
                        if var in daily_data:
                            df[var] = daily_data[var]
                        else:
                            df[var] = np.nan
                    
                    print(f"Collected {len(df)} records for {city_name}")
                    return df
                else:
                    print(f"No daily data found for {city_name}")
                    return None
            else:
                try:
                    error_data = response.json()
                    if 'reason' in error_data:
                        print(f"API error for {city_name}: {error_data['reason']}")
                    else:
                        print(f"API error for {city_name}: {response.status_code}")
                except:
                    print(f"API error for {city_name}: {response.status_code}")
                    print(f"Response: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"Error fetching {city_name} data: {e}")
            return None
    
    def collect_weather_data(self, start_date=START_DATE, end_date=END_DATE):
        print("Starting data collection")
        print(f"Period: {start_date} to {end_date}")
        print(f"Cities: {len(CITIES)}")
        
        all_data = []
        
        for i, city in enumerate(CITIES, 1):
            print(f"[{i}/{len(CITIES)}] {city['name']}")
            
            df = self.fetch_city_data(
                city['lat'], 
                city['lon'], 
                start_date, 
                end_date,
                city['name']
            )
            
            if df is not None:
                all_data.append(df)
            
            time.sleep(1)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Total records collected: {len(combined_df)}")
            
            os.makedirs(f"{DATA_DIR}/raw", exist_ok=True)
            raw_file = f"{DATA_DIR}/raw/weather_raw.csv"
            combined_df.to_csv(raw_file, index=False)
            print(f"Data saved to: {raw_file}")
            
            return combined_df
        else:
            print("No data collected")
            return None
