import os
from datetime import datetime, timedelta

OPEN_METEO_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_CURRENT_URL = "https://api.open-meteo.com/v1/forecast"

CITIES = [
    {'name': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777},
    {'name': 'Delhi', 'lat': 28.6139, 'lon': 77.2090},
    {'name': 'Bangalore', 'lat': 12.9716, 'lon': 77.5946},
    {'name': 'Chennai', 'lat': 13.0827, 'lon': 80.2707},
    {'name': 'Kolkata', 'lat': 22.5726, 'lon': 88.3639},
    {'name': 'Hyderabad', 'lat': 17.3850, 'lon': 78.4867},
    {'name': 'Pune', 'lat': 18.5204, 'lon': 73.8567},
    {'name': 'Ahmedabad', 'lat': 23.0225, 'lon': 72.5714}
]

START_DATE = "2020-01-01"
END_DATE = "2023-12-31"

WEATHER_VARIABLES = [
    'temperature_2m_max',
    'temperature_2m_min',
    'precipitation_sum',
    'wind_speed_10m_max',
    'wind_direction_10m_dominant'
]

RF_CONFIG = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

DATA_DIR = "data"
MODELS_DIR = "models"
