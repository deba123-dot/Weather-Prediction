import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
from config import *

class WeatherDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.city_encoder = LabelEncoder()
        
    def load_data(self, filepath=None):
        if filepath is None:
            filepath = f"{DATA_DIR}/raw/weather_raw.csv"
        
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            print(f"Loaded {len(df)} records")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_features(self, df):
        print("Creating features")
        
        df_processed = df.copy()
        
        df_processed = df_processed.dropna(thresh=len(df_processed.columns) * 0.7)
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_columns] = df_processed[numeric_columns].fillna(df_processed[numeric_columns].mean())
        
        df_processed['year'] = df_processed['date'].dt.year
        df_processed['month'] = df_processed['date'].dt.month
        df_processed['day'] = df_processed['date'].dt.day
        df_processed['day_of_year'] = df_processed['date'].dt.dayofyear
        
        def get_season(month):
            if month in [12, 1, 2]:
                return 0
            elif month in [3, 4, 5]:
                return 1
            elif month in [6, 7, 8, 9]:
                return 2
            else:
                return 3
        
        df_processed['season'] = df_processed['month'].apply(get_season)
        df_processed['is_monsoon'] = (df_processed['month'].isin([6, 7, 8, 9])).astype(int)
        df_processed['city_encoded'] = self.city_encoder.fit_transform(df_processed['city'])
        
        if 'temperature_2m_max' in df_processed.columns and 'temperature_2m_min' in df_processed.columns:
            df_processed['temperature_range'] = df_processed['temperature_2m_max'] - df_processed['temperature_2m_min']
        
        df_processed = df_processed.sort_values(['city', 'date'])
        
        # Updated lag features with available variables
        for var in ['temperature_2m_max', 'temperature_2m_min']:
            if var in df_processed.columns:
                df_processed[f'{var}_yesterday'] = df_processed.groupby('city')[var].shift(1)
        
        # Updated rolling averages
        for var in ['temperature_2m_max']:
            if var in df_processed.columns:
                df_processed[f'{var}_week_avg'] = df_processed.groupby('city')[var].rolling(window=7, min_periods=3).mean().values
        
        df_processed = df_processed.dropna()
        
        print(f"Features created. Shape: {df_processed.shape}")
        return df_processed
    
    def prepare_data_for_training(self, df, target='temperature_2m_max'):
        print(f"Preparing data for training. Target: {target}")
        
        exclude_cols = ['date', 'city', 'latitude', 'longitude', target]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target]
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        os.makedirs(f"{DATA_DIR}/processed", exist_ok=True)
        
        processed_data = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train.reset_index(drop=True),
            'y_test': y_test.reset_index(drop=True),
            'feature_columns': list(X.columns),
            'scaler': self.scaler,
            'city_encoder': self.city_encoder
        }
        
        import joblib
        joblib.dump(processed_data, f"{DATA_DIR}/processed/ml_data.pkl")
        
        print("Data preparation completed")
        return processed_data
