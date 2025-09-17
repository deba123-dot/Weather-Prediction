import pandas as pd
import numpy as np
from src.data_collector import WeatherDataCollector
from src.data_preprocessor import WeatherDataPreprocessor
from src.model_trainer import WeatherRandomForest
import os
from config import *

def main():
    print("Weather Prediction ML Model Training")
    print("=" * 40)
    
    # Data Collection
    print("\nData Collection")
    collector = WeatherDataCollector()
    
    raw_data_path = f"{DATA_DIR}/raw/weather_raw.csv"
    
    if os.path.exists(raw_data_path):
        print(f"Found existing data: {raw_data_path}")
        choice = input("Use existing data? (y/n): ").lower().strip()
        if choice == 'y':
            df_raw = collector.load_data()
        else:
            df_raw = collector.collect_weather_data()
    else:
        df_raw = collector.collect_weather_data()
    
    if df_raw is None or len(df_raw) == 0:
        print("Data collection failed")
        return
    
    # Data Preprocessing
    print("\nData Preprocessing")
    preprocessor = WeatherDataPreprocessor()
    
    df_features = preprocessor.create_features(df_raw)
    ml_data = preprocessor.prepare_data_for_training(df_features)
    
    # Model Training
    print("\nModel Training")
    rf_model = WeatherRandomForest()
    rf_model.train(ml_data['X_train'], ml_data['y_train'])
    
    # Model Evaluation
    print("\nModel Evaluation")
    metrics, predictions = rf_model.evaluate(ml_data['X_test'], ml_data['y_test'])
    
    top_features = rf_model.show_feature_importance(top_n=10)
    
    # Save Model
    print("\nSaving Model")
    model_path, metadata_path = rf_model.save_model(
        ml_data['scaler'],
        ml_data['city_encoder'],
        ml_data['feature_columns']
    )
    
    print("\nTraining Completed")
    print(f"Model: Random Forest")
    print(f"Performance: R2 = {metrics['R2']:.3f}")
    print(f"Model saved: {model_path}")

if __name__ == "__main__":
    main()
