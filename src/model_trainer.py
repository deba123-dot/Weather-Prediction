import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
from config import *

class WeatherRandomForest:
    def __init__(self):
        self.model = RandomForestRegressor(**RF_CONFIG)
        self.is_trained = False
        self.feature_importance = None
        
    def train(self, X_train, y_train):
        print("Training Random Forest model")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Model training completed")
        
    def evaluate(self, X_test, y_test):
        if not self.is_trained:
            print("Model not trained")
            return None
        
        print("Evaluating model")
        
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        print("Performance metrics:")
        print(f"  Mean Absolute Error: {mae:.2f}")
        print(f"  Root Mean Square Error: {rmse:.2f}")
        print(f"  R2 Score: {r2:.3f}")
        print(f"  Mean Absolute Percentage Error: {mape:.2f}%")
        
        return metrics, y_pred
    
    def show_feature_importance(self, top_n=10):
        if self.feature_importance is None:
            print("No feature importance data available")
            return
        
        print(f"Top {top_n} important features:")
        
        for i, row in self.feature_importance.head(top_n).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, scaler, city_encoder, feature_columns):
        if not self.is_trained:
            print("Cannot save untrained model")
            return
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        model_path = f"{MODELS_DIR}/weather_random_forest.pkl"
        joblib.dump(self.model, model_path)
        
        model_data = {
            'scaler': scaler,
            'city_encoder': city_encoder,
            'feature_columns': feature_columns,
            'feature_importance': self.feature_importance,
            'model_config': RF_CONFIG,
            'cities': [city['name'] for city in CITIES]
        }
        
        metadata_path = f"{MODELS_DIR}/model_metadata.pkl"
        joblib.dump(model_data, metadata_path)
        
        print(f"Model saved: {model_path}")
        print(f"Metadata saved: {metadata_path}")
        
        return model_path, metadata_path
