import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os

def train_random_forest(X_train, y_train):
    """
    Trains a RandomForestRegressor.
    """
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_arima(series, order=(5, 1, 0)):
    """
    Trains an ARIMA model on the sales series.
    """
    # Ensure index is datetime and series is sorted
    arima_model = ARIMA(series, order=order)
    fitted_model = arima_model.fit()
    return fitted_model

def save_model(model, filename, output_dir='../models'):
    """
    Saves trained models.
    """
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, filename))

def prepare_data_for_training(featured_df):
    """
    Splits feature DataFrame into X and y for ML models.
    """
    # For RandomForest
    features = ['day', 'month', 'year', 'day_of_week', 'lag_1', 'lag_7']
    X = featured_df[features]
    y = featured_df['sales']
    
    return X, y

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocess import load_data, preprocess_data, aggregate_daily_sales
    from feature_engineering import engineer_features
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'superstore.csv')
    
    print("Loading and preparing data...")
    daily_df = aggregate_daily_sales(preprocess_data(load_data(data_path)))
    featured_df = engineer_features(daily_df)
    
    # Train test split (80/20) for RF
    train_size = int(len(featured_df) * 0.8)
    train_df = featured_df.iloc[:train_size]
    test_df = featured_df.iloc[train_size:]
    
    X_train, y_train = prepare_data_for_training(train_df)
    X_test, y_test = prepare_data_for_training(test_df)
    
    print("Training RandomForest...")
    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, 'rf_model.pkl', os.path.join(script_dir, '..', 'models'))
    
    print("Training ARIMA...")
    # ARIMA usually takes the raw time series (un-lagged, but for simplicity we use the daily sales series)
    # We use daily_df so index is continuous datetime, ARIMA likes that.
    series = daily_df.set_index('date')['sales']
    train_series = series.iloc[:int(len(series)*0.8)]
    arima_model = train_arima(train_series)
    save_model(arima_model, 'arima_model.pkl', os.path.join(script_dir, '..', 'models'))
    
    print("Models saved successfully in 'models/' directory.")
