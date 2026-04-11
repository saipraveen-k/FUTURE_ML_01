import pandas as pd
import numpy as np

def forecast_next_30_days(rf_model, arima_model, last_date, last_sales_history):
    """
    Predicts the next 30 days of sales using both models.
    last_sales_history should be a list or array of the last 7 days of actual sales.
    """
    future_dates = [pd.to_datetime(last_date) + pd.Timedelta(days=i) for i in range(1, 31)]
    
    # --- ARIMA FORECAST ---
    # ARIMA models from statsmodels have a direct `forecast` method
    arima_preds_raw = arima_model.forecast(steps=30)
    arima_preds = arima_preds_raw.values if isinstance(arima_preds_raw, pd.Series) else arima_preds_raw
    arima_preds = [max(0, p) for p in arima_preds] # no negative sales
    
    # --- RANDOM FOREST FORECAST (Iterative) ---
    rf_preds = []
    # We maintain a running history of sales for lag variables
    current_history = list(last_sales_history) # needs exactly 7 elements
    
    for i in range(30):
        # Create features for the target future date
        target_date = future_dates[i]
        
        day = target_date.day
        month = target_date.month
        year = target_date.year
        day_of_week = target_date.dayofweek
        lag_1 = current_history[-1]
        lag_7 = current_history[-7]
        
        # Prepare feature vector (must match training feature order):
        # ['day', 'month', 'year', 'day_of_week', 'lag_1', 'lag_7']
        X_pred = pd.DataFrame([{
            'day': day, 'month': month, 'year': year, 'day_of_week': day_of_week,
            'lag_1': lag_1, 'lag_7': lag_7
        }])
        
        # Predict using RF
        pred_val = rf_model.predict(X_pred)[0]
        # Ensure no negative sales
        pred_val = max(0, pred_val)
        
        rf_preds.append(pred_val)
        
        # Update history
        current_history.append(pred_val)
        current_history = current_history[1:] # Keep length 7

    # Output DataFrame
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'RF_Forecast': rf_preds,
        'ARIMA_Forecast': arima_preds
    })
    
    return forecast_df

if __name__ == "__main__":
    import os
    import joblib
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocess import load_data, preprocess_data, aggregate_daily_sales
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load Models
    rf_model = joblib.load(os.path.join(script_dir, '..', 'models', 'rf_model.pkl'))
    arima_model = joblib.load(os.path.join(script_dir, '..', 'models', 'arima_model.pkl'))
    
    # Load Data to get the last date and last 7 days of sales
    data_path = os.path.join(script_dir, '..', 'data', 'superstore.csv')
    daily_df = aggregate_daily_sales(preprocess_data(load_data(data_path)))
    
    last_date = daily_df['date'].max()
    last_7_days_sales = daily_df['sales'].tail(7).values
    
    print(f"Generating forecast starting from {last_date + pd.Timedelta(days=1)}...")
    forecast_df = forecast_next_30_days(rf_model, arima_model, last_date, last_7_days_sales)
    
    print(forecast_df.head(10))
    # Optionally save forecast
    os.makedirs(os.path.join(script_dir, '..', 'outputs'), exist_ok=True)
    forecast_df.to_csv(os.path.join(script_dir, '..', 'outputs', 'forecast_30_days.csv'), index=False)
    print("Forecast saved to outputs/forecast_30_days.csv")

