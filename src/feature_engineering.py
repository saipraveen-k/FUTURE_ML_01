import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Creates time-based and lag features for the sales dataframe.
    Expected input DataFrame MUST contain 'date' and 'sales' columns.
    """
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Time-based features
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Sort just in case before lagging
    df = df.sort_values('date').reset_index(drop=True)
    
    # Lag features
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    
    # Drop NA rows caused by lagging
    df = df.dropna().reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Test script locally
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocess import load_data, preprocess_data, aggregate_daily_sales
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'superstore.csv')
    
    daily_df = aggregate_daily_sales(preprocess_data(load_data(data_path)))
    featured_df = engineer_features(daily_df)
    
    print(featured_df.head(10))
    print(f"Featured Data Shape: {featured_df.shape}")
