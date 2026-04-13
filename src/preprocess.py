import pandas as pd
import os

def load_data(filepath='../data/superstore.csv'):
    """
    Loads the Superstore dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    # Load dataset with specific encoding if needed, often Windows-1252 or utf-8 works.
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='windows-1252')
    
    return df

def preprocess_data(df):
    """
    Cleans and preprocesses the raw dataframe.
    - Converts 'Order Date' to datetime
    - Sorts data by date
    - Handles missing values (forward fill as requested)
    """
    # Convert to datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    
    # Sort data by date
    df = df.sort_values(by='Order Date').reset_index(drop=True)
    
    # Handle missing values (forward fill)
    df = df.ffill()
    
    return df

def aggregate_daily_sales(df):
    """
    Aggregates data to calculate daily total sales.
    OUTPUT: DataFrame -> [date, sales] (and renaming 'Order Date' to 'date', 'Sales' to 'sales')
    """
    # Group by Date
    daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()
    
    # Rename columns to standard format
    daily_sales = daily_sales.rename(columns={'Order Date': 'date', 'Sales': 'sales'})
    
    return daily_sales

if __name__ == "__main__":
    # Test the preprocessing script
    import __main__
    script_dir = os.path.dirname(os.path.abspath(__main__.__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'superstore.csv')
    
    raw_df = load_data(data_path)
    clean_df = preprocess_data(raw_df)
    daily_df = aggregate_daily_sales(clean_df)
    
    print(daily_df.head())
    print(f"Aggregated Data Shape: {daily_df.shape}")
