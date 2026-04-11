import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

# Ensure we can import from src
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'src'))
from preprocess import load_data, preprocess_data, aggregate_daily_sales

st.set_page_config(page_title="Sales & Demand Forecasting", layout="wide")

# Custom CSS for color scheme
st.markdown("""
<style>
    /* Main background and text colors */
    .main .block-container {
        background-color: #FFFFFF;
    }
    
    /* Header styling - black and white theme */
    .main-header {
        color: #000000;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
        border-bottom: 3px solid #000000;
        padding-bottom: 10px;
    }
    
    /* Metric cards - black and white theme */
    .metric-card {
        background-color: #F5F5F5;
        border-left: 4px solid #000000;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Sidebar styling - black and white theme */
    .sidebar .sidebar-content {
        background-color: #F0F0F0;
    }
    
    /* Button styling - black and white theme */
    .stButton>button {
        background-color: #000000;
        color: white;
        border: none;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #333333;
    }
    
    /* Chart line colors - black */
    .stLineChart line {
        stroke: #000000;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    data_path = os.path.join(script_dir, '..', 'data', 'superstore.csv')
    raw_df = load_data(data_path)
    clean_df = preprocess_data(raw_df)
    return clean_df

@st.cache_data
def load_metrics_and_insights():
    metrics_path = os.path.join(script_dir, '..', 'outputs', 'metrics.txt')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return f.read()
    return "Metrics not yet generated. Run the evaluate.py script."

@st.cache_data
def load_forecast_data():
    forecast_path = os.path.join(script_dir, '..', 'outputs', 'forecast_30_days.csv')
    if os.path.exists(forecast_path):
        df = pd.read_csv(forecast_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

def main():
    st.markdown('<h1 class="main-header">📊 Sales & Demand Forecasting Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("This dashboard provides an overview of historical sales data and predicts future demand using Machine Learning.")
    
    # Load Data
    with st.spinner("Loading Data..."):
        df = load_and_prepare_data()
        
    st.sidebar.header("Filters")
    categories = ['All'] + list(df['Category'].dropna().unique())
    regions = ['All'] + list(df['Region'].dropna().unique())
    
    selected_category = st.sidebar.selectbox("Select Category", categories)
    selected_region = st.sidebar.selectbox("Select Region", regions)
    
    # Filter dataset
    filtered_df = df.copy()
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
        
    # Aggregate to daily
    daily_sales = aggregate_daily_sales(filtered_df)
    
    st.subheader("📈 Historical Sales Chart")
    if daily_sales.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Display Key Metrics
        col1, col2, col3 = st.columns(3)
        avg_sales = daily_sales['sales'].mean()
        total_sales = daily_sales['sales'].sum()
        max_sales = daily_sales['sales'].max()
        
        col1.metric("Average Daily Sales", f"${avg_sales:,.2f}")
        col2.metric("Total Sales Volume", f"${total_sales:,.2f}")
        col3.metric("Max Daily Sales", f"${max_sales:,.2f}")
        
        # Plot Historical
        st.line_chart(daily_sales.set_index('date')['sales'])
        
    st.subheader("🔮 30-Day Outlook (Overall Models)")
    st.info("Note: Forecasting models are trained on the overall aggregate data.")
    
    forecast_df = load_forecast_data()
    if forecast_df is not None:
        plot_df = forecast_df.set_index('Date')
        st.line_chart(plot_df[['RF_Forecast', 'ARIMA_Forecast']])
    else:
        st.error("Forecast data missing. Please ensure the backend training pipeline was executed.")
        
    st.subheader("📋 Model Evaluation & Business Insights")
    insights_text = load_metrics_and_insights()
    st.text(insights_text)
    
if __name__ == "__main__":
    main()
