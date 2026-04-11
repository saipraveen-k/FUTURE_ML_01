# Sales & Demand Forecasting

## 🎯 Problem Statement
Businesses often struggle with inventory management, staff allocation, and financial planning due to unpredictable sales trends. This project builds a reliable Sales & Demand Forecasting system using the Superstore dataset. It predicts the next 30 days of daily sales using advanced Machine Learning techniques.

## 📁 Dataset Info
The project uses the "Sample - Superstore.csv" dataset, which contains store transactional data including columns like `Order Date`, `Sales`, `Category`, `Region`, and `Profit`. The data is cleaned, aggregated to a daily basis, and enhanced with lagging and time-based features.

## 🚀 Approach

1. **Data Preprocessing**: Handled missing values (forward-filled), converted dates properly, and aggregated overall sales to a daily frequency. 
2. **Feature Engineering**: Extracted temporal features (`day`, `month`, `year`, `day_of_week`) and lag features (`lag_1`, `lag_7`) to capture historical momentum.
3. **Model Training**: 
   - **RandomForestRegressor**: Catches non-linear interactions between dates and lag features.
   - **ARIMA**: A robust statistical baseline for time-series forecasting.
4. **Evaluation**: MAE and RMSE are computed to ensure prediction reliability.
5. **Dashboarding**: A Streamlit app is provided for interactive visualization, segment filtering, and insightful tracking.

## 📊 Results & Business Impact

**Results**: The ensemble logic reveals patterns related to periodic demand spikes and weekend slumps. You can view the exact evaluation metrics in `outputs/metrics.txt` and the plot in `outputs/forecast_plot.png`.

**Business Impact**:
- **Inventory Planning**: Predict peak dates precisely.
- **Staff Allocation**: Optimize operations and warehouse staff based on the 30-day outlook volume.
- **Budget Forecasting**: Allow finance teams to stabilize expected revenue pipelines accurately.
- **Risk Reduction**: Anticipate slow periods to act ahead with promotional incentives.

---

## 💻 How to Run the Project

### 1. Install Requirements
Make sure you have Python installed. Use the following command to download dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Forecasts & Train Models
Run the core pipeline scripts to build the models and generate current forecasts:
```bash
python src/train_model.py
python src/forecast.py
python src/evaluate.py
```
*(This places `.pkl` models into `models/` and evaluation assets into `outputs/`)*

### 3. Start the Dashboard
Launch the Streamlit App to explore the final artifacts:
```bash
streamlit run app/dashboard.py
```

### 4. Interactive Analysis
Open the Jupyter Notebook for detailed walkthroughs:
```bash
jupyter notebook notebook/analysis.ipynb
```
