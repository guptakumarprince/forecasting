# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:21:23 2024

@author: Prince Kumar Gupta
"""

import pandas as pd
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

df=pd.read_csv(r"C:\Users\Prince Kumar Gupta\OneDrive\Documents\project_163 document\preprocess data\preprocessed_data.csv")

df=df.rename(columns={'Unnamed: 0':'date'})

df=df.reset_index(drop=True)

# Convert the 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Set the 'date' column as the index
df.set_index('date', inplace=True)
    

# Number of data points to use for testing
num_test_points = 2

# Dictionary to store the best model and its metrics for each column
best_models2 = {}

# Loop through each column (kit) in the DataFrame
for column in df.columns:
    # Extract the data for the current column
    kit_data = df[column]
    
    # Split the data into training and testing sets
    train_data = kit_data.iloc[:-num_test_points]
    test_data = kit_data.iloc[-num_test_points:]
    
    # Fit models
    model_arima = ARIMA(train_data, order=(1, 1, 1))
    model_arima_fit = model_arima.fit()

    model_sarima = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_sarima_fit = model_sarima.fit()

    model_ets = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=12)
    model_ets_fit = model_ets.fit()
    
    model_ar = AutoReg(train_data, lags=1)
    model_ar_fit = model_ar.fit()
    
    # Forecast for each model
    forecast_arima = model_arima_fit.forecast(steps=len(test_data))
    forecast_sarima = model_sarima_fit.forecast(steps=len(test_data))
    forecast_ets = model_ets_fit.forecast(steps=len(test_data))
    forecast_ar = model_ar_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
    
    # Calculate RMSE, MAE, and MAPE for each model
    rmse_arima = np.sqrt(mean_squared_error(test_data, forecast_arima))
    mae_arima = mean_absolute_error(test_data, forecast_arima)
    mape_arima = np.mean(np.abs((test_data - forecast_arima) / test_data)) * 100
    
    rmse_sarima = np.sqrt(mean_squared_error(test_data, forecast_sarima))
    mae_sarima = mean_absolute_error(test_data, forecast_sarima)
    mape_sarima = np.mean(np.abs((test_data - forecast_sarima) / test_data)) * 100
    
    rmse_ets = np.sqrt(mean_squared_error(test_data, forecast_ets))
    mae_ets = mean_absolute_error(test_data, forecast_ets)
    mape_ets = np.mean(np.abs((test_data - forecast_ets) / test_data)) * 100
    
    rmse_ar = np.sqrt(mean_squared_error(test_data, forecast_ar))
    mae_ar = mean_absolute_error(test_data, forecast_ar)
    mape_ar = np.mean(np.abs((test_data - forecast_ar) / test_data)) * 100
    
    # Find the best model for the current column
    best_rmse = min(rmse_arima, rmse_sarima, rmse_ets, rmse_ar)
    best_model = None
    if best_rmse == rmse_arima:
        best_model = 'ARIMA'
        best_mae = mae_arima
        best_mape = mape_arima
    elif best_rmse == rmse_sarima:
        best_model = 'SARIMA'
        best_mae = mae_sarima
        best_mape = mape_sarima
    elif best_rmse == rmse_ets:
        best_model = 'ETS'
        best_mae = mae_ets
        best_mape = mape_ets
    elif best_rmse == rmse_ar:
        best_model = 'AR'
        best_mae = mae_ar
        best_mape = mape_ar
    
    # Store the best model and its metrics for the current column
    best_models2[column] = {
        'Best Model': best_model,
        'RMSE': best_rmse,
        'MAE': best_mae,
        'MAPE': best_mape
    }

# Convert the dictionary to a DataFrame
best_models2_df = pd.DataFrame(best_models2).T

# Save the best models and their metrics to a CSV file
best_models2_df.to_csv('best_models2.csv')

# Print the best models and their metrics
print("Best models and their metrics:")
print(best_models2_df)


# Store the best models and their metrics to a pickle file
best_models2_df.to_pickle('best_models2.pkl')


