# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:40:35 2024

@author: Prince Kumar Gupta
"""

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Load the dataset
df = pd.read_csv(r"C:\Users\Prince Kumar Gupta\OneDrive\Documents\project_163 document\preprocess data\preprocessed_data.csv")
df = df.rename(columns={'Unnamed: 0': 'date'})
df = df.reset_index(drop=True)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Number of data points to use for testing
num_test_points = 4

# Dictionary to store the best model and its metrics for each column
best_models = {}

# Loop through each column (kit) in the DataFrame
for column in df.columns:
    # Extract the data for the current column
    kit_data = df[column]
    
    # Split the data into training and testing sets
    train_data = kit_data.iloc[:-num_test_points]
    test_data = kit_data.iloc[-num_test_points:]
    
    try:
        best_rmse = float('inf')
        best_model = None
        best_forecast = None
        
        # Define parameter grid for ARIMA model
        arima_param_grid = {'p': [1, 2], 'd': [1], 'q': [1, 2]}
        
        # Define parameter grid for SARIMA model
        sarima_param_grid = {'p': [1], 'd': [1], 'q': [1], 'P': [1], 'D': [1], 'Q': [1], 's': [12]}
        
        # Define parameter grid for SARIMAX model
        sarimax_param_grid = {'order': [(1, 1, 1), (2, 1, 1)], 'seasonal_order': [(1, 1, 1, 12), (2, 1, 1, 12)]}
        
        # ARIMA
        for p in arima_param_grid['p']:
            for d in arima_param_grid['d']:
                for q in arima_param_grid['q']:
                    model_arima = ARIMA(train_data, order=(p, d, q))
                    model_arima_fit = model_arima.fit()
                    forecast_arima = model_arima_fit.forecast(steps=len(test_data))
                    rmse_arima = np.sqrt(mean_squared_error(test_data, forecast_arima))
                    
                    if rmse_arima < best_rmse:
                        best_rmse = rmse_arima
                        best_model = 'ARIMA'
                        best_forecast = forecast_arima
        
        # SARIMA
        for p in sarima_param_grid['p']:
            for d in sarima_param_grid['d']:
                for q in sarima_param_grid['q']:
                    for P in sarima_param_grid['P']:
                        for D in sarima_param_grid['D']:
                            for Q in sarima_param_grid['Q']:
                                for s in sarima_param_grid['s']:
                                    model_sarima = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
                                    model_sarima_fit = model_sarima.fit()
                                    forecast_sarima = model_sarima_fit.forecast(steps=len(test_data))
                                    rmse_sarima = np.sqrt(mean_squared_error(test_data, forecast_sarima))
                                    
                                    if rmse_sarima < best_rmse:
                                        best_rmse = rmse_sarima
                                        best_model = 'SARIMA'
                                        best_forecast = forecast_sarima
        
        # SARIMAX
        for order in sarimax_param_grid['order']:
            for seasonal_order in sarimax_param_grid['seasonal_order']:
                model_sarimax = SARIMAX(train_data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                model_sarimax_fit = model_sarimax.fit(disp=False)
                forecast_sarimax = model_sarimax_fit.forecast(steps=len(test_data))
                rmse_sarimax = np.sqrt(mean_squared_error(test_data, forecast_sarimax))
                
                if rmse_sarimax < best_rmse:
                    best_rmse = rmse_sarimax
                    best_model = 'SARIMAX'
                    best_forecast = forecast_sarimax

        # ETS
        model_ets = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=12)
        model_ets_fit = model_ets.fit()
        forecast_ets = model_ets_fit.forecast(steps=len(test_data))
        rmse_ets = np.sqrt(mean_squared_error(test_data, forecast_ets))
        if rmse_ets < best_rmse:
            best_rmse = rmse_ets
            best_model = 'ETS'
            best_forecast = forecast_ets
        
        # AR
        model_ar = AutoReg(train_data, lags=1)
        model_ar_fit = model_ar.fit()
        forecast_ar = model_ar_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
        rmse_ar = np.sqrt(mean_squared_error(test_data, forecast_ar))
        if rmse_ar < best_rmse:
            best_rmse = rmse_ar
            best_model = 'AR'
            best_forecast = forecast_ar

        # Store the best model and its metrics for the current column
        best_models[column] = {
            'Best Model': best_model,
            'RMSE': best_rmse,
            'MAE': mean_absolute_error(test_data, best_forecast),
            'MAPE': np.mean(np.abs((test_data - best_forecast) / test_data)) * 100
        }

        # Save the best model for the column
        if best_model == 'ARIMA':
            joblib.dump(model_arima_fit, os.path.join('best_models_1', f'{column}.pkl'))
        elif best_model == 'SARIMA':
            joblib.dump(model_sarima_fit, os.path.join('best_models_1', f'{column}.pkl'))
        elif best_model == 'SARIMAX':
            joblib.dump(model_sarimax_fit, os.path.join('best_models_1', f'{column}.pkl'))
        elif best_model == 'ETS':
            joblib.dump(model_ets_fit, os.path.join('best_models_1', f'{column}.pkl'))
        elif best_model == 'AR':
            joblib.dump(model_ar_fit, os.path.join('best_models_1', f'{column}.pkl'))
    except Exception as e:
        print(f"An error occurred for {column}: {e}")

# Convert the dictionary to a DataFrame
best_models_df = pd.DataFrame(best_models).T

# Save the best models and their metrics to a CSV file
best_models_df.to_csv('best_models1.csv')

# Print the best models and their metrics
print("Best models and their metrics:")
print(best_models_df)


