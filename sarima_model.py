import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from itertools import product
import warnings
from datetime import datetime

# Suppress specific warnings by using the correct warning class
warnings.filterwarnings("ignore", category=UserWarning)

# Load and preprocess data (same as your original script)
df = pd.read_pickle('datasets/subset.pkl')
df = df[df['AUSPRAEGUNG'] == 'insgesamt']
df = df[df['MONATSZAHL'] == 'Alkoholunfälle']
df = df[df['MONAT'] != 'Summe']
df = df[pd.to_numeric(df['MONAT'], errors='coerce').notna()]
df['date'] = pd.to_datetime(df['MONAT'], format='%Y%m')
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

# Function to train the SARIMA model
def train_sarima_model(data):
    param_grid = {
        'order': list(product(range(0, 3), range(0, 2), range(0, 3))),
        'seasonal_order': list(product(range(0, 2), range(0, 2), range(0, 2), [12]))
    }

    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    
    for order in param_grid['order']:
        for seasonal_order in param_grid['seasonal_order']:
            try:
                model = SARIMAX(data['WERT'], 
                                order=order, 
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                results = model.fit(disp=False)
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = order
                    best_seasonal_order = seasonal_order
            except:
                continue

    # Train the model with the best parameters
    final_model = SARIMAX(data['WERT'], 
                          order=best_order,  
                          seasonal_order=best_seasonal_order,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
    final_result = final_model.fit()
    
    return final_result

# Function to forecast for a specific month and year
def get_forecast_for_month(year, month, steps=12):
    # Filter the data for training the model (up to the requested year and month)
    data = df[df.index.year < year]
    
    # Train the model with the available data
    model = train_sarima_model(data)
    
    # Forecast for the requested number of steps ahead
    forecast = model.get_forecast(steps=steps)
    forecast_values = forecast.predicted_mean.values  # Get all predicted values
    
    # Prepare the forecast dates as list
    forecast_dates = [forecast.predicted_mean.index[i].strftime('%Y-%m-%d') for i in range(steps)]
    
    # Check if the requested year and month are within the forecasted range
    requested_date = f"{year}-{str(month).zfill(2)}-01"
    
    if requested_date in forecast_dates:
        index_of_requested_date = forecast_dates.index(requested_date)
        prediction_value = forecast_values[index_of_requested_date]
        return {"prediction": prediction_value}
    else:
        return {"error": "Forecast for the given year and month is not available."}, 404



# Example function call
# forecast = get_forecast_for_month(2021, 5)
# print(forecast)
