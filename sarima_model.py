import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from itertools import product
import warnings
from datetime import datetime

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load and preprocess data
df = pd.read_pickle('datasets/subset.pkl')
df = df[df['AUSPRAEGUNG'] == 'insgesamt']
df = df[df['MONATSZAHL'] == 'Alkoholunfälle']
df = df[df['MONAT'] != 'Summe']
df = df[pd.to_numeric(df['MONAT'], errors='coerce').notna()]
df['date'] = pd.to_datetime(df['MONAT'], format='%Y%m')
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

# Pre-compute best parameters using data up to 2020
train_data = df[df.index.year < 2020]
param_grid = {
    'order': list(product(range(0, 3), range(0, 2), range(0, 3))),
    'seasonal_order': list(product(range(0, 2), range(0, 2), range(0, 2), [12]))
}

# Find best parameters once
best_aic = float('inf')
BEST_ORDER = None
BEST_SEASONAL_ORDER = None

for order in param_grid['order']:
    for seasonal_order in param_grid['seasonal_order']:
        try:
            model = SARIMAX(train_data['WERT'],
                          order=order,
                          seasonal_order=seasonal_order,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            results = model.fit(disp=False)
            
            if results.aic < best_aic:
                best_aic = results.aic
                BEST_ORDER = order
                BEST_SEASONAL_ORDER = seasonal_order
        except:
            continue

def get_forecast_for_month(year, month, steps=12):
    """Get forecast for a specific month and year using pre-computed parameters."""
    # Filter data for training
    train_data = df[df.index.year < year]
    
    # Use pre-computed best parameters
    model = SARIMAX(train_data['WERT'],
                    order=BEST_ORDER,
                    seasonal_order=BEST_SEASONAL_ORDER,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    final_result = model.fit(disp=False)
    
    # Generate forecast
    forecast = final_result.get_forecast(steps=steps)
    forecast_values = forecast.predicted_mean.values
    
    # Prepare forecast dates
    forecast_dates = [forecast.predicted_mean.index[i].strftime('%Y-%m-%d') 
                     for i in range(steps)]
    
    # Check if requested date is in forecast range
    requested_date = f"{year}-{str(month).zfill(2)}-01"
    if requested_date in forecast_dates:
        index_of_requested_date = forecast_dates.index(requested_date)
        prediction_value = forecast_values[index_of_requested_date]
        return {"prediction": float(prediction_value)}
    else:
        return {"error": "Forecast for the given year and month is not available."}, 404