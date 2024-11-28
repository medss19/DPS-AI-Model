import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.metrics import mean_squared_error

# Load data
filtered_data = pd.read_pickle('subset.pkl')

# Filter out the 'Summe' row and keep only monthly data
filtered_data = filtered_data[filtered_data['MONAT'] != 'Summe']

# Convert MONAT column to datetime
filtered_data['date'] = pd.to_datetime(filtered_data['MONAT'], format='%Y%m')
filtered_data.set_index('date', inplace=True)

# Sort the index
filtered_data.sort_index(inplace=True)

# Prepare time series
y = filtered_data['WERT']

# SARIMA Model
sarima_model = SARIMAX(y, 
                       order=(1, 1, 1),  
                       seasonal_order=(1, 1, 1, 12),  
                       enforce_stationarity=False,
                       enforce_invertibility=False)

# Fit the model
sarima_result = sarima_model.fit()

# Forecast for next 12 months
forecast_steps = 12
forecast = sarima_result.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=y.index[-1] + pd.DateOffset(months=1), 
                                periods=forecast_steps, 
                                freq='M')

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y.index, y.values, label='Historical Data', color='blue')
plt.plot(forecast_index, forecast.predicted_mean, label='Forecasted Data', color='red')
plt.fill_between(forecast_index, 
                 forecast.predicted_mean - 1.96 * forecast.se_mean, 
                 forecast.predicted_mean + 1.96 * forecast.se_mean, 
                 color='pink', alpha=0.3, label='Confidence Interval')
plt.title('Forecast of Alcohol-related Accidents')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('time-series.png', bbox_inches='tight', dpi=300)
plt.show()