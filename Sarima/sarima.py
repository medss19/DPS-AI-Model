import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from itertools import product
import warnings

# Suppress specific warnings by using the correct warning class
warnings.filterwarnings("ignore", category=UserWarning)

def optimize_sarima(train, test, param_grid):
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    
    for order in param_grid['order']:
        for seasonal_order in param_grid['seasonal_order']:
            try:
                model = SARIMAX(train['WERT'], 
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
    
    return best_order, best_seasonal_order

# Parameter grid for search
param_grid = {
    'order': list(product(range(0, 3), range(0, 2), range(0, 3))),
    'seasonal_order': list(product(range(0, 2), range(0, 2), range(0, 2), [12]))
}

# Load and preprocess data (same as your original script)
df = pd.read_pickle('../datasets/subset.pkl')
df = df[df['AUSPRAEGUNG'] == 'insgesamt']
df = df[df['MONATSZAHL'] == 'Alkoholunfälle']
df = df[df['MONAT'] != 'Summe']
df = df[pd.to_numeric(df['MONAT'], errors='coerce').notna()]
df['date'] = pd.to_datetime(df['MONAT'], format='%Y%m')
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

# Split data
train_data = df[df.index.year < 2020]
test_data = df[df.index.year == 2020]
full_data = df

# Find best parameters
best_order, best_seasonal_order = optimize_sarima(train_data, test_data, param_grid)

print(f"Best Order: {best_order}")
print(f"Best Seasonal Order: {best_seasonal_order}")

# Fit final model with best parameters
final_model = SARIMAX(full_data['WERT'], 
                      order=best_order,  
                      seasonal_order=best_seasonal_order,
                      enforce_stationarity=False,
                      enforce_invertibility=False)
final_result = final_model.fit()

# Forecast 2021
forecast_2021 = final_result.get_forecast(steps=12)
forecast_2021_mean = forecast_2021.predicted_mean

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['WERT'], label='Historical Data')
plt.plot(forecast_2021_mean.index, forecast_2021_mean, label='2021 Forecast', color='red')
plt.fill_between(forecast_2021_mean.index, 
                 forecast_2021.conf_int()['lower WERT'], 
                 forecast_2021.conf_int()['upper WERT'], 
                 color='pink', alpha=0.3, label='Confidence Interval')
plt.title('Alcohol-related Accidents Forecast')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sarima.png', bbox_inches='tight', dpi=300)
plt.show()

# Print 2021 Forecast Values
print("\n2021 Monthly Forecast:")
print(forecast_2021_mean)

# Load ground truth data
ground_truth_df = pd.read_csv('../datasets/dataset.csv')

# Filter for Alkoholunfälle, insgesamt
ground_truth_df = ground_truth_df[ 
    (ground_truth_df['AUSPRAEGUNG'] == 'insgesamt') & 
    (ground_truth_df['MONATSZAHL'] == 'Alkoholunfälle')
]

# Remove 'Summe' rows and non-numeric MONAT entries
ground_truth_df = ground_truth_df[ground_truth_df['MONAT'] != 'Summe']
ground_truth_df = ground_truth_df[pd.to_numeric(ground_truth_df['MONAT'], errors='coerce').notna()]

# Convert MONAT to datetime
ground_truth_df['date'] = pd.to_datetime(ground_truth_df['MONAT'], format='%Y%m')
ground_truth_df.set_index('date', inplace=True)

# Filter 2021 data
actual_2021 = ground_truth_df.loc['2021-01-01':'2021-12-01', 'WERT']

# Calculate Error Metrics
mae = mean_absolute_error(actual_2021, forecast_2021_mean)
mse = mean_squared_error(actual_2021, forecast_2021_mean)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(actual_2021, forecast_2021_mean)

# Comparative Analysis
print("2021 Model Testing Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")

# Detailed Comparison
comparison_df = pd.DataFrame({
    'Actual': actual_2021,
    'Predicted': forecast_2021_mean,
    'Absolute Error': np.abs(actual_2021 - forecast_2021_mean),
    'Percentage Error': np.abs((actual_2021 - forecast_2021_mean) / actual_2021) * 100
})
print("\nDetailed Comparison:")
print(comparison_df)

# Visualization of Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(actual_2021.index, actual_2021, label='Actual', marker='o')
plt.plot(forecast_2021_mean.index, forecast_2021_mean, label='Predicted', marker='x')
plt.title('Actual vs Predicted Alcohol-related Accidents in 2021')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sarima_metric.png', bbox_inches='tight', dpi=300)
plt.show()
