import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def prepare_data(file_path):
    """
    Reads the CSV file and processes the data according to the specific format:
    MONATSZAHL,AUSPRAEGUNG,JAHR,MONAT,WERT
    """
    df = pd.read_csv(file_path)
    
    # Remove summary rows where MONAT contains 'Summe'
    df = df[~df['MONAT'].astype(str).str.contains('Summe', case=False)]
    
    # For the remaining rows, MONAT column contains YYYYMM format
    # Extract year and month from MONAT column
    df['year'] = df['MONAT'].astype(str).str[:4]
    df['month'] = df['MONAT'].astype(str).str[4:6]
    
    # Create datetime column
    df['date'] = pd.to_datetime(df['year'] + df['month'], format='%Y%m')
    
    # Ensure WERT is numeric
    df['WERT'] = pd.to_numeric(df['WERT'], errors='coerce')
    
    # Drop any rows where conversion failed
    df = df.dropna(subset=['WERT', 'date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Set frequency to monthly
    df = df.set_index('date')
    df.index.freq = 'MS'
    
    return df[['WERT']]

def sarima_forecast(train, steps):
    """
    Train SARIMA model with robust error handling.
    """
    try:
        # Use simpler SARIMA parameters for better stability
        model = SARIMAX(train, 
                       order=(1, 1, 1),
                       seasonal_order=(1, 1, 1, 12),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        
        with np.errstate(all='ignore'):
            sarima_fit = model.fit(disp=False, method='nm', maxiter=500)
        
        forecast = sarima_fit.forecast(steps=steps)
        return forecast
        
    except Exception as e:
        print(f"SARIMA forecasting failed: {str(e)}")
        # Fallback to even simpler model
        model = SARIMAX(train, 
                       order=(1, 1, 0),
                       seasonal_order=(1, 0, 0, 12),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        
        with np.errstate(all='ignore'):
            sarima_fit = model.fit(disp=False, method='nm', maxiter=500)
        
        return sarima_fit.forecast(steps=steps)

def prophet_forecast(train, steps, forecast_dates):
    """
    Train Prophet model with robust error handling.
    """
    try:
        df_prophet = train.reset_index().rename(columns={'date': 'ds', 'WERT': 'y'})
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        model.fit(df_prophet)
        
        # Create future dataframe with specific dates
        future = pd.DataFrame({'ds': forecast_dates})
        forecast = model.predict(future)
        return pd.Series(forecast['yhat'].values, index=forecast_dates)
        
    except Exception as e:
        print(f"Prophet forecasting failed: {str(e)}")
        return None

def hybrid_forecast(train, actual):
    """
    Combines SARIMA and Prophet forecasts with improved error handling and weighting.
    """
    # Get the dates we need to forecast
    forecast_dates = actual.index
    steps = len(forecast_dates)
    
    print(f"Forecasting {steps} steps ahead...")
    print(f"Training data range: {train.index.min()} to {train.index.max()}")
    print(f"Forecast period: {forecast_dates.min()} to {forecast_dates.max()}")
    
    # Generate forecasts
    sarima_preds = sarima_forecast(train, steps)
    prophet_preds = prophet_forecast(train, steps, forecast_dates)
    
    if prophet_preds is None:
        print("Using SARIMA predictions only due to Prophet failure")
        hybrid_preds = sarima_preds
    else:
        # Simple average combination
        hybrid_preds = (sarima_preds + prophet_preds) / 2
    
    # Evaluate
    mae = mean_absolute_error(actual, hybrid_preds)
    rmse = np.sqrt(mean_squared_error(actual, hybrid_preds))
    print(f"Hybrid Model - Mean Absolute Error: {mae:.2f}")
    print(f"Hybrid Model - Root Mean Squared Error: {rmse:.2f}")
    
    # Plotting results
    plt.figure(figsize=(15, 7))
    
    # Plot training data
    plt.plot(train.index, train.values, label='Training Data', color='gray', alpha=0.5)
    
    # Plot forecasts and actual
    plt.plot(actual.index, actual.values, label='Actual', color='blue')
    plt.plot(sarima_preds.index, sarima_preds.values, label='SARIMA', color='orange', linestyle='dashed')
    if prophet_preds is not None:
        plt.plot(prophet_preds.index, prophet_preds.values, label='Prophet', color='green', linestyle='dashed')
    plt.plot(hybrid_preds.index, hybrid_preds.values, label='Hybrid', color='red')
    
    plt.legend()
    plt.title('Hybrid Forecast vs Actual (with Training Data)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('hybrid.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    return hybrid_preds

if __name__ == "__main__":
    try:
        # File paths
        subset_path = '../datasets/subset.csv'  # Training data (2000-2020)
        dataset_path = '../datasets/data.csv'   # Test data (2021)
        
        # Preprocess datasets
        df_train = prepare_data(subset_path)
        df_actual = prepare_data(dataset_path)
        
        # Extract actuals for 2021
        df_actual = df_actual['WERT']
        df_train = df_train['WERT']
        
        # Print data info
        print("Training data shape:", df_train.shape)
        print("Test data shape:", df_actual.shape)
        
        # Train and forecast
        hybrid_forecast(df_train, df_actual)
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())