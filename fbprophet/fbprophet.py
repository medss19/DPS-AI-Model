import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet as ProphetModel
from sklearn.metrics import mean_absolute_error, mean_squared_error

def prepare_data(file_path):
    """Load and prepare dataset."""
    # Read CSV with proper parsing
    df = pd.read_csv(file_path)
    
    # Apply filters
    df = df[df['MONATSZAHL'] == 'Alkoholunfälle']
    df = df[df['AUSPRAEGUNG'] == 'insgesamt']
    df = df[df['MONAT'] != 'Summe']
    
    # Convert MONAT to datetime
    df['date'] = pd.to_datetime(df['MONAT'].astype(str).str.zfill(6), format='%Y%m')
    
    # Remove any NaN values in WERT column
    df = df.dropna(subset=['WERT'])
    
    # Sort by date
    df = df.sort_values('date')
    
    return df

def plot_forecast(model, forecast, train_df, actual_df):
    """Plot forecast and historical data."""
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.plot(train_df['ds'], train_df['y'], label='Training Data', color='blue')
    
    # Plot actual data
    plt.plot(actual_df['ds'], actual_df['y'], label='Actual Data', color='green', linestyle='--')
    
    # Plot forecast
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
    plt.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        color='red',
        alpha=0.3,
        label='Forecast Uncertainty'
    )
    
    # Add legend and labels
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Accidents (WERT)')
    plt.title('Forecast vs. Actual Data')
    plt.savefig('prophet.png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_prediction_vs_actual(actual_2021, forecast_2021):
    """Plot prediction vs. actual data for 2021."""
    plt.figure(figsize=(10, 6))
    
    # Plot actual data
    plt.plot(actual_2021['date'], actual_2021['WERT'], label='Actual 2021 Data', marker='o', color='green')
    
    # Plot forecasted data
    plt.plot(forecast_2021['ds'], forecast_2021['yhat'], label='Forecasted 2021 Data', marker='o', color='red')
    
    # Add legend and labels
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Accidents (WERT)')
    plt.title('Prediction vs. Actual for 2021')
    plt.savefig('prophet_metric.png', bbox_inches='tight', dpi=300)
    plt.show()

def main():
    try:
        # Load datasets
        print("Loading and preparing data...")
        df_train = prepare_data('../datasets/subset.csv')
        df_actual = prepare_data('../datasets/dataset.csv')
        
        print("\nTraining data summary:")
        print(f"Date range: {df_train['date'].min()} to {df_train['date'].max()}")
        print(f"Total records: {len(df_train)}")
        
        print("\nActual data summary:")
        print(f"Date range: {df_actual['date'].min()} to {df_actual['date'].max()}")
        print(f"Total records: {len(df_actual)}")
        
        # Create Prophet DataFrames
        prophet_train_df = pd.DataFrame({
            'ds': df_train['date'],
            'y': df_train['WERT']
        })
        
        prophet_actual_df = pd.DataFrame({
            'ds': df_actual['date'],
            'y': df_actual['WERT']
        })
        
        # Train model
        print("\nTraining model...")
        model = ProphetModel(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        model.fit(prophet_train_df)
        
        # Generate forecast
        future = model.make_future_dataframe(periods=24, freq='M')  # Extend to cover 2021 and beyond
        forecast = model.predict(future)
        
        # Extract forecast for 2021
        forecast_2021 = forecast[forecast['ds'].dt.year == 2021]
        actual_2021 = df_actual[df_actual['date'].dt.year == 2021]
        
        print("\nForecast for 2021:")
        print(forecast_2021[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_string())
        
        # Calculate metrics
        mae = mean_absolute_error(actual_2021['WERT'], forecast_2021['yhat'])
        rmse = np.sqrt(mean_squared_error(actual_2021['WERT'], forecast_2021['yhat']))
        print(f"\nMean Absolute Error (2021): {mae:.2f}")
        print(f"Root Mean Squared Error (2021): {rmse:.2f}")
        
        # Plot forecast
        print("\nGenerating plots...")
        plot_forecast(model, forecast, prophet_train_df, prophet_actual_df)
        
        # Plot prediction vs. actual
        plot_prediction_vs_actual(actual_2021, forecast_2021)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
