from flask import Flask, request, jsonify
from sarima_model import get_forecast_for_month  # Import the forecast function
from datetime import datetime

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    year = data.get('year')
    month = data.get('month')
    
    if not year or not month:
        return jsonify({"error": "Year and month are required"}), 400
    
    try:
        # Fetch prediction for the given year and month
        forecast = get_forecast_for_month(year=year, month=month, steps=12)
        
        return jsonify(forecast)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
