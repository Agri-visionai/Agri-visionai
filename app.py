from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pickle
import pandas as pd
import os  # Import os for environment variables

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and feature columns
model = pickle.load(open('model.pkl', 'rb'))
model_features = pickle.load(open('model_features.pkl', 'rb'))  # Load the model features

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Safely access the keys using .get() and account for possible variations in naming
    region = data.get('Region') or data.get('region')
    soil_type = data.get('Soil_Type') or data.get('soil_type')
    crop = data.get('Crop') or data.get('crop')
    rainfall_mm = data.get('Rainfall_mm') or data.get('rainfall_mm')
    temperature_celsius = data.get('Temperature_Celsius') or data.get('temperature_celsius')
    fertilizer_used = data.get('Fertilizer_Used') or data.get('fertilizer_used')
    irrigation_used = data.get('Irrigation_Used') or data.get('irrigation_used')
    weather_condition = data.get('Weather_Condition') or data.get('weather_condition')
    days_to_harvest = data.get('Days_to_Harvest') or data.get('days_to_harvest')
    
    # Prepare the data for prediction
    input_data = pd.DataFrame({
        'Region': [region],
        'Soil_Type': [soil_type],
        'Crop': [crop],
        'Rainfall_mm': [rainfall_mm],
        'Temperature_Celsius': [temperature_celsius],
        'Fertilizer_Used': [fertilizer_used],
        'Irrigation_Used': [irrigation_used],
        'Weather_Condition': [weather_condition],
        'Days_to_Harvest': [days_to_harvest]
    })
    
    # One-hot encode the input features as done during training
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)
    input_data_encoded = input_data_encoded.reindex(columns=model_features, fill_value=0)

    # Make the prediction
    prediction = model.predict(input_data_encoded)

    return jsonify({'predicted_yield': prediction[0]})

if __name__ == '__main__':
    # Run the app on host 0.0.0.0 and port from the environment or default to 5000
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
