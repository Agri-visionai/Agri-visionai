from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model and feature columns
model = pickle.load(open('model.pkl', 'rb'))
model_features = pickle.load(open('model_features.pkl', 'rb'))

@app.route('/')
def index():
    return "Welcome to the Agri-Vision AI Prediction API! Use the /predict endpoint to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract data from the request
    region = data['Region']
    soil_type = data['Soil_Type']
    crop = data['Crop']
    rainfall_mm = data['Rainfall_mm']
    temperature_celsius = data['Temperature_Celsius']
    fertilizer_used = data['Fertilizer_Used']
    irrigation_used = data['Irrigation_Used']
    weather_condition = data['Weather_Condition']
    days_to_harvest = data['Days_to_Harvest']
    
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
    app.run(debug=True)
