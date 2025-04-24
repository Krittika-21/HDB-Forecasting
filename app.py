# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 09:43:14 2025

@author: kritt
"""

import numpy as np
# import pandas as pd
from flask import Flask, request, jsonify, render_template   # render_template helps to redirect to the initial homepage we have
import pickle
# import joblib

import io
import requests

# Initialize flask app
app = Flask(__name__)

# Model and encoder loading for Flask App

# model = pickle.load(open('RFGmodel.pkl', 'rb'))
# encoder = pickle.load(open('encoder.pkl', 'rb'))

# Model and encoder blobs loading from Azure Blob Storage (load from Azure)

model_sas_url = "https://hdbmodel.blob.core.windows.net/models/RFGmodel.pkl?sp=r&st=2025-04-24T02:01:07Z&se=2025-05-30T10:01:07Z&spr=https&sv=2024-11-04&sr=b&sig=2ZO0uL7jIvpq4BV5W%2BxKRxDzhECtA7v%2F%2FN%2F8muY7xJs%3D"
encoder_sas_url = "https://hdbmodel.blob.core.windows.net/models/encoder.pkl?sp=r&st=2025-04-24T01:58:46Z&se=2025-05-30T09:58:46Z&spr=https&sv=2024-11-04&sr=b&sig=oDbSh9UUN3AseMHaOzfTBLjd%2BXPU2I5z15XoIRqi1OY%3D"

model_bytes = requests.get(model_sas_url).content
encoder_bytes = requests.get(encoder_sas_url).content

model = pickle.load(io.BytesIO(model_bytes))
encoder = pickle.load(io.BytesIO(encoder_bytes))

# define route notes to route API url to tell app where it should be directed to

#Start up page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    '''
    For rendering results on HTML GUI
    '''
    # Get features from the form
    try:
        year = int(request.form['year'])
        month = int(request.form['month'])
        flat_age = int(request.form['flat_age'])
        remaining_lease = int(request.form['remaining_years_lease'])
        floor_area = int(request.form['floor_area_sqm'])
        town = request.form['town']
        flat_type = request.form['flat_type']
        flat_model = request.form['flat_model']
        
    except KeyError:
        return render_template('index.html', prediction_text="Please provide valid input values.")
    
    # Combine features in the same order as the model expects
    input_features = [year, month, flat_age, remaining_lease, floor_area, town, flat_type, flat_model]
    
    # Transform the categorical data with the encoder (make sure it expects the same number of features)
    flat_model_encoded = encoder.transform([input_features[5:]]).flatten()
    
    # Concatenate the numerical features with the encoded flat_model
    features = np.array([year, month, flat_age, remaining_lease, floor_area] + list(flat_model_encoded)).reshape(1, -1)

    # Prediction using the model
    prediction = model.predict(features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Predicted resale price of flat is ${output}')

@app.route('/predict_api_json', methods =['POST'])
def predict_api_json():
    '''
    For direct API calls (by passing in json file)
    '''
    # Get data from json request
    data = request.get_json(force=True)
    
    # Extract features from JSON
    year = data.get('year')
    month = data.get('month')
    flat_age = data.get('flat_age')
    remaining_lease = data.get('remaining_years_lease')
    floor_area = data.get('floor_area_sqm')
    town = data.get('town')
    flat_type = data.get('flat_type')
    flat_model = data.get('flat_model')
    
    # One-hot encode categorical features
    encoded = encoder.transform([[town, flat_type, flat_model]])
    
    # Combine with numerical features (using numpy's concatenate function)
    input_arr = np.concatenate(([year, month, flat_age, remaining_lease, floor_area], encoded.flatten()))
    
    # Prediction by model
    prediction = model.predict([input_arr])
    output = prediction[0]
    
    # Log the prediction to console
    print(f"Prediction: The predicted resale price is ${output}")
    
    return jsonify({'Predicted Resale Price': round(output, 2)})

if __name__ == "__main__":
    app.run(debug=True)
