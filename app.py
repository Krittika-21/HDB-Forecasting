# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 09:43:14 2025

@author: kritt
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template   # render_template helps to redirect to the initial homepage we have
# import pickle
import joblib
import requests
import json

# Initialize flask app
app = Flask(__name__)

# Azure ML endpoint url (REST endpoint) and key (in consume tab)
endpoint_url = "https://hdb-forecasting-kphte.southeastasia.inference.ml.azure.com/score"
api_key = "9cJ30e5Snopflm4zFKnljXZNE9PHn1eowjk6LqMXI1tLIC0ylF6qJQQJ99BDAAAAAAAAAAAAINFRAZML1vKh"

# model = joblib.load(open('RFGmodel.pkl', 'rb'))
# encoder = joblib.load(open('encoder.pkl', 'rb'))

#%% AZURE BLOB STORAGE (not used)

# import os
# from azure.storage.blob import BlobServiceClient
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
# CONTAINER_NAME = "models"

# # Initialize Azure Blob client
# blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

# # Helper to download blob and load as pickle object
# def load_pickle_from_blob(blob_name):
#     blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
#     blob_data = blob_client.download_blob().readall()
#     return pickle.loads(blob_data)

# # Load model and encoder
# model = load_pickle_from_blob("RFGmodel.pkl")
# encoder = load_pickle_from_blob("encoder.pkl")
#%%

# define route notes to route API url to tell app where it should be directed to

#Start up page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST']) #ADDED 'GET'
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
    
    # Prepare the payload (ADDED)
    input_payload = {
        "year": year,
        "month": month,
        "flat_age": flat_age,
        "remaining_years_lease": remaining_lease,
        "floor_area_sqm": floor_area,
        "town": town,
        "flat_type": flat_type,
        "flat_model": flat_model
    }
    
    # Set headers (ADDED)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # Send POST request to Azure ML endpoint (ADDED)
    response = requests.post(endpoint_url, headers=headers, data=json.dumps(input_payload))

    if response.status_code == 200:
        try:
            prediction = response.json().get('predicted_resale_price')
            return render_template('index.html', prediction_text=f"Predicted resale price is ${round(prediction, 2)}")
        except (ValueError, KeyError) as e:
            return render_template('index.html', prediction_text="Invalid response format from model.")
    else:
        return render_template('index.html', prediction_text=f"Azure ML error {response.status_code}: {response.text}")
    
    # # Combine features in the same order as the model expects
    # input_features = [year, month, flat_age, remaining_lease, floor_area, town, flat_type, flat_model]
    
    # # Transform the categorical data with the encoder (make sure it expects the same number of features)
    # flat_model_encoded = encoder.transform([input_features[5:]]).flatten()
    
    # # Concatenate the numerical features with the encoded flat_model
    # features = np.array([year, month, flat_age, remaining_lease, floor_area] + list(flat_model_encoded)).reshape(1, -1)

    # # Prediction using the model
    # prediction = model.predict(features)
    # output = round(prediction[0], 2)

    # return render_template('index.html', prediction_text=f'Predicted resale price of flat is ${output}')

# @app.route('/predict_api_json', methods =['POST'])
# def predict_api_json():
#     '''
#     For direct API calls (by passing in json file)
#     '''
#     # Get data from json request
#     data = request.get_json(force=True)
    
#     # Extract features from JSON
#     year = data.get('year')
#     month = data.get('month')
#     flat_age = data.get('flat_age')
#     remaining_lease = data.get('remaining_years_lease')
#     floor_area = data.get('floor_area_sqm')
#     town = data.get('town')
#     flat_type = data.get('flat_type')
#     flat_model = data.get('flat_model')
    
#     # One-hot encode categorical features
#     encoded = encoder.transform([[town, flat_type, flat_model]])
    
#     # Combine with numerical features (using numpy's concatenate function)
#     input_arr = np.concatenate(([year, month, flat_age, remaining_lease, floor_area], encoded.flatten()))
    
#     # Prediction by model
#     prediction = model.predict([input_arr])
#     output = prediction[0]
    
#     # Log the prediction to console
#     print(f"Prediction: The predicted resale price is ${output}")
    
#     return jsonify({'Predicted Resale Price': round(output, 2)})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
