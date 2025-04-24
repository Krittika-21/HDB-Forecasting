# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 09:43:14 2025

@author: kritt
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import io
import requests
import logging

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy loading model and encoder from Azure Blob Storage
model = None
encoder = None

# Define your actual SAS URLs
model_sas_url = "https://hdbmodel.blob.core.windows.net/models/RFGmodel.pkl?sp=r&st=2025-04-24T02:01:07Z&se=2025-05-30T10:01:07Z&spr=https&sv=2024-11-04&sr=b&sig=2ZO0uL7jIvpq4BV5W%2BxKRxDzhECtA7v%2F%2FN%2F8muY7xJs%3D"
encoder_sas_url = "https://hdbmodel.blob.core.windows.net/models/encoder.pkl?sp=r&st=2025-04-24T01:58:46Z&se=2025-05-30T09:58:46Z&spr=https&sv=2024-11-04&sr=b&sig=oDbSh9UUN3AseMHaOzfTBLjd%2BXPU2I5z15XoIRqi1OY%3D"

def load_model_and_encoder():
    global model, encoder
    if model is None or encoder is None:
        try:
            logger.info("Downloading model and encoder from Azure Blob Storage...")
            model_response = requests.get(model_sas_url)
            encoder_response = requests.get(encoder_sas_url)

            model_response.raise_for_status()
            encoder_response.raise_for_status()

            model = pickle.load(io.BytesIO(model_response.content))
            encoder = pickle.load(io.BytesIO(encoder_response.content))
            logger.info("Model and encoder loaded successfully.")

        except requests.exceptions.RequestException as req_err:
            logger.error(f"Error fetching from Azure Blob Storage: {req_err}")
        except pickle.UnpicklingError as pkl_err:
            logger.error(f"Error unpickling model or encoder: {pkl_err}")
        except Exception as e:
            logger.error(f"Unexpected error loading model/encoder: {e}")

# Startup page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    load_model_and_encoder()
    if model is None or encoder is None:
        return render_template('index.html', prediction_text="Error: Model could not be loaded. Check server logs.")

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

    try:
        input_features = [year, month, flat_age, remaining_lease, floor_area, town, flat_type, flat_model]
        flat_model_encoded = encoder.transform([input_features[5:]]).flatten()
        features = np.array([year, month, flat_age, remaining_lease, floor_area] + list(flat_model_encoded)).reshape(1, -1)
        prediction = model.predict(features)
        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f'Predicted resale price of flat is ${output}')
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return render_template('index.html', prediction_text="Prediction failed due to internal error.")

@app.route('/predict_api_json', methods=['POST'])
def predict_api_json():
    load_model_and_encoder()
    if model is None or encoder is None:
        return jsonify({'error': 'Model could not be loaded. Check server logs for details.'}), 500

    try:
        data = request.get_json(force=True)

        year = data.get('year')
        month = data.get('month')
        flat_age = data.get('flat_age')
        remaining_lease = data.get('remaining_years_lease')
        floor_area = data.get('floor_area_sqm')
        town = data.get('town')
        flat_type = data.get('flat_type')
        flat_model = data.get('flat_model')

        encoded = encoder.transform([[town, flat_type, flat_model]])
        input_arr = np.concatenate(([year, month, flat_age, remaining_lease, floor_area], encoded.flatten()))
        prediction = model.predict([input_arr])
        output = round(prediction[0], 2)

        logger.info(f"API Prediction: ${output}")
        return jsonify({'Predicted Resale Price': output})

    except Exception as e:
        logger.error(f"API prediction failed: {e}")
        return jsonify({'error': 'Prediction failed due to internal error.'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
