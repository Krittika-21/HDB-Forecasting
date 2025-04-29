# score.py

import os
import json
import numpy as np
import joblib

# Global variables for model and encoder
model = None
encoder = None

def init():
    global model
    global encoder

    model_dir = os.getenv('AZUREML_MODEL_DIR')

    # Go inside the "model_encoder" folder
    model_folder = os.path.join(model_dir, 'model_encoder') 

    # Load the model and encoder
    model_path = os.path.join(model_folder, 'RFGmodel_joblib.joblib')
    encoder_path = os.path.join(model_folder, 'encoder.pkl')

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

def run(raw_data):
    try:
        # Ensure input is parsed correctly
        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode('utf-8')
        data = json.loads(raw_data)

        # Extract features safely
        year = data.get('year')
        month = data.get('month')
        flat_age = data.get('flat_age')
        remaining_lease = data.get('remaining_years_lease')
        floor_area = data.get('floor_area_sqm')
        town = data.get('town')
        flat_type = data.get('flat_type')
        flat_model = data.get('flat_model')

        # Validate important fields
        if None in [year, month, flat_age, remaining_lease, floor_area, town, flat_type, flat_model]:
            return {"error": "Missing one or more required input fields."}

        # Encode categorical variables
        categorical_features = [[town, flat_type, flat_model]]
        encoded_features = encoder.transform(categorical_features).flatten()

        # Combine all features
        numerical_features = [year, month, flat_age, remaining_lease, floor_area]
        final_features = np.array(numerical_features + list(encoded_features)).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_features)
        predicted_price = round(float(prediction[0]), 2)

        return {"predicted_resale_price": predicted_price}

    except Exception as e:
        return {"error": str(e)}
