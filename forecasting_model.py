# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 09:02:11 2025

@author: krittika
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# pickle library to save model
import pickle 

# Load data
hdb_data = pd.read_csv(r"C:\Users\kritt\OneDrive\Desktop\ST Logs\hdb forecast model\sg-resale-flat-prices-2017-onwards.csv\sg-resale-flat-prices-2017-onwards.csv")
#%% FUNCTION DECLARATIONS

"""
    parse_remaining_lease() function: extract years and months from remaining_lease column
"""
def parse_remaining_lease(remaining_lease_str):
    years = 0
    months = 0
    
    # Extract years
    if "year" in remaining_lease_str:
        years = int(remaining_lease_str.split("year")[0].strip())
    
    # Extract months if present
    if "month" in remaining_lease_str:
        parts_of_str =  remaining_lease_str.split("year")
        if len(parts_of_str) > 1:
            months_part = parts_of_str[1]
            months = int(months_part.strip().split("month")[0].strip().replace('s', ''))
            
    # Combine years and months into float value
    return round(years + months / 12, 2)

"""
    evaluate_model() function: evaluate model performance
"""
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
#%%
# Display general statistical data

# Set option to display all columns
pd.set_option('display.max_columns', None)

# Display general statistics of data
print(hdb_data.info())
print(hdb_data.head())
print(hdb_data.describe())
#%%

# Feature Engineering (create new features)

# Convert month col to datetime format
hdb_data["month"] = pd.to_datetime(hdb_data["month"])
hdb_data.rename(columns = {"month":"timestamp"}, inplace=True)

# Extract year and month number

#hdb_data["year"] = hdb_data["month"].dt.year
hdb_data.insert(1, "year", hdb_data["timestamp"].dt.year)

hdb_data.insert(2, "month", hdb_data["timestamp"].dt.month)

# Find flat age and remaining years of lease
hdb_data.insert(3, "flat_age", hdb_data['year'] - hdb_data['lease_commence_date'])
hdb_data.insert(4, "remaining_years_lease", hdb_data["remaining_lease"].apply(parse_remaining_lease))
#%%

# One-hot encoding for categorical columns (town, flat type, flat model)

# Find number of unique values in each categorical column
for cat_col in hdb_data[["town", "flat_type", "flat_model"]]:
    print(f"Number of unique values in {cat_col} : {len(hdb_data[cat_col].unique())}")

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(hdb_data[['town', 'flat_type', 'flat_model']])
one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(['town', 'flat_type', 'flat_model']))
hdb_data = pd.concat([hdb_data, one_hot_encoded_df], axis=1)
hdb_data = hdb_data.drop(['town', 'flat_type', 'flat_model'], axis=1)
#%% MODEL DEVELOPMENT (1)- simple Linear Regression, Random Forest Regressor

# Target Feature
y = hdb_data['resale_price']

# Features to be selected including encoded columns from encoded df
selected_cols = ['year', 'month', 'flat_age', 'remaining_years_lease', 'floor_area_sqm'] + list(one_hot_encoded_df.columns)
X = hdb_data[selected_cols]

# Train-Test split (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Simple linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# Simple Random Forest Regressor model
model2 = RandomForestRegressor(random_state=42)
model2.fit(X_train, y_train)
#%% Model Evaluation

# print("Linear Regression Model Evaluation Scores:")
# y_pred = model.predict(X_test)
# evaluate_model(y_test, y_pred)

print("Random Forest Regressor Evaluation Scores:")
y_pred2 = model2.predict(X_test)
evaluate_model(y_test, y_pred2)
#%% Visualization of predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred2, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Resale Price")
plt.ylabel("Predicted Resale Price")
plt.title("Random Forest: Actual vs Predicted")
plt.grid(True)
plt.show()
#%% Save model and encoder using pickle
pickle.dump(model2, open('C:/Users/kritt/OneDrive/Desktop/ST Logs/hdb forecast model/RFGmodel.pkl', 'wb'))
pickle.dump(encoder, open('C:/Users/kritt/OneDrive/Desktop/ST Logs/hdb forecast model/encoder.pkl', 'wb'))