# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 10:00:08 2025

File to give json file using post request
Result value for data in json file can be printed to console if request.py file run while app is running. 

@author: kritt
"""

import requests

url = 'http://localhost:5000/predict_api_json'

# Example data to be passed as json format 
payload = {
    'year': 2024,
    'month': 4,
    'flat_age': 10,
    'remaining_years_lease': 89,
    'floor_area_sqm': 90,
    'town': 'ANG MO KIO',
    'flat_type': '4 ROOM',
    'flat_model': 'Improved'
}

r = requests.post(url, json=payload)    
  
print(r.json())