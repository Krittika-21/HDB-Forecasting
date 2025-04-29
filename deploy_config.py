# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 09:24:14 2025

@author: kritt
"""

from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig

# Define the deployment configuration
deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    tags={"area": "hdb forecasting"},
    description="HDB Resale Price Prediction Model"
)
