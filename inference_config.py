# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 09:23:09 2025

@author: kritt
"""

from azureml.core import Environment
from azureml.core.model import InferenceConfig

# Define the environment
env = Environment.from_conda_specification(name="myenv", file_path="environment.yml")

# Create the inference configuration
inference_config = InferenceConfig(entry_script="score.py", environment=env)
