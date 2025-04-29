from azureml.core import Workspace, Environment, Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig
from azureml.core.authentication import AzureCliAuthentication

# Use Azure CLI Authentication (which uses the same login from az login)
cli_auth = AzureCliAuthentication()

# Authenticate and load the workspace using the CLI authentication
workspace = Workspace.from_config(auth=cli_auth)

# Register the model (make sure the model file exists on your local machine)
model = Model.register(workspace=workspace,
                       model_name='hdb_forecast_model',
                       model_path='C:/Users/kritt/OneDrive/Desktop/ST Logs/hdb forecast model/RFGmodel.pkl')  # Path to the model on your local machine

# Create the environment (based on the environment.yml you prepared)
environment = Environment.from_conda_specification(name="hdb-env", file_path="environment.yml")

# Create an inference configuration
inference_config = InferenceConfig(entry_script="score.py", environment=environment)

# Specify the ACI deployment configuration
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy the model as a web service to Azure Container Instances (ACI)
service = Model.deploy(workspace=workspace,
                       name='hdb-forecast-service',
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=aci_config)

# Wait for the deployment to complete
service.wait_for_deployment(show_output=True)

print(f"Service deployed at: {service.scoring_uri}")
