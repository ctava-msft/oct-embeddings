from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
)
from azure.ai.ml.entities import OnlineRequestSettings, ProbeSettings
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential
)
import time
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

required_vars = [
    "SUBSCRIPTION_ID",
    "RESOURCE_GROUP",
    "WORKSPACE_NAME",
    "MODEL_NAME"
    ]

for var in required_vars:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
        raise ValueError(f"Missing required environment variable: {var}")
        
try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

try:


    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace_name = os.getenv("WORKSPACE_NAME")
    model_name = os.getenv("MODEL_NAME")
    instance_type = os.getenv("INSTANCE_TYPE")

    workspace_ml_client = MLClient(
        credential,
        subscription_id,
        resource_group,
        workspace_name
    )


    registry_ml_client = MLClient(
        credential,
        subscription_id,
        resource_group,
        registry_name="azureml",
    )

    foundation_model = registry_ml_client.models.get(name=model_name, label="latest")
    print(
        f"\n\nUsing model name: {foundation_model.name}, version: {foundation_model.version}, id: {foundation_model.id} for inferencing"
    )

    # Endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name
    timestamp = int(time.time())
    online_endpoint_name = model_name + "-" + str(timestamp)
    # Create an online endpoint
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        description="Online endpoint for "
        + foundation_model.name,
        auth_mode="key",
    )
    workspace_ml_client.begin_create_or_update(endpoint).wait()


    deployment_name = f"{model_name[:16]}-mlflow-deploy"

    # Create a deployment
    demo_deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=online_endpoint_name,
        model=foundation_model.id,
        instance_type=instance_type,
        instance_count=1,
        request_settings=OnlineRequestSettings(
            max_concurrent_requests_per_instance=1,
            request_timeout_ms=90000,
            max_queue_wait_ms=500,
        ),
        liveness_probe=ProbeSettings(
            failure_threshold=49,
            success_threshold=1,
            timeout=299,
            period=180,
            initial_delay=180,
        ),
        readiness_probe=ProbeSettings(
            failure_threshold=10,
            success_threshold=1,
            timeout=10,
            period=10,
            initial_delay=10,
        ),
    )
    workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()
    endpoint.traffic = {deployment_name: 100}
    workspace_ml_client.begin_create_or_update(endpoint).result()


except Exception as ex:
    print(ex)