from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential
)
import logging
import os
from dotenv import load_dotenv
import time
import os
import urllib
from zipfile import ZipFile
import base64
import json


load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

required_vars = [
    "SUBSCRIPTION_ID",
    "RESOURCE_GROUP",
    "WORKSPACE_NAME",
    "ENDPOINT_NAME",
    "DEPLOYMENT_NAME",
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

subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")

workspace_ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

registry_ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    registry_name="azureml",
)

# Change to a different location if you prefer
dataset_parent_dir = "./data"

# Create data folder if it doesnt exist.
os.makedirs(dataset_parent_dir, exist_ok=True)

# Download data
download_url = "https://automlsamplenotebookdata.blob.core.windows.net/image-classification/fridgeObjects.zip"

# Extract current dataset name from dataset url
dataset_name = os.path.split(download_url)[-1].split(".")[0]
# Get dataset path for later use
dataset_dir = os.path.join(dataset_parent_dir, dataset_name)

# Get the data zip file path
data_file = os.path.join(dataset_parent_dir, f"{dataset_name}.zip")

# Download the dataset
urllib.request.urlretrieve(download_url, filename=data_file)

# Extract files
with ZipFile(data_file, "r") as zip:
    print("extracting files...")
    zip.extractall(path=dataset_parent_dir)
    print("done")
# Delete zip file
os.remove(data_file)



sample_image_1 = os.path.join(dataset_dir, "milk_bottle", "99.jpg")
sample_image_2 = os.path.join(dataset_dir, "can", "1.jpg")


def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()


request_json = {
    "input_data": {
        "columns": ["image", "text"],
        "index": [0, 1],
        "data": [
            [
                base64.encodebytes(read_image(sample_image_1)).decode("utf-8"),
                "",
            ],  # the "text" column should contain empty string
            [base64.encodebytes(read_image(sample_image_2)).decode("utf-8"), ""],
        ],
    }
}

# Create request json
request_file_name = "sample_request_data.json"
with open(request_file_name, "w") as request_file:
    json.dump(request_json, request_file)


# Score the sample_score.json file using the online endpoint with the azureml endpoint invoke method
response = workspace_ml_client.online_endpoints.invoke(
    endpoint_name=os.getenv("ENDPOINT_NAME"),
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    request_file=request_file_name,
)
print(f"raw response: {response}\n")