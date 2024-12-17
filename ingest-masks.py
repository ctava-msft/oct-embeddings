from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential
)
import json
import os
from dotenv import load_dotenv
import requests
import logging
from tqdm.auto import tqdm
import base64
import cv2


from segment import Segment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

required_vars = [
    "SEARCH_KEY",
    "SEARCH_SERVICE_NAME",
    "SEARCH_INDEX_NAME",
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

SEARCH_KEY = os.getenv("SEARCH_KEY")
SEARCH_SERVICE_NAME = os.getenv("SEARCH_SERVICE_NAME")
SEARCH_INDEX_NAME = os.getenv("SEARCH_INDEX_NAME")
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")
API_VERSION = "2023-07-01-Preview"
endpoint_name=os.getenv("ENDPOINT_NAME")
deployment_name=os.getenv("DEPLOYMENT_NAME")

_REQUEST_FILE_NAME = "request.json"

def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()

def make_request_images(image_path, input_points, input_boxes, input_labels):
    request_json = {
        "input_data": {
            "columns": ["image", "input_points", "input_boxes", "input_labels", "multimask_output"],
            "data": [
                [
                    base64.encodebytes(read_image(image_path)).decode("utf-8"),
                    input_points,  # input_points
                    input_boxes,   # input_boxes
                    input_labels,  # input_labels
                    False          # multimask_output
                ],
            ],
        }
    }
    with open(_REQUEST_FILE_NAME, "wt") as f:
        json.dump(request_json, f)

ADD_DATA_REQUEST_URL = "https://{search_service_name}.search.windows.net/indexes/{index_name}/docs/index?api-version={api_version}".format(
    search_service_name=SEARCH_SERVICE_NAME,
    index_name=SEARCH_INDEX_NAME,
    api_version=API_VERSION,
)

dataset_parent_dir = "./data"
dataset_name = "OCT-5"
dataset_dir = os.path.join(dataset_parent_dir, dataset_name)

image_paths = [
    os.path.join(dp, f)
    for dp, dn, filenames in os.walk(dataset_dir)
    for f in filenames
    if os.path.splitext(f)[1] == ".jpeg"
]


try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()


workspace_ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

segmenter = Segment()
for idx, image_path in enumerate(tqdm(image_paths)):
    ID = idx
    FILENAME = image_path
    MAX_RETRIES = 3

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    input_points, input_boxes, input_labels = segmenter.process_image(image)

    # get embedding from endpoint
    embedding_request = make_request_images(image_path, input_points, input_boxes, input_labels)

    response = None
    request_failed = False
    IMAGE_EMBEDDING = None
    for r in range(MAX_RETRIES):
        try:
            response = workspace_ml_client.online_endpoints.invoke(
                endpoint_name=endpoint_name,
                deployment_name=deployment_name,
                request_file=_REQUEST_FILE_NAME,
            )
            response = json.loads(response)
            IMAGE_EMBEDDING = response[0]["image_features"]
            print(f"Successfully retrieved embeddings for image {FILENAME}.")
            break
        except Exception as e:
            print(f"Unable to get embeddings for image {FILENAME}: {e}")
            print(response)
            if r == MAX_RETRIES - 1:
                print(f"attempt {r} failed, reached retry limit")
                request_failed = True
            else:
                print(f"attempt {r} failed, retrying")

    # add embedding to index
    if IMAGE_EMBEDDING:
        add_data_request = {
            "value": [
                {
                    "id": str(ID),
                    "filename": FILENAME,
                    "imageEmbeddings": IMAGE_EMBEDDING,
                    "@search.action": "upload",
                }
            ]
        }
        response = requests.post(
            ADD_DATA_REQUEST_URL,
            json=add_data_request,
            headers={"api-key": SEARCH_KEY},
        )
        if response.status_code == 200:
            print(f"Successfully added data to index for image {FILENAME}.")
        else:
            print(f"Failed to add data to index for image {FILENAME}. Response: {response.content}")