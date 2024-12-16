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
import numpy as np


from segment import Segment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

required_vars = [
    "AISEARCH_KEY",
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

AISEARCH_KEY = os.getenv("AISEARCH_KEY")
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
                    json.dumps(input_points),  # Serialize input_points to JSON string
                    json.dumps(input_boxes),   # Serialize input_boxes to JSON string
                    json.dumps(input_labels),  # Serialize input_labels to JSON string
                    False                      # multimask_output
                ],
            ],
        }
    }
    with open(_REQUEST_FILE_NAME, "wt") as f:
        json.dump(request_json, f)

def load_masks(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    input_points = data["input_points"]
    input_boxes = data["input_boxes"]
    input_labels = data["input_labels"]

    # if not (len(input_points) == len(input_boxes) == len(input_labels)):
    #     if len(input_points) == 1 and len(input_boxes) > 1:
    #         input_points = input_points * len(input_boxes)
    #     elif len(input_boxes) == 1 and len(input_points) > 1:
    #         input_boxes = input_boxes * len(input_points)
    #     else:
    #         logger.error(
    #             f"Mismatch in masks.json: {len(input_points)} input_points vs {len(input_boxes)} input_boxes."
    #         )
    #         raise ValueError(
    #             f"You should provide as many bounding boxes as input points per box. "
    #             f"Got {len(input_boxes)} boxes and {len(input_points)} points."
    #         )

    return input_points, input_boxes, input_labels

ADD_DATA_REQUEST_URL = "https://{search_service_name}.search.windows.net/indexes/{index_name}/docs/index?api-version={api_version}".format(
    search_service_name=SEARCH_SERVICE_NAME,
    index_name=SEARCH_INDEX_NAME,
    api_version=API_VERSION,
)

dataset_parent_dir = "./data"
dataset_name = "samples"
dataset_dir = os.path.join(dataset_parent_dir, dataset_name)

image_paths = [
    os.path.join(dp, f)
    for dp, dn, filenames in os.walk(dataset_dir)
    for f in filenames
    if os.path.splitext(f)[1] == ".png"
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
    MAX_RETRIES = 5

    # read image and get masks
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    input_points, input_boxes, input_labels = segmenter.process_image(image)

    # create embedding request
    embedding_request = make_request_images(image_path, input_points, input_boxes, input_labels)

    # get embeddings from endpoint
    response = None
    request_failed = False
    IMAGE_EMBEDDING = []
    for r in range(MAX_RETRIES):
        try:
            response = workspace_ml_client.online_endpoints.invoke(
                endpoint_name=endpoint_name,
                deployment_name=deployment_name,
                request_file=_REQUEST_FILE_NAME,
            )
            response = json.loads(response)
            # Parse the response to extract IMAGE_EMBEDDING and other key information
            response_dict = response[0]
            response_content = response_dict['response']
            predictions = response_content['predictions']
            IMAGE_EMBEDDING = []
            total_masks = 0
            for prediction in predictions:
                total_masks += len(prediction['masks_per_prediction'])

            size_per_mask = int(np.ceil(512 / total_masks))
            resize_dim = int(np.ceil(np.sqrt(size_per_mask)))
            # Ensure resize_dim is such that each mask embedding length is size_per_mask
            mask_embedding_length = resize_dim * resize_dim
            IMAGE_EMBEDDING = []
            for prediction in predictions:
                masks_per_prediction = prediction['masks_per_prediction']
                for mask_info in masks_per_prediction:
                    encoded_binary_mask = mask_info['encoded_binary_mask']
                    iou_score = mask_info['iou_score']
                    # Decode the base64-encoded mask
                    mask_bytes = base64.b64decode(encoded_binary_mask)
                    nparr = np.frombuffer(mask_bytes, np.uint8)
                    mask_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    # Resize mask to match size_per_mask
                    mask_image_resized = cv2.resize(mask_image, (resize_dim, resize_dim))
                    # Flatten and convert to a list of floats
                    mask_flattened = mask_image_resized.flatten().astype(float).tolist()
                    # Pad or truncate to size_per_mask
                    if len(mask_flattened) < size_per_mask:
                        mask_flattened += [0.0] * (size_per_mask - len(mask_flattened))
                    else:
                        mask_flattened = mask_flattened[:size_per_mask]
                    IMAGE_EMBEDDING.extend(mask_flattened)
            # If IMAGE_EMBEDDING is less than 512, pad with zeros
            if len(IMAGE_EMBEDDING) < 512:
                IMAGE_EMBEDDING += [0.0] * (512 - len(IMAGE_EMBEDDING))
            elif len(IMAGE_EMBEDDING) > 512:
                IMAGE_EMBEDDING = IMAGE_EMBEDDING[:512]  # Ensure length is 512
            print(f"Successfully retrieved embeddings for image {FILENAME}.")
            break
        except Exception as e:
            print(f"Unable to get embeddings for image {FILENAME}: {e}")
            #print(response)
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
            headers={"api-key": AISEARCH_KEY},
        )
        if response.status_code == 200:
            print(f"Successfully added data to index for image {FILENAME}.")
        else:
            print(f"Failed to add data to index for image {FILENAME}. Response: {response.content}")