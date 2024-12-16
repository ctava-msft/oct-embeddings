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
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

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


# try:
#     credential = DefaultAzureCredential()
#     credential.get_token("https://management.azure.com/.default")
# except Exception as ex:
#     credential = InteractiveBrowserCredential()


# workspace_ml_client = MLClient(
#     credential=credential,
#     subscription_id=subscription_id,
#     resource_group_name=resource_group,
#     workspace_name=workspace_name,
# )


# TEXT_QUERY = "provide an oct normal image"
# K = 5  # number of results to retrieve
# _REQUEST_FILE_NAME = "request.json"

# def make_request_text(text_sample):
#     request_json = {
#         "input_data": {
#             "columns": ["image", "text"],
#             "data": [["", text_sample]],
#         }
#     }

#     with open(_REQUEST_FILE_NAME, "wt") as f:
#         json.dump(request_json, f)


# make_request_text(TEXT_QUERY)
# response = workspace_ml_client.online_endpoints.invoke(
#     endpoint_name=endpoint_name,
#     deployment_name=deployment_name,
#     request_file=_REQUEST_FILE_NAME,
# )
# response = json.loads(response)
# QUERY_TEXT_EMBEDDING = response[0]["text_features"]

# QUERY_REQUEST_URL = "https://{search_service_name}.search.windows.net/indexes/{index_name}/docs/search?api-version={api_version}".format(
#     search_service_name=SEARCH_SERVICE_NAME,
#     index_name=SEARCH_INDEX_NAME,
#     api_version=API_VERSION,
# )


# search_request = {
#     "vectors": [{"value": QUERY_TEXT_EMBEDDING, "fields": "imageEmbeddings", "k": K}],
#     "select": "filename",
# }


# response = requests.post(
#     QUERY_REQUEST_URL, json=search_request, headers={"api-key": AISEARCH_KEY}
# )
# neighbors = json.loads(response.text)["value"]


# K1, K2 = 3, 4


# def make_pil_image(image_path):
#     pil_image = Image.open(image_path)
#     return pil_image


# _, axes = plt.subplots(nrows=K1 + 1, ncols=K2, figsize=(64, 64))
# for i in range(K1 + 1):
#     for j in range(K2):
#         axes[i, j].axis("off")

# i, j = 0, 0

# for neighbor in neighbors:
#     pil_image = make_pil_image(neighbor["filename"])
#     axes[i, j].imshow(np.asarray(pil_image), aspect="auto")
#     axes[i, j].text(1, 1, "{:.4f}".format(neighbor["@search.score"]), fontsize=32)

#     j += 1
#     if j == K2:
#         i += 1
#         j = 0

# plt.show()


# visualize sample input bounding box as prompt and output mask
import io
import base64
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_mask(mask, ax, random_color=False):
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
        mask = mask > 0
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


df_from_json = pd.read_json(response)
encoded_mask = df_from_json["response"][3]["predictions"][0]["masks_per_prediction"][0][
    "encoded_binary_mask"
]
mask_iou = df_from_json["response"][3]["predictions"][0]["masks_per_prediction"][0][
    "iou_score"
]

# Load the sample image
img = Image.open(io.BytesIO(base64.b64decode(encoded_mask)))
raw_image = Image.open(sample_image).convert("RGB")

# Display the original image and bounding box
fig, axes = plt.subplots(1, 2, figsize=(15, 15))
axes[0].imshow(np.array(raw_image))
show_box([125, 240, 375, 425], axes[0])
axes[0].title.set_text(f"Input image with bounding box as prompt.")
axes[0].axis("off")

axes[1].imshow(np.array(raw_image))
show_mask(img, axes[1])
axes[1].title.set_text(f"Output mask with iou score: {mask_iou:.3f}")
axes[1].axis("off")
# Adjust the spacing between subplots
fig.tight_layout()
plt.show()