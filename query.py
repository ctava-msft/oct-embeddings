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

def run_query(query):
    logger.info(f"Running query: {query}")

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


    TEXT_QUERY = query
    K = 4  # number of results to retrieve
    _REQUEST_FILE_NAME = "request.json"

    def make_request_text(text_sample):
        request_json = {
            "input_data": {
                "columns": ["image", "text"],
                "data": [["", text_sample]],
            }
        }

        with open(_REQUEST_FILE_NAME, "wt") as f:
            json.dump(request_json, f)


    make_request_text(TEXT_QUERY)
    response = workspace_ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        request_file=_REQUEST_FILE_NAME,
    )
    response = json.loads(response)
    QUERY_TEXT_EMBEDDING = response[0]["text_features"]

    # Enhanced logging for embedding
    logger.debug(f"Query Text Embedding: {QUERY_TEXT_EMBEDDING}")

    QUERY_REQUEST_URL = "https://{search_service_name}.search.windows.net/indexes/{index_name}/docs/search?api-version={api_version}".format(
        search_service_name=SEARCH_SERVICE_NAME,
        index_name=SEARCH_INDEX_NAME,
        api_version=API_VERSION,
    )

    # Adjust search parameters
    search_request = {
        "vectors": [{"value": QUERY_TEXT_EMBEDDING, "fields": "imageEmbeddings", "k": K}],
        #"filter": f"imagetype eq '{query}'",
        "select": "filename, imagetype, imageEmbeddings",
        "top": 50
    }

    logger.info(f"Search Request: {search_request}")

    response = requests.post(
        QUERY_REQUEST_URL, json=search_request, headers={"api-key": SEARCH_KEY}
    )
    #logger.debug(f"Search Response: {response.text}")
    if response.status_code != 200:
        logger.error(f"Search request failed with status code {response.status_code}: {response.text}")
        raise ValueError(f"Search request failed with status code {response.status_code}")
    try:
        response_json = json.loads(response.text)
        neighbors = response_json["value"]
    except KeyError:
        logger.error(f"'value' key not found in the response: {response_json}")
        raise

    K1, K2 = 3, 4

    def make_pil_image(image_path):
        pil_image = Image.open(image_path)
        return pil_image

    _, axes = plt.subplots(nrows=K1 + 1, ncols=K2, figsize=(64, 64))
    for i in range(K1 + 1):
        for j in range(K2):
            axes[i, j].axis("off")

    i, j = 0, 0

    for neighbor in neighbors:
        pil_image = make_pil_image(neighbor["filename"])
        axes[i, j].imshow(np.asarray(pil_image), aspect="auto")
        axes[i, j].text(1, 1, "{:.4f}".format(neighbor["@search.score"]), fontsize=32)

        j += 1
        if j == K2:
            i += 1
            j = 0

    plt.show()

def main():
    logger.info("Starting query script execution.")
    print("Ready to accept your question.")
    query = input("What type of images: ")
    run_query(query)

if __name__ == "__main__":
    main()