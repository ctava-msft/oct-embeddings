import os
from dotenv import load_dotenv
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

required_vars = [
    "AISEARCH_KEY",
    "SEARCH_SERVICE_NAME",
    "SEARCH_INDEX_NAME",
]

for var in required_vars:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
        raise ValueError(f"Missing required environment variable: {var}")


AISEARCH_KEY = os.getenv("AISEARCH_KEY")
SEARCH_SERVICE_NAME = os.getenv("SEARCH_SERVICE_NAME")
SEARCH_INDEX_NAME = os.getenv("SEARCH_INDEX_NAME")
API_VERSION = "2023-07-01-Preview"
CREATE_INDEX_REQUEST_URL = "https://{search_service_name}.search.windows.net/indexes?api-version={api_version}".format(
    search_service_name=SEARCH_SERVICE_NAME, api_version=API_VERSION
)

create_request = {
    "name": SEARCH_INDEX_NAME,
    "fields": [
        {
            "name": "id",
            "type": "Edm.String",
            "key": True,
            "searchable": True,
            "retrievable": True,
            "filterable": True,
        },
        {
            "name": "filename",
            "type": "Edm.String",
            "searchable": True,
            "filterable": True,
            "sortable": True,
            "retrievable": True,
        },
        {
            "name": "imageEmbeddings",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "retrievable": True,
            "dimensions": 512,
            "vectorSearchConfiguration": "my-vector-config",
        },
    ],
    "vectorSearch": {
        "algorithmConfigurations": [
            {
                "name": "my-vector-config",
                "kind": "hnsw",
                "hnswParameters": {
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine",
                },
            }
        ]
    },
}
response = requests.post(
    CREATE_INDEX_REQUEST_URL,
    json=create_request,
    headers={"api-key": AISEARCH_KEY},
)
response.raise_for_status()
print(response.json())