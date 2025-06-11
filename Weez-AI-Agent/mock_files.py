import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey
from datetime import datetime
import os

# Replace with your Cosmos DB credentials
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT",)
COSMOS_KEY = os.getenv("COSMOS_KEY")
DATABASE_NAME = "weezyai"
CONTAINER_NAME = "files"

# Initialize the Cosmos client
client = cosmos_client.CosmosClient(COSMOS_ENDPOINT, {'masterKey': COSMOS_KEY})

# Create/get the database
db = client.create_database_if_not_exists(id=DATABASE_NAME)

# Create/get the container (assuming you want to partition by user_id)
container = db.create_container_if_not_exists(
    id=CONTAINER_NAME,
    partition_key=PartitionKey(path="/user_id"),
    offer_throughput=400
)

# Sample document
document = {
    "id": "sample-doc-1",
    "user_id": "sayyadshakil@gmail.com",
    "document_title": "Introduction and History of AI",
    "fileName": "history_ai_notes.pdf",
    "filePath": "C:/Users/Shakil/Documents/AI/history_ai_notes.pdf",
    "textSummary": "This document explains the origins, evolution, and milestones in the field of artificial intelligence.",
    "platform": "local",
    "embedding": [0.12, 0.34, 0.56, 0.78, 0.91],
    "uploaded_at": datetime.utcnow().isoformat() + "Z"
}

# Upload the document
try:
    container.create_item(body=document)
    print("✅ Document uploaded successfully!")
except exceptions.CosmosResourceExistsError:
    print("⚠️ Document with this ID already exists.")
