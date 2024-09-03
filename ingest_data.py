# ingest_data.py

import os
import csv
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from opensearchpy import OpenSearch
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Connect to OpenSearch
client = OpenSearch(
    hosts=[{'host': os.getenv('OPENSEARCH_HOST'), 'port': os.getenv('OPENSEARCH_PORT')}],
    http_auth=(os.getenv('OPENSEARCH_USERNAME'), os.getenv('OPENSEARCH_PASSWORD'))
)

# Define the index
index_name = "image_index2"

# Create the index with appropriate mapping
index_body = {
    "mappings": {
        "properties": {
            "productId": {"type": "keyword"},
            "gender": {"type": "keyword"},
            "category": {"type": "keyword"},
            "subCategory": {"type": "keyword"},
            "productType": {"type": "keyword"},
            "colour": {"type": "keyword"},
            "usage": {"type": "keyword"},
            "productTitle": {"type": "text"},
            "imagePath": {"type": "keyword"},
            "imageURL": {"type": "keyword"},
            "actualPrice": {"type": "integer"},
            "discountPrice": {"type": "integer"},
            "rating": {"type": "float"},
            "reviews": {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 512  # Ensure this matches the dimension of your CLIP embeddings
            }
        }
    },
    "settings": {
        "index": {
            "knn": True
        }
    }
}

# Create the index if it doesn't exist
if client.indices.exists(index=index_name):
    client.indices.delete(index=index_name)
    print("deleting old index with the same name")
client.indices.create(index=index_name, body=index_body)

def process_image(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    return outputs.detach().numpy().flatten()

def is_valid_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return True, image
    except Exception as e:
        print(f"Error fetching image from {url}: {e}")
        return False, None

def index_data(csv_file):
    success_count = 0
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            valid, image = is_valid_image(row['ImageURL'])
            if not valid:
                continue
            image_embedding = process_image(image)
            document = {
                "productId": row["ProductId"],
                "gender": row["Gender"],
                "category": row["Category"],
                "subCategory": row["SubCategory"],
                "productType": row["ProductType"],
                "colour": row["Colour"],
                "usage": row["Usage"],
                "productTitle": row["ProductTitle"],
                "imagePath": "images/" + row["Image"],
                "imageURL": row["ImageURL"],
                "actualPrice": random.randint(1200, 5000),
                "discountPrice": random.randint(500, 1199),  # Ensure discountPrice is less than actualPrice
                "rating": round(random.uniform(1.0, 5.0), 1),  # Ensure rating is a float
                "reviews": random.randint(10, 999),
                "embedding": image_embedding.tolist()
            }
            client.index(index=index_name, body=document)
            success_count += 1
            print(f"Successfully indexed {success_count} documents.")

if __name__ == "__main__":
    csv_file = "fashion.csv"  # Update with the path to your CSV file
    index_data(csv_file)
