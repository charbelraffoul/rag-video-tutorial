# ingest_bim_manifest.py
import json
import base64
import os
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import DataType, Property, Configure
import requests
from config import WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_CLASS, VECTOR_DIMENSION

# Choose an embedding model
# all-MiniLM-L6-v2 (384 dims) or all-mpnet-base-v2 (768 dims)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Function to get embeddings from Hugging Face Inference API (free tier)
HF_API_KEY = os.getenv("HF_API_KEY")  # safer than hardcoding
HF_URL = f"https://api-inference.huggingface.co/feature-extraction/{EMBEDDING_MODEL}"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

def embed_text(text: str):
    r = requests.post(HF_URL, headers=HF_HEADERS, json={"inputs": text})
    r.raise_for_status()
    return r.json()

def embed_image_base64(img_path: str):
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    # Convert to base64 if your embedding service needs it
    # If using CLIP through Hugging Face:
    clip_model = "openai/clip-vit-base-patch32"
    clip_url = f"https://api-inference.huggingface.co/feature-extraction/{clip_model}"
    r = requests.post(clip_url, headers=HF_HEADERS, data=img_bytes)
    r.raise_for_status()
    return r.json()

def main():
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=wvc.init.Auth.api_key(WEAVIATE_API_KEY)
    )

    # Create schema if not exists
    if not client.collections.exists(WEAVIATE_CLASS):
        client.collections.create(
            name=WEAVIATE_CLASS,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # manual vectors
            properties=[
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="framePath", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="sceneId", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="sceneStart", data_type=wvc.config.DataType.NUMBER),
                wvc.config.Property(name="sceneEnd", data_type=wvc.config.DataType.NUMBER),
                wvc.config.Property(name="duration", data_type=wvc.config.DataType.NUMBER),
                wvc.config.Property(name="imageVector", data_type=DataType.NUMBER_ARRAY, vector_index_type="hnsw")
            ]
        )

    col = client.collections.get(WEAVIATE_CLASS)

    with open("video_manifest.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row["text"].strip():
                continue

            # Generate vectors
            text_vec = embed_text(row["text"])
            img_vec = embed_image_base64(row["frame_path"])

            col.data.insert(
                properties={
                    "text": row["text"],
                    "framePath": row["frame_path"],
                    "sceneId": row["scene_id"],
                    "sceneStart": row["scene_start"],
                    "sceneEnd": row["scene_end"],
                    "duration": row["duration"],
                    "imageVector": img_vec
                },
                vector=text_vec
            )

    print("✅ Finished ingestion")

if __name__ == "__main__":
    main()
