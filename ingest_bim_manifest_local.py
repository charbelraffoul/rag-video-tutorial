# ingest_bim_manifest_local.py
import json, os
from PIL import Image
from sentence_transformers import SentenceTransformer
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, Property, DataType
from config import WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_CLASS

# Local models (CPU is fine)
TXT_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"  # 768-d
IMG_MODEL_ID = "clip-ViT-B-32"                            # 512-d

txt_model = SentenceTransformer(TXT_MODEL_ID)
img_model = SentenceTransformer(IMG_MODEL_ID)

def encode_text(s: str):
    if not s: return None
    return txt_model.encode(s, normalize_embeddings=True).tolist()

def encode_image(path: str):
    if not os.path.exists(path): return None
    img = Image.open(path).convert("RGB")
    return img_model.encode(img, normalize_embeddings=True).tolist()

def main():
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=wvc.init.Auth.api_key(WEAVIATE_API_KEY),
    )
    try:
        # If you already created WEAVIATE_CLASS with a different config,
        # delete it in the dashboard first or change the class name here.
        if not client.collections.exists(WEAVIATE_CLASS):
            client.collections.create(
                name=WEAVIATE_CLASS,
                vector_config=[
                    Configure.Vectors.self_provided(
                        name="text",
                        vector_index_config=Configure.VectorIndex.hnsw(metric="cosine"),
                    ),
                    Configure.Vectors.self_provided(
                        name="image",
                        vector_index_config=Configure.VectorIndex.hnsw(metric="cosine"),
                    ),
                ],
                properties=[
                    Property(name="sceneId",    data_type=DataType.INT),
                    Property(name="sceneStart", data_type=DataType.NUMBER),
                    Property(name="sceneEnd",   data_type=DataType.NUMBER),
                    Property(name="duration",   data_type=DataType.NUMBER),
                    Property(name="framePath",  data_type=DataType.TEXT),
                    Property(name="text",       data_type=DataType.TEXT),
                ],
            )

        col = client.collections.get(WEAVIATE_CLASS)

        # Ingest
        with col.batch.dynamic() as batch, open("video_manifest.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = (row.get("text") or "").strip()
                if not text:
                    # you can still store empty scenes if you want
                    continue

                tvec = encode_text(text)
                ivec = encode_image(row["frame_path"])

                batch.add_object(
                    properties={
                        "sceneId":    row["scene_id"],
                        "sceneStart": row["scene_start"],
                        "sceneEnd":   row["scene_end"],
                        "duration":   row["duration"],
                        "framePath":  row["frame_path"],
                        "text":       text,
                    },
                    vectors={
                        "text": tvec,
                        "image": ivec,
                    },
                )

        print("✅ Finished ingestion (local embeddings, named vectors).")
    finally:
        client.close()

if __name__ == "__main__":
    main()
