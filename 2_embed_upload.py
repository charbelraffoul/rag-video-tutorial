#!/usr/bin/env python3
"""
2_embed_upload.py

Chunks a video transcript, embeds each chunk, and uploads to Weaviate Cloud.
"""
import os
import dotenv
import nltk
import tiktoken
from nltk.tokenize import sent_tokenize
from langchain_community.embeddings import HuggingFaceEmbeddings
from weaviate.connect.helpers import connect_to_weaviate_cloud
from weaviate.auth import AuthApiKey

# Load environment variables
dotenv.load_dotenv()
WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
TRANSCRIPT_PATH  = os.getenv("TRANSCRIPT_PATH", "transcript.txt")
CLASS_NAME       = os.getenv("WEAVIATE_CLASS", "TranscriptChunk")

# Ensure NLTK tokenizer is present
nltk.download('punkt', quiet=True)

def chunk_text(text, max_tokens=200, overlap=50, model_name="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model_name)
    sentences = sent_tokenize(text)
    chunks, cur_chunk, cur_tokens = [], [], 0
    for sent in sentences:
        tok = len(enc.encode(sent))
        if cur_tokens + tok > max_tokens:
            chunks.append(" ".join(cur_chunk))
            # build overlap
            carry, overlap_sents = 0, []
            for s in reversed(cur_chunk):
                st = len(enc.encode(s))
                if carry + st > overlap:
                    break
                overlap_sents.insert(0, s)
                carry += st
            cur_chunk, cur_tokens = overlap_sents, carry
        cur_chunk.append(sent)
        cur_tokens += tok
    if cur_chunk:
        chunks.append(" ".join(cur_chunk))
    return chunks

# Read and chunk transcript
if not os.path.isfile(TRANSCRIPT_PATH):
    raise FileNotFoundError(f"Transcript not found: {TRANSCRIPT_PATH}")
with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
    transcript = f.read()
chunks = chunk_text(transcript)
print(f"ℹ️ Generated {len(chunks)} chunks from transcript.")

# Connect to Weaviate Cloud
client = connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=AuthApiKey(WEAVIATE_API_KEY)
)
print("✅ Weaviate ready:", client.is_ready())

# Create or get class
existing = client.collections.list_all()
if CLASS_NAME not in existing:
    client.collections.create(
        name=CLASS_NAME,
        properties=[
            {"name": "chunk_id", "dataType": ["text"]},
            {"name": "text",     "dataType": ["text"]}
        ],
    )
    print(f"✅ Created class '{CLASS_NAME}'")
else:
    print(f"✅ Class '{CLASS_NAME}' already exists")

# Embed and upload chunks
embedder  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
collection = client.collections.get(CLASS_NAME)
for idx, chunk in enumerate(chunks, start=1):
    vector = embedder.embed_query(chunk)
    collection.data.insert(
        properties={"chunk_id": str(idx), "text": chunk},
        vector=vector
    )
    print(f"Uploaded chunk {idx}/{len(chunks)}")
print("✅ All chunks uploaded to Weaviate!")
