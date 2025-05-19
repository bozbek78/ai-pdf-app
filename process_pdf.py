import os
import fitz  # PyMuPDF
import uuid
import requests
import json
from openai_utils import get_embedding_from_openai


ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_text = [page.get_text() for page in doc]
    doc.close()
    return pages_text


def upsert_vector(text, embedding, page_num, collection_name):
    print("📄 Sayfa numarası:", page_num)
    print("🧠 Embedding uzunluğu:", len(embedding))
    print("📎 İlk 5 değer:", embedding[:5])

    headers = {
        "Content-Type": "application/json",
        "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
    }

    url = f"{ASTRA_DB_API_ENDPOINT}/api/vectordb/v1/{ASTRA_DB_KEYSPACE}/{collection_name}/vectors"

    vector_id = f"{uuid.uuid4().hex}_p{page_num}_text"

    payload = {
        "vectors": [
            {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": text,
                    "page": page_num
                }
            }
        ]
    }

    print("📦
