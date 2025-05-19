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

    print("📦 Gönderilecek veri JSON:", json.dumps(payload)[:500])  # İlk 500 karakteri göster

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print("✅ Astra vektör başarıyla yüklendi:", vector_id)
    except requests.exceptions.RequestException as e:
        print("❌ upsert_vector hata mesajı:", e)
        if hasattr(e, 'response') and e.response is not None:
            print("🌐 Sunucu cevabı:", e.response.text)


def process_pdf_to_astra(pdf_file_path, collection_name="standart_3"):
    print(f"📂 PDF dosyası işleniyor: {pdf_file_path}")
    pages_text = extract_text_from_pdf(pdf_file_path)

    for page_num, page_text in enumerate(pages_text):
        if not page_text.strip():
            continue

        print(f"📤 OpenAI'a embedding gönderiliyor... (sayfa {page_num})")
        embedding = get_embedding_from_openai(page_text)

        if embedding:
            upsert_vector(page_text, embedding, page_num, collection_name)
        else:
            print(f"⚠️ Sayfa {page_num} için embedding alınamadı.")
