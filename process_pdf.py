import os
import fitz  # PyMuPDF
import uuid
import requests
import json
from openai_utils import get_embedding_from_openai, generate_image_labels
from google_drive_utils import upload_to_google_drive

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_text = [page.get_text() for page in doc]
    doc.close()
    return pages_text


def extract_images_from_pdf(pdf_path, output_dir="pdf_images"):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    saved_images = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{uuid.uuid4().hex}_p{page_index}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            saved_images.append((image_path, page_index))
    doc.close()
    return saved_images


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

    print("📦 Gönderilecek veri JSON:", json.dumps(payload)[:500])
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print("✅ Astra vektör başarıyla yüklendi:", vector_id)
    except requests.exceptions.RequestException as e:
        print("❌ upsert_vector hata mesajı:", e)
        if hasattr(e, 'response') and e.response is not None:
            print("🌐 Sunucu cevabı:", e.response.text)


def update_image_labels_in_astra(image_id, labels, collection_name="images"):
    url = f"{ASTRA_DB_API_ENDPOINT}/api/vectordb/v1/{ASTRA_DB_KEYSPACE}/{collection_name}/vectors"
    headers = {
        "Content-Type": "application/json",
        "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
    }

    payload = {
        "vectors": [
            {
                "id": image_id,
                "metadata": {
                    "labels": labels
                }
            }
        ]
    }

    try:
        response = requests.put(url, headers=headers, json=payload)
        response.raise_for_status()
        print("✅ Etiketler güncellendi:", image_id)
    except requests.exceptions.RequestException as e:
        print("❌ Etiket güncelleme hatası:", e)
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

    # Görselleri çıkar
    images = extract_images_from_pdf(pdf_file_path)
    for image_path, page_num in images:
        uploaded_url = upload_to_google_drive(image_path)
        labels = generate_image_labels(image_path)
        image_id = os.path.basename(image_path).split('.')[0]
        update_image_labels_in_astra(image_id, labels)
