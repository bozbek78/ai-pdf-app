import os
import fitz
import hashlib
from PIL import Image
import io
from google_drive_utils import upload_image_to_drive
import numpy as np
import requests

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_COLLECTION = "pdf_data"
ASTRA_DB_NAMESPACE = "default_keyspace"
IMAGE_OUTPUT_DIR = "pdf_images"

HEADERS = {
    "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
    "Content-Type": "application/json"
}

def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def document_exists(doc_id):
    url = f"{ASTRA_DB_API_ENDPOINT}/api/json/v1/{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}/{doc_id}"
    response = requests.get(url, headers=HEADERS)
    return response.status_code == 200

def insert_document(doc_id, payload):
    url = f"{ASTRA_DB_API_ENDPOINT}/api/json/v1/{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}"
    payload["_id"] = doc_id
    payload["$vector"] = np.random.rand(1536).tolist()  # placeholder
    response = requests.post(url, headers=HEADERS, json=payload)
    return response.status_code, response.text

def process_pdf_to_astra(files):
    if not files:
        return "❌ Hiçbir dosya yüklenmedi."

    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    results = []

    for file in files:
        file_path = file.name if hasattr(file, "name") else file
        file_hash = sha256_file(file_path)
        if document_exists(file_hash):
            results.append(f"⏭️ {os.path.basename(file_path)} zaten işlenmiş.")
            continue

        doc = fitz.open(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                doc_id = f"{file_hash}_p{i+1}_text"
                payload = {
                    "page": i + 1,
                    "type": "text",
                    "file": base_name,
                    "content": text[:1000]
                }
                insert_document(doc_id, payload)

            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                filename = f"{base_name}_p{i+1}_img{img_index+1}.{image_ext}"
                filepath = os.path.join(IMAGE_OUTPUT_DIR, filename)
                image_pil.save(filepath)

                upload_image_to_drive(filepath, base_name)

                doc_id = f"{file_hash}_p{i+1}_img{img_index+1}"
                payload = {
                    "page": i + 1,
                    "type": "image",
                    "file": filename,
                    "content": "image extracted"
                }
                insert_document(doc_id, payload)

        insert_document(file_hash, {"type": "file", "name": base_name})
        results.append(f"✅ {base_name} işlendi ve yüklendi.")

    return "\n".join(results)

def list_images():
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    return [os.path.join(IMAGE_OUTPUT_DIR, f) for f in os.listdir(IMAGE_OUTPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def update_image_label(filepath, label):
    filename = os.path.basename(filepath)
    doc_id = filename.replace(".", "_")
    payload = {"label": label}
    url = f"{ASTRA_DB_API_ENDPOINT}/api/json/v1/{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}/{doc_id}"
    response = requests.put(url, headers=HEADERS, json=payload)
    return "✅ Etiket güncellendi" if response.status_code == 200 else "❌ Etiket güncellenemedi"
