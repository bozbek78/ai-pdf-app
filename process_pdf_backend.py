
import fitz
from PIL import Image
import io
import os
import numpy as np
import requests
import json

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_COLLECTION = "test"
ASTRA_DB_NAMESPACE = "default_keyspace"
IMAGE_OUTPUT_DIR = "pdf_images"

HEADERS = {
    "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
    "Content-Type": "application/json"
}

def generate_vector(dim=1024):
    return np.random.rand(dim).tolist()

def insert_document(doc_id, payload):
    url = f"{ASTRA_DB_API_ENDPOINT}/api/json/v1/{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}"
    payload["_id"] = doc_id
    payload["$vector"] = generate_vector()
    response = requests.post(url, headers=HEADERS, json=payload)
    return response.status_code, response.text

def document_exists(doc_id):
    url = f"{ASTRA_DB_API_ENDPOINT}/api/json/v1/{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}/{doc_id}"
    response = requests.get(url, headers=HEADERS)
    return response.status_code == 200

def process_pdf_to_astra(file):
    if file is None:
        return "❌ Dosya bulunamadı."

    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    results = []
    file_path = file.name if hasattr(file, "name") else file
    file_base = os.path.splitext(os.path.basename(file_path))[0]
    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc):
        doc_id_prefix = f"{file_base}_page{page_num+1}"

        text = page.get_text().strip()
        if text:
            doc_id = f"{doc_id_prefix}_text"
            if not document_exists(doc_id):
                payload = {
                    "page": page_num + 1,
                    "type": "text",
                    "file": "-",
                    "content": text[:1000]
                }
                status, msg = insert_document(doc_id, payload)
                results.append(f"✅ {doc_id} - metin yüklendi." if status == 201 else f"❌ {doc_id} - hata: {msg}")
            else:
                results.append(f"⏭️ {doc_id} - zaten yüklü.")

        images = page.get_images(full=True)
        if images:
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_filename = f"{doc_id_prefix}_img{img_index+1}.{image_ext}"
                image_path = os.path.join(IMAGE_OUTPUT_DIR, image_filename)
                image_pil.save(image_path)

                doc_id = f"{doc_id_prefix}_img{img_index+1}"
                if not document_exists(doc_id):
                    payload = {
                        "page": page_num + 1,
                        "type": "image",
                        "file": image_filename,
                        "content": "image saved"
                    }
                    status, msg = insert_document(doc_id, payload)
                    results.append(f"✅ {doc_id} - görsel yüklendi." if status == 201 else f"❌ {doc_id} - hata: {msg}")
                else:
                    results.append(f"⏭️ {doc_id} - görsel zaten var.")
        else:
            pix = page.get_pixmap(dpi=300)
            image_filename = f"{doc_id_prefix}_rendered.png"
            image_path = os.path.join(IMAGE_OUTPUT_DIR, image_filename)
            pix.save(image_path)

            doc_id = f"{doc_id_prefix}_rendered"
            if not document_exists(doc_id):
                payload = {
                    "page": page_num + 1,
                    "type": "rendered_page",
                    "file": image_filename,
                    "content": "page rendered"
                }
                status, msg = insert_document(doc_id, payload)
                results.append(f"✅ {doc_id} - render kaydedildi." if status == 201 else f"❌ {doc_id} - hata: {msg}")
            else:
                results.append(f"⏭️ {doc_id} - render zaten var.")

    return "\n".join(results)
