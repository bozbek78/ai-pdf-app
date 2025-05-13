import os
import io
import fitz
import hashlib
import numpy as np
from PIL import Image
import requests
from google_drive_utils import upload_image_to_drive
from openai_utils import get_query_embedding   # vektör üretmek için

# --- ASTRA yapılandırması -------------------------------------------------
ASTRA_DB_API_ENDPOINT      = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_COLLECTION        = os.getenv("ASTRA_DB_COLLECTION",  "pdf_data")   # ENV > varsayılan
ASTRA_DB_NAMESPACE         = os.getenv("ASTRA_DB_KEYSPACE",   "default_keyspace")

IMAGE_OUTPUT_DIR = "pdf_images"

HEADERS = {
    "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
    "Content-Type": "application/json"
}
# -------------------------------------------------------------------------

def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# ---------- Vector DB yardımcıları ---------------------------------------
def document_exists(doc_id: str) -> bool:
    """Belge ID’si koleksiyonda var mı?"""
    url = (f"{ASTRA_DB_API_ENDPOINT}/api/vectordb/v1/"
           f"{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}/query")
    payload = {"ids": [doc_id], "topK": 1}
    r = requests.post(url, headers=HEADERS, json=payload)
    return r.status_code == 200 and r.json().get("count", 0) > 0

def insert_document(doc_id: str, payload: dict, vector: list):
    """Vector + metadata’yı koleksiyona ekle."""
    url = (f"{ASTRA_DB_API_ENDPOINT}/api/vectordb/v1/"
           f"{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}/vectors")
    vec_doc = {
        "_id":    doc_id,
        "$vector": vector,
        **payload
    }
    r = requests.post(url, headers=HEADERS, json={"vectors": [vec_doc]})
    return r.status_code, r.text[:120]

# -------------------------------------------------------------------------
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
                vector = get_query_embedding(text[:8192])  # OpenAI 3‑small, 1536 dim.
                meta   = {
                    "page": i + 1,
                    "type": "text",
                    "file": base_name,
                    "content": text[:1000]
                }
                insert_document(doc_id, meta, vector)

            # ---------- görüntüler ----------
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext   = base_image["ext"]

                image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                filename  = f"{base_name}_p{i+1}_img{img_index+1}.{image_ext}"
                filepath  = os.path.join(IMAGE_OUTPUT_DIR, filename)
                image_pil.save(filepath)
                upload_image_to_drive(filepath, base_name)

                doc_id = f"{file_hash}_p{i+1}_img{img_index+1}"
                meta   = {
                    "page": i + 1,
                    "type": "image",
                    "file": filename,
                    "content": "image extracted"
                }
                vector = np.random.rand(1536).tolist()  # görüntü için placeholder
                insert_document(doc_id, meta, vector)

        # dosyanın kendisi için “root” belge (vektörsüz) – opsiyonel
        insert_document(file_hash, {"type": "file", "name": base_name}, np.zeros(1536).tolist())
        results.append(f"✅ {base_name} işlendi ve yüklendi.")

    return "\n".join(results)

# ---------- yardımcılar ---------------------------------------------------
def list_images():
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    return [
        os.path.join(IMAGE_OUTPUT_DIR, f)
        for f in os.listdir(IMAGE_OUTPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

def update_image_label(filepath, label):
    filename = os.path.basename(filepath)
    doc_id = filename.replace(".", "_")
    meta_update = {"label": label}

    url = (f"{ASTRA_DB_API_ENDPOINT}/api/vectordb/v1/"
           f"{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}/vectors")
    r = requests.put(url, headers=HEADERS,
                     json={"vectors": [{"_id": doc_id, **meta_update}]})
    return "✅ Etiket güncellendi" if r.status_code == 200 else "❌ Etiket güncellenemedi"
