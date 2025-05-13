import os
import io
import fitz                       # PyMuPDF
import hashlib
import numpy as np
from PIL import Image
import requests

from google_drive_utils import upload_image_to_drive
from openai_utils      import get_query_embedding      # 1 024‑boyutlu embed

# ───────────────────────── Astra yapılandırması ──────────────────────────
ASTRA_DB_API_ENDPOINT      = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_COLLECTION        = os.getenv("ASTRA_DB_COLLECTION",  "pdf_data")
ASTRA_DB_NAMESPACE         = os.getenv("ASTRA_DB_KEYSPACE",   "default_keyspace")

IMAGE_OUTPUT_DIR = "pdf_images"

HEADERS = {
    "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
    "Content-Type": "application/json"
}
# ─────────────────────────────────────────────────────────────────────────


# ---------- yardımcı ------------------------------------------------------
def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------- Vector DB etkileşimi -----------------------------------------
def document_exists(doc_id: str) -> bool:
    """Belge ID’si koleksiyonda zaten var mı? (Vectordb query ile)"""
    url = (f"{ASTRA_DB_API_ENDPOINT}/api/vectordb/v1/"
           f"{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}/query")
    payload = {"ids": [doc_id], "topK": 1}
    r = requests.post(url, headers=HEADERS, json=payload, timeout=10)
    return r.ok and r.json().get("count", 0) > 0


def upsert_vector(doc_id: str, metadata: dict, vector: list[float]):
    """Vektör + meta bilgiyi koleksiyona ekle / güncelle."""
    url = (f"{ASTRA_DB_API_ENDPOINT}/api/vectordb/v1/"
           f"{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}/vectors")
    body = {"vectors": [{**metadata, "_id": doc_id, "$vector": vector}]}
    r = requests.post(url, headers=HEADERS, json=body, timeout=10)
    r.raise_for_status()        # hata varsa log’da görünür


# ---------- Ana işlev ----------------------------------------------------
def process_pdf_to_astra(files):
    """
    PDF sayfa metinlerini ve gömülü görselleri:
      • Google Drive’a (görseller)
      • Astra Vectordb’ye (vektör + meta)
    aktarır.
    """
    if not files:
        return "❌ Hiçbir dosya seçilmedi."

    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    results = []

    for f in files:
        file_path = f.name if hasattr(f, "name") else f
        file_hash = sha256_file(file_path)

        if document_exists(file_hash):
            results.append(f"⏭️ {os.path.basename(file_path)} zaten yüklü.")
            continue

        doc       = fitz.open(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        for page_no, page in enumerate(doc, start=1):

            # ---- Sayfa metni ------------------------------------------------
            text = page.get_text().strip()
            if text:
                doc_id  = f"{file_hash}_p{page_no}_text"
                vector  = get_query_embedding(text[:8_192])   # 1 024‑boyut
                meta    = {
                    "page":    page_no,
                    "type":    "text",
                    "file":    base_name,
                    "content": text[:1_000]
                }
                upsert_vector(doc_id, meta, vector)

            # ---- Sayfa görselleri -----------------------------------------
            for img_idx, img in enumerate(page.get_images(full=True), start=1):
                xref        = img[0]
                base_image  = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext   = base_image["ext"]

                image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                filename  = f"{base_name}_p{page_no}_img{img_idx}.{image_ext}"
                filepath  = os.path.join(IMAGE_OUTPUT_DIR, filename)
                image_pil.save(filepath)

                # Drive’a yükle
                upload_image_to_drive(filepath, base_name)

                # Astra’ya placeholder vektör (isteğe göre gerçek resim embed’i ekleyebilirsiniz)
                doc_id = f"{file_hash}_p{page_no}_img{img_idx}"
                meta   = {
                    "page": page_no,
                    "type": "image",
                    "file": filename,
                    "content": "image extracted"
                }
                upsert_vector(doc_id, meta, np.random.rand(1_024).tolist())

        # İsteğe bağlı kök kayıt (vektörsüz)
        upsert_vector(file_hash,
                      {"type": "file", "name": base_name},
                      np.zeros(1_024).tolist())

        results.append(f"✅ {base_name} işlendi ve yüklendi.")

    return "\n".join(results)


# ---------- Görsel listesi & etiket güncelleme ---------------------------
def list_images():
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    return [
        os.path.join(IMAGE_OUTPUT_DIR, f)
        for f in os.listdir(IMAGE_OUTPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def update_image_label(filepath, label):
    filename = os.path.basename(filepath)
    doc_id   = filename.replace(".", "_")

    url = (f"{ASTRA_DB_API_ENDPOINT}/api/vectordb/v1/"
           f"{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}/vectors")

    body = {"vectors": [{"_id": doc_id, "label": label}]}
    r = requests.put(url, headers=HEADERS, json=body, timeout=10)
    return "✅ Etiket güncellendi" if r.ok else "❌ Etiket güncellenemedi"
