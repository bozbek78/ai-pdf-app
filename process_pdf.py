import os, io, hashlib, numpy as np, requests, fitz           # PyMuPDF
from PIL import Image
from google_drive_utils import upload_image_to_drive
from openai_utils      import get_query_embedding             # 1 024‑boyutlu embed

# ── Astra ayarları ───────────────────────────────────────────
ASTRA_DB_API_ENDPOINT      = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_COLLECTION        = os.getenv("ASTRA_DB_COLLECTION",  "pdf_data")
ASTRA_DB_NAMESPACE         = os.getenv("ASTRA_DB_KEYSPACE",   "default_keyspace")

HEADERS = {
    "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
    "Content-Type": "application/json"
}

IMAGE_OUTPUT_DIR = "pdf_images"         # yerel klasör (Render’da da oluşur)
# ─────────────────────────────────────────────────────────────


def sha256_file(path: str) -> str:
    """Dosyayı hash’leyip benzersiz kimlik üretir."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def upsert_vector(doc_id: str, metadata: dict, vector: list[float]):
    """Vektör + meta bilgiyi Astra VectorDB v1’e ekler (varsa günceller)."""
    url  = f"{ASTRA_DB_API_ENDPOINT}/api/vectordb/v1/{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}/vectors"
    body = {"vectors": [{
        "id":      doc_id,      # eski _id değil
        "values":  vector,      # eski $vector değil
        "metadata": metadata
    }]}
    requests.post(url, headers=HEADERS, json=body, timeout=10).raise_for_status()


def process_pdf_to_astra(files):
    """PDF’leri işler; metin & görselleri Drive’a + Astra’ya gönderir."""
    if not files:
        return "❌ Hiçbir dosya seçilmedi."

    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    results = []

    for f in files:
        file_path = f.name if hasattr(f, "name") else f
        file_hash = sha256_file(file_path)

        doc       = fitz.open(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        for page_no, page in enumerate(doc, start=1):

            # ── Sayfa metni ────────────────────────────────────────────────
            text = page.get_text().strip()
            if text:
                upsert_vector(
                    f"{file_hash}_p{page_no}_text",
                    {
                        "page":    page_no,
                        "type":    "text",
                        "file":    base_name,
                        "content": text[:1_000]
                    },
                    get_query_embedding(text[:8_192])          # 1 024 eleman
                )

            # ── Sayfa görselleri ──────────────────────────────────────────
            for img_idx, img in enumerate(page.get_images(full=True), start=1):
                xref        = img[0]
                base_image  = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext   = base_image["ext"]

                image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                filename  = f"{base_name}_p{page_no}_img{img_idx}.{image_ext}"
                filepath  = os.path.join(IMAGE_OUTPUT_DIR, filename)
                image_pil.save(filepath)

                upload_image_to_drive(filepath, base_name)   # Google Drive

                upsert_vector(
                    f"{file_hash}_p{page_no}_img{img_idx}",
                    {
                        "page":    page_no,
                        "type":    "image",
                        "file":    filename,
                        "content": "image extracted"
                    },
                    np.random.rand(1_024).tolist()            # yer tutucu vektör
                )

        # Dosyanın “kök” kaydı (isteğe bağlı)
        upsert_vector(
            file_hash,
            {"type": "file", "name": base_name},
            np.zeros(1_024).tolist()
        )

        results.append(f"✅ {base_name} işlendi ve yüklendi.")

    return "\n".join(results)


# ── Yardımcılar: görsel listesi & etiket güncelleme ──────────────────────
def list_images():
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    return [os.path.join(IMAGE_OUTPUT_DIR, f)
            for f in os.listdir(IMAGE_OUTPUT_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))]


def update_image_label(filepath, label):
    filename = os.path.basename(filepath)
    doc_id   = filename.replace(".", "_")
    url  = f"{ASTRA_DB_API_ENDPOINT}/api/vectordb/v1/{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}/vectors"
    body = {"vectors": [{
        "id":       doc_id,
        "metadata": {"label": label}
    }]}
    r = requests.put(url, headers=HEADERS, json=body, timeout=10)
    return "✅ Etiket güncellendi" if r.ok else "❌ Etiket güncellenemedi"
