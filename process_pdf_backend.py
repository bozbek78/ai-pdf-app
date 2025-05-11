
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import numpy as np
from astrapy import DataAPIClient

# Config
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_COLLECTION = "test"
IMAGE_OUTPUT_DIR = "pdf_images"

def generate_vector(dim=1024):
    return np.random.rand(dim).tolist()

def process_pdf_to_astra(file):
    if file is None:
        return "❌ Dosya bulunamadı."

    client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
    db = client.get_database_by_api_endpoint(ASTRA_DB_API_ENDPOINT)
    collection = db[ASTRA_DB_COLLECTION]

    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    results = []
    file_path = file.name if hasattr(file, "name") else file
    file_base = os.path.splitext(os.path.basename(file_path))[0]
    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc):
        doc_id_prefix = f"{file_base}_page{page_num+1}"

        # Metin
        text = page.get_text().strip()
        if text:
            doc_id = f"{doc_id_prefix}_text"
            if collection.find_one({"_id": doc_id}) is None:
                collection.insert_one({
                    "_id": doc_id,
                    "$vector": generate_vector(),
                    "page": page_num + 1,
                    "type": "text",
                    "file": "-",
                    "content": text[:1000]
                })
                results.append(f"✅ {doc_id} - metin yüklendi.")
            else:
                results.append(f"⏭️ {doc_id} - zaten yüklü.")

        # Görsel
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
                if collection.find_one({"_id": doc_id}) is None:
                    collection.insert_one({
                        "_id": doc_id,
                        "$vector": generate_vector(),
                        "page": page_num + 1,
                        "type": "image",
                        "file": image_filename,
                        "content": "image saved"
                    })
                    results.append(f"✅ {doc_id} - görsel yüklendi.")
                else:
                    results.append(f"⏭️ {doc_id} - görsel zaten var.")
        else:
            # Görsel yoksa tüm sayfayı render et
            pix = page.get_pixmap(dpi=300)
            image_filename = f"{doc_id_prefix}_rendered.png"
            image_path = os.path.join(IMAGE_OUTPUT_DIR, image_filename)
            pix.save(image_path)

            doc_id = f"{doc_id_prefix}_rendered"
            if collection.find_one({"_id": doc_id}) is None:
                collection.insert_one({
                    "_id": doc_id,
                    "$vector": generate_vector(),
                    "page": page_num + 1,
                    "type": "rendered_page",
                    "file": image_filename,
                    "content": "page rendered"
                })
                results.append(f"✅ {doc_id} - render kaydedildi.")
            else:
                results.append(f"⏭️ {doc_id} - render zaten var.")

    return "\n".join(results)
