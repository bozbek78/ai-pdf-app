
import os
import fitz  # PyMuPDF
import uuid
import logging

from openai_utils import create_embedding
from google_drive_utils import upload_image_to_drive
import requests
from dotenv import load_dotenv

load_dotenv()

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_API_KEY = os.getenv("ASTRA_DB_API_KEY")

HEADERS = {
    "Content-Type": "application/json",
    "x-cassandra-token": ASTRA_DB_API_KEY,
}

def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    data = []

    for i, page in enumerate(doc):
        text = page.get_text()
        image_list = page.get_images(full=True)

        page_images = []
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            img_filename = f"{uuid.uuid4()}.png"
            with open(img_filename, "wb") as img_file:
                img_file.write(image_bytes)

            page_images.append(img_filename)

        data.append({
            "page": i + 1,
            "text": text,
            "images": page_images,
        })

    return data

def upsert_vector(payload):
    url = f"{ASTRA_DB_API_ENDPOINT}/vectors"
    try:
        response = requests.post(url, headers=HEADERS, json=payload)
        response.raise_for_status()
        print("✅ AstraDB'ye vektör başarıyla kaydedildi.")
    except requests.exceptions.HTTPError as e:
        print("❌ upsert_vector hatası:", e)
        print("📤 Gönderilen veri:", payload)

def process_pdf_to_astra(pdf_path):
    filename = os.path.basename(pdf_path)
    print(f"🔍 PDF işleniyor: {filename}")

    extracted = extract_text_and_images(pdf_path)

    for page_data in extracted:
        text = page_data["text"]
        page_num = page_data["page"]
        images = page_data["images"]

        if not text.strip():
            print(f"⚠️ Sayfa {page_num} boş, atlanıyor.")
            continue

        embedding = create_embedding(text)

        for image_path in images:
            image_id = upload_image_to_drive(image_path, filename)
            print(f"📸 Görsel yüklendi (Google Drive ID): {image_id}")

            vector_payload = {
                "vectors": [
                    {
                        "id": f"{filename}_{page_num}_{image_id}",
                        "values": embedding,
                        "metadata": {
                            "filename": filename,
                            "page": page_num,
                            "text": text,
                            "image_id": image_id,
                        },
                    }
                ]
            }
            upsert_vector(vector_payload)

    print("✅ PDF işleme tamamlandı:", filename)
