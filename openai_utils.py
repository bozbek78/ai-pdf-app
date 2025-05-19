import base64
import json
import requests
import openai
import os

from PIL import Image
from io import BytesIO

# Ortam değişkenlerinden API anahtarlarını al
openai.api_key = os.getenv("OPENAI_API_KEY")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_COLLECTION = os.getenv("ASTRA_DB_COLLECTION")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")


def get_image_tags(image_base64: str, prompt_text: str = "") -> list[str]:
    """
    OpenAI GPT-4 Vision ile görselden etiket çıkarır.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": "Sen bir görsel etiketleme asistanısın. Görseldeki objeleri ve içerikleri etiket listesi halinde ver."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text or "Bu görseli etiketle:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}    
                    ]
                }
            ],
            max_tokens=300
        )
        tags_raw = response.choices[0].message.content.strip()
        tags = [tag.strip("- ").strip() for tag in tags_raw.split("\n") if tag]
        return tags

    except Exception as e:
        print(f"Hata (get_image_tags): {e}")
        return []


def store_image_with_tags_to_astra(image_base64: str, tags: list[str], doc_id: str):
    """
    Görseli ve etiketleri Astra DB'ye kaydeder.
    """
    url = f"{ASTRA_DB_API_ENDPOINT}/api/rest/v2/namespaces/{ASTRA_DB_KEYSPACE}/collections/{ASTRA_DB_COLLECTION}"
    headers = {
        "X-Cassandra-Token": ASTRA_DB_APPLICATION_TOKEN,
        "Content-Type": "application/json"
    }

    data = {
        "_id": doc_id,
        "type": "image",
        "base64": image_base64,
        "tags": tags
    }

    response = requests.put(f"{url}/{doc_id}", headers=headers, data=json.dumps(data))
    if response.status_code not in (200, 201):
        print(f"AstraDB kayıt hatası: {response.status_code} - {response.text}")
        raise Exception("AstraDB görsel kaydetme hatası")
    return True


def encode_image_to_base64(image_path: str) -> str:
    """
    Dosya yolundan görseli base64 string'e çevirir.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def decode_base64_to_image(base64_str: str) -> Image.Image:
    """
    Base64 string'den PIL Image nesnesi üretir.
    """
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))


def query_openai_with_astra_context(question: str, context_texts: list[str]) -> str:
    """
    OpenAI ChatCompletion API kullanarak, AstraDB'den alınan bağlamla soruyu yanıtlar.
    """
    try:
        context = "\n\n".join(context_texts)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Sen bir belge analizi uzmanısın. Aşağıdaki bağlam metnine göre soruları yanıtla."},
                {"role": "user", "content": f"Bağlam:\n{context}\n\nSoru:\n{question}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Hata (query_openai_with_astra_context): {e}")
        return "Üzgünüm, yanıt üretilemedi."
