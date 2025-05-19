import os
import base64
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_openai_with_astra_context(question: str, context: str = "") -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Aşağıdaki veriler AstraDB'den alınmış bilgileri içeriyor."},
                {"role": "user", "content": f"{context}\n\nSoru: {question}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Hata oluştu: {e}"

def auto_label_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Bu görselde neler var? 3-5 arası anahtar kelimeyle etiketle."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                }
            ],
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Görsel etiketleme hatası: {e}"

def update_image_label(image_name: str, new_label: str) -> str:
    try:
        print(f"{image_name} görseline '{new_label}' etiketi eklendi.")
        return f"Etiket '{new_label}' görsel '{image_name}' için kaydedildi."
    except Exception as e:
        print(f"Etiket güncellerken hata: {e}")
        return "Etiket güncellenemedi."

def get_embedding_from_openai(text: str, model: str = "text-embedding-3-small") -> list:
    try:
        response = client.embeddings.create(
            model=model,
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        return f"Gömme alınamadı: {e}"

def generate_image_labels(image_paths: list) -> dict:
    results = {}
    for image_path in image_paths:
        label = auto_label_image(image_path)
        results[image_path] = label
    return results
