import base64
import requests
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def encode_image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded


def get_image_tags(image_base64: str, prompt_text: str) -> list[str]:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        tags_raw = response.choices[0].message.content.strip()
        return tags_raw.split("\n") if tags_raw else []
    except Exception as e:
        print(f"Hata (get_image_tags): {e}")
        return []


def auto_label_image(image_path: str) -> list[str]:
    image_base64 = encode_image_to_base64(image_path)
    prompt_text = "Lütfen bu görseli analiz et ve görseldeki önemli nesne, ortam veya detayları başlıklar halinde sırala."
    return get_image_tags(image_base64, prompt_text)


def update_image_label(current_labels: list[str], new_labels: list[str]) -> list[str]:
    # Önce tüm etiketleri birleştir
    combined = current_labels + new_labels
    # Küçük harf ve boşluklardan arındırarak benzersizleştir
    cleaned = list(set(label.strip().lower() for label in combined if label.strip()))
    return sorted(cleaned)


def get_embedding_from_openai(text: str, model: str = "text-embedding-ada-002") -> list[float]:
    try:
        response = openai.Embedding.create(
            input=[text],
            model=model
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"Hata (get_embedding_from_openai): {e}")
        return []


def generate_image_labels(image_base64: str) -> list[str]:
    prompt_text = "Görselde gördüğün nesneleri madde madde yaz."
    return get_image_tags(image_base64, prompt_text)
