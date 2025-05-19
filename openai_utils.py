import openai
import base64
import requests
from PIL import Image
from io import BytesIO
import os

# OpenAI API anahtarını environment üzerinden al
openai.api_key = os.getenv("OPENAI_API_KEY")

def encode_image(image_path):
    """Görseli base64 formatında şifrele"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def tag_image_with_gpt4vision(image_path):
    """GPT-4 Vision ile görselden etiket üret"""
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Bu görselin içeriğine göre 3-5 arası etiket üret. Kısa ve genel terimler kullan."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"OpenAI API hatası: {response.status_code} - {response.text}")

    result = response.json()
    message = result['choices'][0]['message']['content']
    tags = [t.strip() for t in message.split(',')]
    return tags

# Örnek kullanım
# tags = tag_image_with_gpt4vision("ornek.png")
# print("Etiketler:", tags)
