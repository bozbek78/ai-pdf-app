# openai_utils.py
import os
import json
import numpy as np
import requests
import openai

# ───────────────── OpenAI Ayarları ────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")

# ───────────────── Embedding Fonksiyonu ───────────────────────────
def get_query_embedding(text: str) -> list[float]:
    """
    OpenAI 'text-embedding-ada-002' modeli ile 1536 boyutlu embedding üretir.
    Hata durumunda 1536 boyutunda rastgele vektör döner.
    """
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        print("⚠️ OpenAI embedding hatası:", e)
        return np.random.rand(1536).tolist()

# ───────────────── Opsiyonel: Astra Vektör Arama ─────────────────
ASTRA_DB_API_ENDPOINT      = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_COLLECTION        = os.getenv("ASTRA_DB_COLLECTION", "pdf_data")
ASTRA_DB_NAMESPACE         = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")

HEADERS = {
    "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
    "Content-Type": "application/json"
}

def _astra_post(url: str, payload: dict, timeout: int = 20):
    r = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def nearest_documents(vector: list[float], top_k: int = 3) -> list[dict]:
    """
    Verilen vektöre en yakın top_k dokümanı Astra Vectordb’den getirir.
    """
    url = f"{ASTRA_DB_API_ENDPOINT}/api/vectordb/v1/{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}/query"
    try:
        data = _astra_post(url, {"vector": vector, "topK": top_k})
        return [hit["document"] for hit in data.get("results", [])]
    except Exception as e:
        print("⚠️ Vectordb query hatası:", e)
        return []

def query_openai_with_astra_context(question: str) -> tuple[str, str]:
    """
    • Soru için embedding vektörü alır.
    • Astra’dan en ilgili 3 dokümanı çeker.
    • OpenAI ile yanıtlar.
    """
    vector = get_query_embedding(question)
    docs   = nearest_documents(vector, top_k=3)

    if not docs:
        return "❌ Eşleşen içerik bulunamadı.", ""

    context_lines = [
        f"- Sayfa {d.get('page','?')}: {d.get('content','')[:400]}"
        for d in docs
    ]
    context = "\n".join(context_lines)

    prompt = (
        f"Kullanıcı şu soruyu sordu: “{question}”\n"
        f"Aşağıdaki içeriklere göre yanıtla:\n{context}\n\nCevap:"
    )

    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = res["choices"][0]["message"]["content"]
        return answer.strip(), context
    except Exception as e:
        err = f"❌ GPT yanıt hatası: {e}"
        print(err)
        return err, context
