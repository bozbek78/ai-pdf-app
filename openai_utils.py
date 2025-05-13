# openai_utils.py
import os
import json
import numpy as np
import requests
import openai

# ───────────────── Astra ayarları ─────────────────────────────────────────
ASTRA_DB_API_ENDPOINT      = os.getenv("ASTRA_DB_API_ENDPOINT")            # https://xxxx.apps.astra.datastax.com
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")       # AstraCS:…
ASTRA_DB_COLLECTION        = os.getenv("ASTRA_DB_COLLECTION", "pdf_data")  # Koleksiyon
ASTRA_DB_NAMESPACE         = os.getenv("ASTRA_DB_KEYSPACE",  "default_keyspace")

HEADERS = {
    "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
    "Content-Type": "application/json"
}

# ───────────────── OpenAI ayarları ────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")

# ───────────────── Yardımcılar ────────────────────────────────────────────
def _astra_post(url: str, payload: dict, timeout: int = 20):
    """Basit POST; hata olduğunda istisna fırlatır."""
    r = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ───────────────── Embedding (NV‑Embed‑QA, 1024) ─────────────────────────
def get_query_embedding(text: str) -> list[float]:
    """
    Metni Datastax’in dahili "nv-embed-qa" modeline gönderir.
    Hata durumunda rastgele 1024‑boyutlu vektör döner ki
    akış bozulmadan devam edebilsin.
    """
    url = f"{ASTRA_DB_API_ENDPOINT}/api/embeddings/text"
    try:
        data = _astra_post(url, {"text": text, "model": "nv-embed-qa"})
        return data["embedding"]
    except Exception as e:
        print("⚠️  Embedding hatası:", e)
        return np.random.rand(1024).tolist()

# ───────────────── Vector sorgu (vectordb v1) ────────────────────────────
def nearest_documents(vector: list[float], top_k: int = 3) -> list[dict]:
    """
    Verilen vektöre en yakın top_k dokümanı Astra Vectordb’den getirir.
    Dönen öğeler doğrudan 'document' sözlükleridir.
    """
    url = (f"{ASTRA_DB_API_ENDPOINT}/api/vectordb/v1/"
           f"{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}/query")
    try:
        data = _astra_post(url, {"vector": vector, "topK": top_k})
        return [hit["document"] for hit in data.get("results", [])]
    except Exception as e:
        print("⚠️  Vectordb query hatası:", e)
        return []

# ───────────────── Soru–cevap akışı ──────────────────────────────────────
def query_openai_with_astra_context(question: str) -> tuple[str, str]:
    """
    • Soru için embed vektörü alır.  
    • Astra’dan en ilgili 3 dokümanı çeker.  
    • İçeriği prompt’a ekleyip OpenAI (GPT‑3.5‑Turbo) ile yanıtlattırır.
    Dönen tuple = (cevap, kullanılan context metni)
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
