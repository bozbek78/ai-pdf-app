import os
import numpy as np
import openai
import requests

openai.api_key = os.getenv("OPENAI_API_KEY")

# --- ASTRA yapılandırması -------------------------------------------------
ASTRA_DB_API_ENDPOINT      = os.getenv("ASTRA_DB_API_ENDPOINT")               # https://xxxx.apps.astra.datastax.com
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")          # AstraCS:...
ASTRA_DB_COLLECTION        = os.getenv("ASTRA_DB_COLLECTION",  "pdf_data")    # <── 3. madde: ENV’den okunuyor
ASTRA_DB_NAMESPACE         = os.getenv("ASTRA_DB_KEYSPACE",   "default_keyspace")  # <── 3. madde: ENV’den
# --------------------------------------------------------------------------

HEADERS = {
    "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
    "Content-Type": "application/json"
}

def get_query_embedding(text: str):
    """Metni OpenAI ile embed'e çevir; hata olursa rastgele vektör döner."""
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        print("Embedding hatası:", e)
        return np.random.rand(1536).tolist()

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def fetch_all_documents():
    url = f"{ASTRA_DB_API_ENDPOINT}/api/json/v1/{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json().get("data", [])
    return []

def query_openai_with_astra_context(query_text: str):
    """Astra'daki en yakın 3 dokümanı bul ve OpenAI ile yanıt oluştur."""
    query_vector = get_query_embedding(query_text)
    documents = fetch_all_documents()

    scored_docs = []
    for doc in documents:
        vec = doc.get("$vector")
        if vec:
            score = cosine_similarity(query_vector, vec)
            scored_docs.append((score, doc))
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    top_docs = [doc for _, doc in scored_docs[:3]]

    if not top_docs:
        return "❌ Eşleşen içerik bulunamadı.", ""

    context = ""
    for d in top_docs:
        page = d.get("page", "?")
        content = d.get("content", "")[:400]
        context += f"- Sayfa {page}: {content}\n"

    prompt = (
        f"Kullanıcı şu soruyu sordu: '{query_text}'\n"
        f"Aşağıdaki içeriklere göre yanıtla:\n{context}\n\nCevap:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response["choices"][0]["message"]["content"], context
    except Exception as e:
        return f"❌ GPT yanıt hatası: {str(e)}", context
