
import os
import openai
import requests
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_COLLECTION = "test"
ASTRA_DB_NAMESPACE = "default_keyspace"

HEADERS = {
    "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
    "Content-Type": "application/json"
}

def get_query_embedding(text, dim=1024):
    # Simülasyon; yerine OpenAI API embed kullanılabilir
    np.random.seed(abs(hash(text)) % (10**8))
    return np.random.rand(dim).tolist()

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def fetch_all_documents():
    url = f"{ASTRA_DB_API_ENDPOINT}/api/json/v1/{ASTRA_DB_NAMESPACE}/{ASTRA_DB_COLLECTION}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json().get("data", [])
    return []

def query_openai_with_astra_context(query_text):
    query_vector = get_query_embedding(query_text)
    documents = fetch_all_documents()

    # Skorla ve sırala
    scored_docs = []
    for doc in documents:
        vec = doc.get("$vector")
        if vec:
            score = cosine_similarity(query_vector, vec)
            scored_docs.append((score, doc))
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    top_docs = [doc for _, doc in scored_docs[:3]]

    if not top_docs:
        return "❌ Eşleşen içerik bulunamadı."

    # İçeriği GPT'ye hazırla
    context = ""
    for d in top_docs:
        context += f"- Sayfa {d.get('page', '?')} [{d.get('type', '')}]: {d.get('content', '')[:400]}\n"

    prompt = f"Kullanıcı şu soruyu sordu: '{query_text}'\nAşağıdaki içeriklere göre yanıtla:\n{context}\n\nCevap:"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ GPT yanıt hatası: {str(e)}"
