
import os
import openai
from astrapy import DataAPIClient
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_COLLECTION = "test"

# Dummy embedding simulation (replace with OpenAI API call)
def get_query_embedding(text, dim=1024):
    # Normally use openai.Embedding.create(...), here we simulate for test
    np.random.seed(abs(hash(text)) % (10**8))
    return np.random.rand(dim).tolist()

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def query_openai_with_astra_context(query_text):
    client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
    db = client.get_database_by_api_endpoint(ASTRA_DB_API_ENDPOINT)
    collection = db[ASTRA_DB_COLLECTION]

    query_vector = get_query_embedding(query_text)
    all_docs = collection.find()
    
    # Hesapla ve sırala
    scored_docs = []
    for doc in all_docs:
        vector = doc.get("$vector")
        if vector:
            score = cosine_similarity(query_vector, vector)
            scored_docs.append((score, doc))
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    top_docs = [d for _, d in scored_docs[:3]]

    if not top_docs:
        return "❌ Uygun içerik bulunamadı."

    # Toplam içerik oluştur
    context = ""
    for d in top_docs:
        snippet = d.get("content", "")[:500]
        context += f"- Sayfa {d.get('page', '?')} [{d.get('type', '')}]: {snippet}\n"

    prompt = f"Kullanıcı şu soruyu sordu: '{query_text}'\nAşağıdaki içeriklere göre yanıt ver:\n{context}\n\nCevap:"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ GPT yanıt hatası: {str(e)}"
