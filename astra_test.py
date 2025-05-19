import os, requests
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT") + "/api/embeddings/text"
headers = {
    "Authorization": "Bearer " + API_KEY,
    "Content-Type": "application/json"
}
data = {
    "text": "test sentence",
    "model": "openai/text-embedding-ada-002"
}
response = requests.post(ENDPOINT, headers=headers, json=data)
print(response.status_code)
print(response.text)
