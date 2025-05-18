from fastapi import FastAPI
from gradio_interface import build_interface
from dotenv import load_dotenv
from gradio.routes import mount_gradio_app  # ✅ bu şart

load_dotenv()

app = FastAPI()
gr_app = build_interface()

# ✅ Gradio 3.5 ile FastAPI entegrasyonu
app = mount_gradio_app(app, gr_app, path="/")

@app.get("/")
async def status():
    return {"durum": "Gradio uygulaması çalışıyor"}
