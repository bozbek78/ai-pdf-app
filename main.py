from fastapi import FastAPI
from gradio_interface import build_interface
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
gr_app = build_interface()
app = gr_app.mount_gradio_app(app, gr_app, path="/")

@app.get("/")
async def status():
    return {"durum": "Gradio uygulaması çalışıyor"}
