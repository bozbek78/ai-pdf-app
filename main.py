import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)

import gradio as gr
from fastapi import FastAPI
from gradio_interface import build_interface


app = FastAPI()

gr_app = build_interface()
app = gr.mount_gradio_app(app, gr_app, path="/")

@app.get("/")
async def status():
    return {"durum": "Gradio uygulaması çalışıyor"}
