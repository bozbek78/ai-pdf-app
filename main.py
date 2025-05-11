
import gradio as gr
import uvicorn
from fastapi import FastAPI
from gradio_interface import build_interface

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Gradio app is running"}

gradio_app = build_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860)
