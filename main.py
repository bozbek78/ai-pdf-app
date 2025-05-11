import gradio as gr
from gradio_interface import build_interface

if __name__ == "__main__":
    app = build_interface()
    app.launch(server_name="0.0.0.0", server_port=7860)
