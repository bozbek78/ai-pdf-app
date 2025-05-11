import gradio as gr
from gradio_interface import build_interface

gr_app = build_interface()

if __name__ == "__main__":
    gr_app.launch(server_name="0.0.0.0", server_port=7860)
