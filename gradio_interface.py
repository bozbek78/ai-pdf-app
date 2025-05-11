import gradio as gr
import os
from process_pdf_backend import process_pdf_to_astra
from openai_smart_query import query_openai_with_astra_context

def build_interface():
    with gr.Blocks(title="AI PDF Engine") as demo:
        with gr.Tab("📄 PDF Yükle"):
            gr.Markdown("## 📘 PDF'den Metin ve Görsel Yükleyici")
            file_input = gr.File(label="📄 PDF Yükle", file_types=[".pdf"])
            output_log = gr.Textbox(label="📤 İşlem Günlüğü", lines=20, interactive=False)
            process_button = gr.Button("🚀 İşlem Başlat")
            process_button.click(fn=process_pdf_to_astra, inputs=[file_input], outputs=[output_log])

        with gr.Tab("🔍 Soru Sor"):
            gr.Markdown("## 🔍 PDF'ten Vektörel Bilgi Cevaplama")
            query_input = gr.Textbox(label="Soru", placeholder="örnek: iç vida çapı nedir?")
            query_output = gr.Textbox(label="Yanıt", lines=8, interactive=False)
            query_button = gr.Button("🧠 Yanıtla")
            query_button.click(fn=query_openai_with_astra_context, inputs=[query_input], outputs=[query_output])
    return demo
