
import gradio as gr
import os
from process_pdf_backend import process_pdf_to_astra
from openai_smart_query import query_openai_with_astra_context
from PIL import Image

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

        with gr.Tab("🖼️ Görsel Etiketleme"):
            gr.Markdown("## 🏷️ PDF Görsellerini Etiketle ve Kaydet")
            image_gallery = gr.Gallery(label="📷 Çıkarılmış Görseller", show_label=True, elem_id="image_gallery")
            tag_input = gr.Textbox(label="Etiket (örnek: teknik çizim, vida ucu)")
            save_button = gr.Button("💾 Etiketi Kaydet")
            tag_output = gr.Textbox(label="Durum", interactive=False)

            def list_images():
                folder = "pdf_images"
                if not os.path.exists(folder):
                    return []
                return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

            def save_tag(tag):
                with open("image_tags.csv", "a", encoding="utf-8") as f:
                    f.write(f"{tag}\n")
                return "✅ Etiket kaydedildi."

            image_gallery.update(value=list_images())
            save_button.click(fn=save_tag, inputs=[tag_input], outputs=[tag_output])

    return demo
