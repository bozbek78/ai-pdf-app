import gradio as gr
from openai_utils import (
    query_openai_with_astra_context,
    auto_label_image,
    update_image_label,
)
from process_pdf import process_pdf_to_astra, list_images


def build_interface():
    with gr.Blocks(title="Akıllı PDF Yorumlayıcı") as demo:
        with gr.Tab("📂 PDF Yükle"):
            pdf_input = gr.File(label="PDF dosyalarını yükleyin", file_types=[".pdf"], file_count="multiple")
            upload_btn = gr.Button("PDF'leri işle ve yükle")
            upload_output = gr.Textbox(label="İşlem Sonuçları")

            upload_btn.click(fn=process_pdf_to_astra, inputs=pdf_input, outputs=upload_output)

        with gr.Tab("🔍 Soru Sor"):
            question = gr.Textbox(label="Sorunuzu yazın")
            ask_btn = gr.Button("Yanıtla")
            answer = gr.Textbox(label="GPT Yanıtı")
            sources = gr.Textbox(label="Kullanılan Belgeler", lines=6)

            def ask_with_sources(q):
                reply, srcs = query_openai_with_astra_context(q)
                return reply, srcs

            ask_btn.click(fn=ask_with_sources, inputs=question, outputs=[answer, sources])

        with gr.Tab("🖼️ Görseller ve Etiketleme"):
            image_list = list_images()
            image_dropdown = gr.Dropdown(choices=image_list, label="Bir görsel seçin")
            image_gallery = gr.Image(label="Seçilen Görsel", type="filepath")
            label_input = gr.Textbox(label="Etiket (ne görünüyor?)")
            auto_label_btn = gr.Button("Otomatik Etiketle")
            save_label_btn = gr.Button("Etiketi Kaydet")
            label_result = gr.Textbox(label="Durum")

            def on_select_image(file_path):
                return file_path, ""

            image_dropdown.change(fn=on_select_image, inputs=image_dropdown, outputs=[image_gallery, label_input])
            auto_label_btn.click(fn=auto_label_image, inputs=image_dropdown, outputs=label_input)
            save_label_btn.click(fn=update_image_label, inputs=[image_dropdown, label_input], outputs=label_result)

    return demo
