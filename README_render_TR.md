# Akıllı PDF Yorumlayıcı (Türkçe)

Bu uygulama, PDF dosyalarını yükleyerek metin, tablo ve görselleri çıkarır, vektörleştirir, AstraDB'ye kaydeder ve GPT ile içerik bazlı soruları yanıtlar. Görseller Google Drive'a klasörlenerek yüklenir ve etiketlenebilir.

## 🚀 Render Üzerinden Kurulum

1. Render hesabınıza giriş yapın: https://render.com/
2. Yeni Web Service oluşturun → "Deploy from GitHub" yerine `.zip` yükleyerek yükleyin
3. Ortam değişkenlerini tanımlayın:

```
OPENAI_API_KEY = OpenAI API anahtarınız
ASTRA_DB_API_ENDPOINT = Astra veritabanı API URL
ASTRA_DB_APPLICATION_TOKEN = Astra erişim token'ınız
```

4. Deploy sırasında: `render.yaml` dosyası otomatik ayarları yükleyecek

## 🧪 Özellikler

- 📄 Toplu PDF yükleme
- 🔍 Soru sorarak içeriklere GPT ile erişim
- 🖼 Görsel kesme + klasörlü Drive yükleme
- 🏷 Etiket düzenleme ve AstraDB’ye kaydetme
- 📚 Kullanılan belgelerin gösterimi

## ⚠️ İlk Kurulumda Gerekli Dosya

Google Drive bağlantısı için:
- `credentials.json` (Google Console'dan alınmalı)
- İlk çalıştırmada `token.json` oluşacaktır

---

## 📂 Klasör Yapısı

```
ai_pdf_app/
├── main.py
├── gradio_interface.py
├── process_pdf.py
├── openai_utils.py
├── google_drive_utils.py
├── requirements.txt
├── render.yaml
├── README_render_TR.md
├── templates/
├── static/
└── pdf_images/
```
