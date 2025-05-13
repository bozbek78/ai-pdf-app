import os
import mimetypes
import json
import logging

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_FILE = "/etc/secrets/token.json"

def get_drive_service():
    """Return an authenticated Google Drive service client."""
    if not os.path.exists(TOKEN_FILE):
        logging.error("❌ token.json bulunamadı: %s", TOKEN_FILE)
        raise FileNotFoundError(
            "Google Drive token dosyası bulunamadı. İlk çalıştırmada 'token.json' oluşturulmalıdır."
        )

    logging.info("✅ token.json bulundu, okunuyor…")

    # JSON doğruluğunu da kontrol edelim
    try:
        with open(TOKEN_FILE, "r") as f:
            creds_dict = json.load(f)
        logging.info("✅ token.json JSON formatı geçerli.")
    except json.JSONDecodeError as exc:
        logging.exception("❌ token.json bozuk JSON! %s", exc)
        raise

    creds = Credentials.from_authorized_user_info(creds_dict, SCOPES)
    return build("drive", "v3", credentials=creds)


def get_or_create_folder(service, folder_name: str) -> str:
    """Return Google Drive folder ID, creating the folder if necessary."""
    query = (
        f"name='{folder_name}' and "
        "mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = (
        service.files()
        .list(q=query, spaces="drive", fields="files(id, name)")
        .execute()
    )
    items = results.get("files", [])
    if items:
        return items[0]["id"]

    file_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    file = service.files().create(body=file_metadata, fields="id").execute()
    return file.get("id")


def upload_image_to_drive(image_path: str, pdf_folder_name: str) -> str:
    """Upload an image to the target PDF‑named folder; return file ID."""
    service = get_drive_service()
    folder_id = get_or_create_folder(service, pdf_folder_name)

    file_metadata = {
        "name": os.path.basename(image_path),
        "parents": [folder_id],
    }
    mime_type, _ = mimetypes.guess_type(image_path)
    media = MediaFileUpload(image_path, mimetype=mime_type)

    uploaded = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )
    logging.info("📤 Yüklendi: %s (id=%s)", image_path, uploaded["id"])
    return uploaded["id"]
