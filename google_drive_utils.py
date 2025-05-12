import os
import mimetypes
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

def get_drive_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    else:
        raise Exception("Google Drive token dosyası bulunamadı. İlk çalıştırmada 'token.json' oluşturulmalıdır.")
    return build('drive', 'v3', credentials=creds)

def get_or_create_folder(service, folder_name):
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields="files(id, name)").execute()
    items = results.get('files', [])
    if items:
        return items[0]['id']
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    file = service.files().create(body=file_metadata, fields='id').execute()
    return file.get('id')

def upload_image_to_drive(image_path, pdf_folder_name):
    service = get_drive_service()
    folder_id = get_or_create_folder(service, pdf_folder_name)

    file_metadata = {
        'name': os.path.basename(image_path),
        'parents': [folder_id]
    }
    mime_type, _ = mimetypes.guess_type(image_path)
    media = MediaFileUpload(image_path, mimetype=mime_type)
    uploaded_file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return uploaded_file.get('id')
