import os
import uuid
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET_NAME = "missing-person-images"


def upload_image(file_or_bytes, original_filename=None):
    """
    Flexible upload function.
    Works with:
    1) upload_image(file_object)
    2) upload_image(file_bytes, filename)
    """
    try:
        # Case 1: file object from Streamlit uploader
        if original_filename is None:
            file_obj = file_or_bytes
            file_bytes = file_obj.getvalue()
            original_filename = file_obj.name
            content_type = file_obj.type
        else:
            # Case 2: bytes + filename
            file_bytes = file_or_bytes
            content_type = "image/jpeg"

        # Extract extension
        file_ext = original_filename.split(".")[-1]

        # Unique filename
        unique_filename = f"{uuid.uuid4()}.{file_ext}"

        # Upload to Supabase
        supabase.storage.from_(BUCKET_NAME).upload(
            unique_filename,
            file_bytes,
            {"content-type": content_type}
        )

        # Get public URL
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(unique_filename)

        return public_url

    except Exception as e:
        print("Upload error:", e)
        return None
