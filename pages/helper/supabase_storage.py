import uuid
import streamlit as st
from supabase import create_client

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET_NAME = "missing-person-images"


def upload_image(file):

    try:
        file_ext = file.name.split(".")[-1]
        filename = f"{uuid.uuid4()}.{file_ext}"

        file_bytes = file.read()

        supabase.storage.from_(BUCKET_NAME).upload(
            filename,
            file_bytes,
            {"content-type": file.type},
        )

        url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)

        return url

    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None
