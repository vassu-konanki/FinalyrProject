import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client

# ==============================
# LOAD ENV VARIABLES
# ==============================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials are missing. Check environment variables.")

# ==============================
# CREATE SUPABASE CLIENT
# ==============================

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Storage bucket name
BUCKET_NAME = "missing-person-images"


# ==============================
# UPLOAD IMAGE FUNCTION
# ==============================

def upload_image(file_obj):
    """
    Upload image to Supabase Storage and return public URL
    """

    try:
        # Generate unique filename
        file_ext = file_obj.name.split(".")[-1]
        filename = f"{uuid.uuid4()}.{file_ext}"

        file_bytes = file_obj.read()

        # Upload file
        supabase.storage.from_(BUCKET_NAME).upload(
            filename,
            file_bytes,
            {"content-type": file_obj.type}
        )

        # Get public URL
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)

        return public_url

    except Exception as e:
        st.error(f"Image upload failed: {e}")
        return None
