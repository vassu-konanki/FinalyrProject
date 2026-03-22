import os
import uuid
import streamlit as st
from supabase import create_client

# ==============================
# LOAD ENV (LOCAL + CLOUD SAFE)
# ==============================

SUPABASE_URL = None
SUPABASE_KEY = None

if "SUPABASE_URL" in st.secrets:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv()

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")


# ==============================
# INIT CLIENT
# ==============================

if not SUPABASE_URL or not SUPABASE_KEY:
    st.warning("⚠️ Supabase not configured")
    supabase = None
else:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"❌ Supabase init failed: {e}")
        supabase = None


BUCKET_NAME = "missing-person-images"


# ==============================
# ✅ FINAL FIXED UPLOAD FUNCTION
# ==============================

def upload_image(file_or_bytes, original_filename=None):

    if supabase is None:
        st.error("❌ Supabase not initialized")
        return None

    try:
        # =========================
        # CASE 1: file object (Streamlit)
        # =========================
        if hasattr(file_or_bytes, "read"):

            file_obj = file_or_bytes

            file_obj.seek(0)  # ✅ safe now

            file_bytes = file_obj.read()
            original_filename = file_obj.name

        # =========================
        # CASE 2: bytes
        # =========================
        else:
            file_bytes = file_or_bytes

            if original_filename is None:
                original_filename = "image.jpg"

        # =========================
        # FILE NAME
        # =========================
        file_ext = original_filename.split(".")[-1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_ext}"

        # =========================
        # UPLOAD
        # =========================
        supabase.storage.from_(BUCKET_NAME).upload(
            unique_filename,
            file_bytes
        )

        # =========================
        # GET PUBLIC URL
        # =========================
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(unique_filename)

        return public_url

    except Exception as e:
        st.error(f"❌ Upload exception: {e}")
        return None
