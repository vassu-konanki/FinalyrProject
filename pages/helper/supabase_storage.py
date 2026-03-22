import os
import uuid
import streamlit as st
from supabase import create_client

# ==============================
# LOAD ENV (LOCAL + CLOUD SAFE)
# ==============================

SUPABASE_URL = None
SUPABASE_KEY = None

# ✅ 1. Try Streamlit secrets (CLOUD)
try:
    if "SUPABASE_URL" in st.secrets:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    pass

# ✅ 2. Fallback to .env (LOCAL)
if not SUPABASE_URL:
    try:
        from dotenv import load_dotenv
        load_dotenv()

        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    except Exception:
        pass


# ==============================
# INIT CLIENT (SAFE)
# ==============================

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None


BUCKET_NAME = "missing-person-images"


# ==============================
# UPLOAD FUNCTION
# ==============================

def upload_image(file_or_bytes, original_filename=None):
    """
    Supports:
    1) upload_image(file_object)
    2) upload_image(file_bytes, filename)
    """

    # ❌ Prevent crash if credentials missing
    if supabase is None:
        st.error("❌ Supabase is not configured. Add credentials in Streamlit secrets.")
        return None

    try:
        # ✅ CASE 1: Streamlit file object
        if original_filename is None:
            file_obj = file_or_bytes

            # Reset pointer (VERY IMPORTANT)
            file_obj.seek(0)

            file_bytes = file_obj.read()
            original_filename = file_obj.name
            content_type = file_obj.type

        # ✅ CASE 2: bytes + filename
        else:
            file_bytes = file_or_bytes
            content_type = "image/jpeg"

        # Extract extension safely
        file_ext = original_filename.split(".")[-1].lower()

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
        st.error("❌ Image upload failed.")
        return None
