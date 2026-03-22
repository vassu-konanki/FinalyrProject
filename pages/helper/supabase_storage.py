import os
import uuid
import streamlit as st
from supabase import create_client


# ==============================
# LOAD SECRETS (CLOUD SAFE)
# ==============================

SUPABASE_URL = None
SUPABASE_KEY = None

if "SUPABASE_URL" in st.secrets:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
else:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")


# ==============================
# VALIDATION
# ==============================

if not SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ Supabase not configured")
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


BUCKET_NAME = "missing-person-images"


# ==============================
# UPLOAD FUNCTION (FIXED)
# ==============================

def upload_image(file_obj):
    """
    Upload image to Supabase storage
    """

    if supabase is None:
        return None

    try:
        # 🔥 Reset pointer
        file_obj.seek(0)

        file_bytes = file_obj.read()
        file_name = file_obj.name
        file_ext = file_name.split(".")[-1]

        unique_name = f"{uuid.uuid4()}.{file_ext}"

        # Upload
        res = supabase.storage.from_(BUCKET_NAME).upload(
            path=unique_name,
            file=file_bytes,
            file_options={"content-type": file_obj.type}
        )

        # 🔥 Check error
        if hasattr(res, "error") and res.error:
            print("Upload error:", res.error)
            return None

        # Get public URL
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(unique_name)

        return public_url

    except Exception as e:
        print("Upload exception:", e)
        return None
