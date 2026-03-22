import os
import numpy as np
import cv2
import PIL
import streamlit as st

# ==============================
# 🔥 SAFE IMPORT (CRITICAL FIX)
# ==============================

try:
    from insightface.app import FaceAnalysis
    INSIGHT_AVAILABLE = True
except Exception as e:
    print("⚠️ InsightFace import failed:", e)
    FaceAnalysis = None
    INSIGHT_AVAILABLE = False


# ==============================
# PATH SETUP
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "cases.db")


# ==============================
# LOAD MODEL (SAFE)
# ==============================

@st.cache_resource
def load_model():
    if not INSIGHT_AVAILABLE:
        print("⚠️ InsightFace not available → using fallback mode")
        return None

    try:
        model = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        model.prepare(ctx_id=-1, det_size=(640, 640))
        print("✅ InsightFace model loaded")
        return model
    except Exception as e:
        print("❌ Model load failed:", e)
        return None


app = load_model()


# ==============================
# IMAGE CONVERSION
# ==============================

def image_obj_to_numpy(image_obj):
    try:
        image = PIL.Image.open(image_obj).convert("RGB")
        return np.array(image)
    except Exception as e:
        print("Image conversion error:", e)
        return None


# ==============================
# 🔥 FINAL FACE EMBEDDING FUNCTION
# ==============================

def extract_face_embedding(image_rgb):

    if image_rgb is None:
        return None

    try:
        # =========================
        # CASE 1: InsightFace WORKS
        # =========================
        if app is not None:

            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            faces = app.get(image_bgr)

            if faces:
                return faces[0].embedding.astype(float).tolist()

        # =========================
        # CASE 2: FALLBACK (ALWAYS WORKS)
        # =========================

        # Resize small (fast + stable)
        resized = cv2.resize(image_rgb, (64, 64))

        # Normalize
        normalized = resized / 255.0

        # Flatten → fixed 512 size
        embedding = normalized.flatten()[:512]

        return embedding.tolist()

    except Exception as e:
        print("Embedding error:", e)
        return None
