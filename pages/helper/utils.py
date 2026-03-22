import os
import numpy as np
import cv2
import PIL
import streamlit as st
from insightface.app import FaceAnalysis

# ==============================
# PATH SETUP
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "cases.db")


# ==============================
# LOAD MODEL (FIXED)
# ==============================

@st.cache_resource
def load_model():
    model = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    model.prepare(ctx_id=-1, det_size=(640, 640))  # CPU FIX
    return model


app = load_model()


# ==============================
# IMAGE CONVERSION
# ==============================

def image_obj_to_numpy(image_obj):
    image = PIL.Image.open(image_obj).convert("RGB")
    return np.array(image)


# ==============================
# 🔥 FACE EMBEDDING (FINAL FIX)
# ==============================

def extract_face_embedding(image_rgb):

    try:
        h, w, _ = image_rgb.shape

        # ---------------------------
        # STEP 1: NORMAL DETECTION
        # ---------------------------
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        faces = app.get(image_bgr)

        if faces:
            return faces[0].embedding.astype(float).tolist()

        # ---------------------------
        # STEP 2: CROP UPPER BODY
        # ---------------------------
        crop = image_rgb[0:int(h * 0.6), int(w * 0.2):int(w * 0.8)]

        crop = cv2.resize(crop, (640, 640))
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        faces = app.get(crop_bgr)

        if faces:
            return faces[0].embedding.astype(float).tolist()

        # ---------------------------
        # STEP 3: FORCE RESIZE
        # ---------------------------
        resized = cv2.resize(image_rgb, (640, 640))
        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

        faces = app.get(resized_bgr)

        if faces:
            return faces[0].embedding.astype(float).tolist()

        # ---------------------------
        # STEP 4: FINAL FALLBACK (IMPORTANT)
        # ---------------------------
        st.warning("⚠️ Face not clearly detected. Using fallback embedding.")

        flat = resized.flatten()[:512]
        return flat.astype(float).tolist()

    except Exception as e:
        st.error(f"Face extraction failed: {str(e)}")
        return None
