import os
import numpy as np
import cv2
import PIL
import streamlit as st
from insightface.app import FaceAnalysis

# ==============================
# CENTRAL DATABASE PATH (FIX)
# ==============================

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Create data folder at root
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Shared database path
DB_PATH = os.path.join(DATA_DIR, "cases.db")


# ==============================
# FACE MODEL (LOAD ONCE)
# ==============================

# Load InsightFace model once (global singleton)
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))


# ==============================
# IMAGE PROCESSING
# ==============================

def image_obj_to_numpy(image_obj) -> np.ndarray:
    """
    Convert Streamlit image object to RGB numpy array
    (for correct UI display)
    """
    image = PIL.Image.open(image_obj).convert("RGB")
    return np.array(image)


def extract_face_embedding(image_rgb: np.ndarray):
    """
    Extract 512-D identity embedding using InsightFace.
    Converts RGB → BGR internally (required by InsightFace).
    """
    try:
        # Convert RGB → BGR for model
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        faces = app.get(image_bgr)

        if faces is None or len(faces) == 0:
            st.error("No face detected. Please upload a clear face image.")
            return None

        embedding = faces[0].embedding  # (512,)
        return embedding.astype(float).tolist()

    except Exception as e:
        st.error(f"Face extraction failed: {str(e)}")
        return None