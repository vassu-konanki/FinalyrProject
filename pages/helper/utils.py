import os
import numpy as np
import cv2
import PIL
import streamlit as st
from insightface.app import FaceAnalysis

# ==============================
# CENTRAL DATABASE PATH
# ==============================

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Create data folder at root
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Shared database path
DB_PATH = os.path.join(DATA_DIR, "cases.db")


# ==============================
# LOAD FACE MODEL (CACHED)
# ==============================

@st.cache_resource
def load_face_model():
    """
    Load InsightFace model once and reuse.
    Streamlit cache prevents repeated downloads.
    """
    model = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    # ctx_id = -1 → CPU (Streamlit Cloud has no GPU)
    model.prepare(ctx_id=-1, det_size=(640, 640))

    return model


# Load model
app = load_face_model()


# ==============================
# IMAGE PROCESSING
# ==============================

def image_obj_to_numpy(image_obj) -> np.ndarray:
    """
    Convert Streamlit uploaded image to RGB numpy array
    """
    image = PIL.Image.open(image_obj).convert("RGB")
    return np.array(image)


# ==============================
# FACE EMBEDDING EXTRACTION
# ==============================

def extract_face_embedding(image_rgb: np.ndarray):
    """
    Extract 512D face embedding using InsightFace
    """

    try:

        # Convert RGB → BGR (required by OpenCV / InsightFace)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        faces = app.get(image_bgr)

        if faces is None or len(faces) == 0:
            st.error("⚠️ No face detected. Please upload a clear face image.")
            return None

        embedding = faces[0].embedding  # 512 dimensional vector

        return embedding.astype(float).tolist()

    except Exception as e:
        st.error(f"Face extraction failed: {str(e)}")
        return None
