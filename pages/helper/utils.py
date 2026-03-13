import os
import numpy as np
import PIL
import streamlit as st

# Safe imports without Streamlit UI
try:
    import cv2
except Exception:
    cv2 = None

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None


# ==============================
# CENTRAL DATABASE PATH
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "cases.db")


# ==============================
# FACE MODEL (LOAD ONCE)
# ==============================

@st.cache_resource
def load_face_model():

    if FaceAnalysis is None or cv2 is None:
        return None

    try:
        model = FaceAnalysis(
            name="buffalo_sc",
            providers=["CPUExecutionProvider"]
        )

        model.prepare(ctx_id=-1, det_size=(640, 640))

        return model

    except Exception:
        return None


app = load_face_model()


# ==============================
# IMAGE PROCESSING
# ==============================

def image_obj_to_numpy(image_obj) -> np.ndarray:

    image = PIL.Image.open(image_obj).convert("RGB")
    return np.array(image)


# ==============================
# EXTRACT FACE EMBEDDING
# ==============================

def extract_face_embedding(image_rgb: np.ndarray):

    if app is None:
        return None

    try:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        faces = app.get(image_bgr)

        if not faces:
            return None

        embedding = faces[0].embedding

        return embedding.astype(float).tolist()

    except Exception:
        return None
