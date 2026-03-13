import os
import numpy as np
import streamlit as st
import PIL

try:
    import cv2
except Exception as e:
    st.error(f"OpenCV failed: {e}")

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    st.error(f"InsightFace failed: {e}")

# ==============================
# DATABASE PATH
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "cases.db")

# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():
    try:
        model = FaceAnalysis(
            name="buffalo_sc",  # smaller model
            providers=["CPUExecutionProvider"]
        )
        model.prepare(ctx_id=-1)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

app = load_model()

# ==============================
# IMAGE → NUMPY
# ==============================

def image_obj_to_numpy(image_obj):
    image = PIL.Image.open(image_obj).convert("RGB")
    return np.array(image)

# ==============================
# EXTRACT EMBEDDING
# ==============================

def extract_face_embedding(image_rgb):

    if app is None:
        st.error("Face model not loaded")
        return None

    try:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        faces = app.get(image_bgr)

        if len(faces) == 0:
            st.warning("No face detected")
            return None

        embedding = faces[0].embedding

        return embedding.astype(float).tolist()

    except Exception as e:
        st.error(f"Face extraction failed: {e}")
        return None
