import os
import numpy as np
import streamlit as st
import PIL

# Safe OpenCV import
try:
    import cv2
except Exception as e:
    st.error(f"OpenCV failed to load: {e}")

# Safe InsightFace import
try:
    from insightface.app import FaceAnalysis
except Exception as e:
    st.error(f"InsightFace failed to load: {e}")


# ==============================
# DATABASE PATH
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "cases.db")


# ==============================
# LOAD FACE MODEL (CACHED)
# ==============================

@st.cache_resource
def load_model():
    """
    Load InsightFace model only once
    """

    try:
        model = FaceAnalysis(
            name="buffalo_sc",   # smaller model for cloud
            providers=["CPUExecutionProvider"]
        )

        model.prepare(ctx_id=-1, det_size=(640, 640))

        return model

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


app = load_model()


# ==============================
# IMAGE → NUMPY
# ==============================

def image_obj_to_numpy(image_obj):
    """
    Convert uploaded Streamlit image → numpy array
    """

    try:
        image = PIL.Image.open(image_obj).convert("RGB")
        return np.array(image)

    except Exception as e:
        st.error(f"Image conversion failed: {e}")
        return None


# ==============================
# EXTRACT FACE EMBEDDING
# ==============================

def extract_face_embedding(image_rgb):

    if app is None:
        st.error("Face model not loaded")
        return None

    try:

        # Convert RGB → BGR (OpenCV format)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        faces = app.get(image_bgr)

        if faces is None or len(faces) == 0:
            st.warning("⚠️ No face detected. Upload a clear face image.")
            return None

        embedding = faces[0].embedding

        return embedding.astype(float).tolist()

    except Exception as e:
        st.error(f"Face extraction failed: {e}")
        return None
