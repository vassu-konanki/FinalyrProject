import os
import numpy as np
import streamlit as st
import PIL

# Safe imports
try:
    import cv2
except Exception as e:
    st.error(f"OpenCV failed to load: {e}")

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    st.error(f"InsightFace failed to load: {e}")
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

    if FaceAnalysis is None:
        return None

    try:
        model = FaceAnalysis(
            name="buffalo_sc",  # smaller model
            providers=["CPUExecutionProvider"]
        )

        # ctx_id = -1 → CPU
        model.prepare(ctx_id=-1, det_size=(640, 640))

        return model

    except Exception as e:
        st.error(f"Face model failed to load: {e}")
        return None


app = load_face_model()


# ==============================
# IMAGE PROCESSING
# ==============================

def image_obj_to_numpy(image_obj) -> np.ndarray:
    """
    Convert Streamlit image object to RGB numpy array
    """
    image = PIL.Image.open(image_obj).convert("RGB")
    return np.array(image)


# ==============================
# EXTRACT FACE EMBEDDING
# ==============================

def extract_face_embedding(image_rgb: np.ndarray):

    if app is None:
        st.error("Face model is not loaded.")
        return None

    try:
        # Convert RGB → BGR for OpenCV
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
