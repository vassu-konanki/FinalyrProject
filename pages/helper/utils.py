import os
import numpy as np
import cv2
import PIL
import streamlit as st
from insightface.app import FaceAnalysis

# ==============================
# DATABASE PATH
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "cases.db")


# ==============================
# LOAD FACE MODEL
# ==============================

@st.cache_resource
def load_model():

    model = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    model.prepare(ctx_id=-1, det_size=(640, 640))

    return model


app = load_model()


# ==============================
# IMAGE → NUMPY
# ==============================

def image_obj_to_numpy(image_obj):

    image = PIL.Image.open(image_obj).convert("RGB")

    return np.array(image)


# ==============================
# EXTRACT FACE EMBEDDING
# ==============================

def extract_face_embedding(image_rgb):

    try:

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        faces = app.get(image_bgr)

        if len(faces) == 0:

            st.error("No face detected")

            return None

        embedding = faces[0].embedding

        return embedding.astype(float).tolist()

    except Exception as e:

        st.error(f"Face extraction failed: {e}")

        return None
