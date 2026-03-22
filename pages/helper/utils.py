import numpy as np
import PIL
import streamlit as st

# Safe imports
try:
    import cv2
except Exception:
    cv2 = None

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None


# ==============================
# LOAD MODEL (SAFE)
# ==============================

@st.cache_resource
def load_face_model():
    if FaceAnalysis is None or cv2 is None:
        print("⚠️ InsightFace or OpenCV not available")
        return None

    try:
        model = FaceAnalysis(
            name="buffalo_sc",
            providers=["CPUExecutionProvider"]
        )
        model.prepare(ctx_id=-1, det_size=(640, 640))
        return model
    except Exception as e:
        print("Model load error:", e)
        return None


app = load_face_model()


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
# FACE EMBEDDING (SAFE VERSION)
# ==============================

def extract_face_embedding(image_rgb):

    # ✅ Prevent crash
    if image_rgb is None:
        return None

    if app is None or cv2 is None:
        print("⚠️ Model not loaded")
        return None

    try:
        # Resize
        resized = cv2.resize(image_rgb, (640, 640))
        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

        faces = app.get(resized_bgr)

        if faces:
            return faces[0].embedding.astype(float).tolist()

        # 🔥 fallback zooms
        h, w, _ = image_rgb.shape

        zooms = [
            image_rgb[0:int(h*0.6), :],
            image_rgb[int(h*0.1):int(h*0.7), :],
            image_rgb[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)],
        ]

        for z in zooms:
            if z is None or z.size == 0:
                continue

            z = cv2.resize(z, (640, 640))
            z_bgr = cv2.cvtColor(z, cv2.COLOR_RGB2BGR)

            faces = app.get(z_bgr)

            if faces:
                return faces[0].embedding.astype(float).tolist()

        return None

    except Exception as e:
        print("Face embedding error:", e)
        return None
