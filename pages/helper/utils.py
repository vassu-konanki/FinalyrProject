import numpy as np
import PIL
import streamlit as st

try:
    import cv2
except:
    cv2 = None

try:
    from insightface.app import FaceAnalysis
except:
    FaceAnalysis = None


# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_face_model():
    if FaceAnalysis is None or cv2 is None:
        return None

    model = FaceAnalysis(
        name="buffalo_sc",
        providers=["CPUExecutionProvider"]
    )
    model.prepare(ctx_id=-1, det_size=(640, 640))
    return model


app = load_face_model()


# ==============================
# IMAGE CONVERSION
# ==============================

def image_obj_to_numpy(image_obj):
    image = PIL.Image.open(image_obj).convert("RGB")
    return np.array(image)


# ==============================
# 🔥 FINAL EMBEDDING FUNCTION
# ==============================

def extract_face_embedding(image_rgb):

    if app is None or cv2 is None:
        return None

    try:
        h, w, _ = image_rgb.shape

        # 🔥 STEP 1: FORCE RESIZE (MOST IMPORTANT)
        resized = cv2.resize(image_rgb, (640, 640))
        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

        faces = app.get(resized_bgr)

        if faces:
            return faces[0].embedding.astype(float).tolist()

        # 🔥 STEP 2: TRY MULTIPLE ZOOMS (NO DETECTOR)
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
