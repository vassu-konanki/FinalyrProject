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

    model = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
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
# 🔥 AGGRESSIVE CROPPING
# ==============================

def aggressive_crops(image):
    h, w, _ = image.shape

    return [
        image[0:int(h*0.5), int(w*0.2):int(w*0.8)],   # top-center
        image[0:int(h*0.6), :],                       # full top
        image[int(h*0.1):int(h*0.7), int(w*0.2):int(w*0.8)],
        image[int(h*0.2):int(h*0.8), int(w*0.3):int(w*0.7)],
    ]

def extract_face_embedding(image_rgb):

    if app is None or cv2 is None:
        return None

    try:
        # 🔥 STEP 1: NORMAL DETECTION
        resized = cv2.resize(image_rgb, (640, 640))
        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

        faces = app.get(resized_bgr)

        if faces:
            return faces[0].embedding.astype(float).tolist()

        # 🔥 STEP 2: FALLBACK (VERY IMPORTANT)
        # 👉 Even if face not detected, create embedding

        # Normalize image
        img = resized / 255.0

        # Flatten + reduce size
        embedding = img.flatten()[:512]   # fixed size

        return embedding.tolist()

    except Exception as e:
        print("Embedding error:", e)
        return None


# ==============================
# FINAL EMBEDDING FUNCTION
# ==============================

def extract_face_embedding(image_rgb):

    if app is None or cv2 is None:
        return None

    try:
        # 🔥 STEP 1: FORCE CROPS FIRST (KEY FIX)
        crops = aggressive_crops(image_rgb)

        for crop in crops:
            if crop is None or crop.size == 0:
                continue

            crop = cv2.resize(crop, (640, 640))
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

            faces = app.get(crop_bgr)

            if faces:
                return faces[0].embedding.astype(float).tolist()

        # 🔥 STEP 2: FULL IMAGE RESIZE
        resized = cv2.resize(image_rgb, (640, 640))
        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

        faces = app.get(resized_bgr)

        if faces:
            return faces[0].embedding.astype(float).tolist()

        # 🔥 STEP 3: FINAL FAIL
        return None

    except Exception as e:
        print("Face detection error:", e)
        return None
