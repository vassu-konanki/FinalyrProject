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

try:
    import mediapipe as mp
except:
    mp = None


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
# 🔥 STRONG MEDIAPIPE DETECTOR
# ==============================

def detect_face_mediapipe(image):
    if mp is None:
        return None

    mp_face = mp.solutions.face_detection

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.2) as face_detection:
        results = face_detection.process(image)

        if not results.detections:
            return None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        h, w, _ = image.shape

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        w_box = int(bbox.width * w)
        h_box = int(bbox.height * h)

        # 🔥 VERY IMPORTANT: LARGE PADDING
        pad = 120
        x = max(0, x - pad)
        y = max(0, y - pad)

        face = image[y:y + h_box + pad, x:x + w_box + pad]

        if face is None or face.size == 0:
            return None

        return face


# ==============================
# 🔥 FORCE ZOOM CROPS
# ==============================

def force_face_zoom(image):
    h, w, _ = image.shape

    crops = [
        image[0:int(h*0.5), int(w*0.25):int(w*0.75)],   # top-center
        image[int(h*0.1):int(h*0.6), int(w*0.2):int(w*0.8)],
        image[int(h*0.2):int(h*0.7), int(w*0.3):int(w*0.7)],
    ]

    return crops


# ==============================
# FINAL EMBEDDING FUNCTION
# ==============================

def extract_face_embedding(image_rgb):

    if app is None or cv2 is None:
        return None

    try:
        # =========================
        # STEP 1: MediaPipe FIRST
        # =========================
        face_crop = detect_face_mediapipe(image_rgb)

        if face_crop is not None:
            face_crop = cv2.resize(face_crop, (640, 640))
            face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)

            faces = app.get(face_bgr)

            if faces:
                return faces[0].embedding.astype(float).tolist()

        # =========================
        # STEP 2: FORCE ZOOM
        # =========================
        crops = force_face_zoom(image_rgb)

        for crop in crops:
            if crop is None or crop.size == 0:
                continue

            crop = cv2.resize(crop, (640, 640))
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

            faces = app.get(crop_bgr)

            if faces:
                return faces[0].embedding.astype(float).tolist()

        # =========================
        # STEP 3: FINAL TRY
        # =========================
        resized = cv2.resize(image_rgb, (640, 640))
        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

        faces = app.get(resized_bgr)

        if faces:
            return faces[0].embedding.astype(float).tolist()

        return None

    except Exception as e:
        print("Face detection error:", e)
        return None
