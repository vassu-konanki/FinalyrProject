import os
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

try:
    import mediapipe as mp
except Exception:
    mp = None


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
# IMAGE CONVERSION
# ==============================

def image_obj_to_numpy(image_obj) -> np.ndarray:
    image = PIL.Image.open(image_obj).convert("RGB")
    return np.array(image)


# ==============================
# PREPROCESS IMAGE (VERY IMPORTANT)
# ==============================

def preprocess_image(image_rgb):
    """
    Improves detection accuracy
    """
    if cv2 is None:
        return image_rgb

    # Resize to standard size
    image_rgb = cv2.resize(image_rgb, (640, 640))

    # Improve contrast
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.equalizeHist(l)

    lab = cv2.merge((l, a, b))
    image_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return image_rgb


# ==============================
# MEDIAPIPE FACE DETECTION
# ==============================

def detect_face_mediapipe(image_rgb):
    if mp is None:
        return None

    mp_face = mp.solutions.face_detection

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.3) as face_detection:
        results = face_detection.process(image_rgb)

        if not results.detections:
            return None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        h, w, _ = image_rgb.shape

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        w_box = int(bbox.width * w)
        h_box = int(bbox.height * h)

        # Add padding (IMPORTANT FIX)
        pad = 40
        x = max(0, x - pad)
        y = max(0, y - pad)

        face = image_rgb[y:y + h_box + pad, x:x + w_box + pad]

        if face.size == 0:
            return None

        return face


# ==============================
# EXTRACT EMBEDDING (FINAL FIX)
# ==============================

def extract_face_embedding(image_rgb: np.ndarray):

    if app is None or cv2 is None:
        return None

    try:
        # 🔥 Step 1: Preprocess
        image_rgb = preprocess_image(image_rgb)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # 🔥 Step 2: Try direct detection
        faces = app.get(image_bgr)

        if faces:
            return faces[0].embedding.astype(float).tolist()

        # 🔥 Step 3: MediaPipe fallback
        face_crop = detect_face_mediapipe(image_rgb)

        if face_crop is None:
            return None

        # 🔥 Step 4: Resize cropped face
        face_crop = cv2.resize(face_crop, (640, 640))
        face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)

        faces = app.get(face_bgr)

        if not faces:
            return None

        return faces[0].embedding.astype(float).tolist()

    except Exception:
        return None
