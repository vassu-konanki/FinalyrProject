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
# CENTRAL DATABASE PATH
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "cases.db")


# ==============================
# LOAD FACE MODEL (INSIGHTFACE)
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


def enhance_image(image_rgb):
    """
    Improve detection by resizing & sharpening
    """
    if cv2 is None:
        return image_rgb

    # Resize (important for small faces)
    image_rgb = cv2.resize(image_rgb, (640, 640))

    # Sharpening filter
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])

    image_rgb = cv2.filter2D(image_rgb, -1, kernel)

    return image_rgb


# ==============================
# MEDIAPIPE FALLBACK DETECTION
# ==============================

def detect_face_mediapipe(image_rgb):
    if mp is None:
        return None

    mp_face = mp.solutions.face_detection

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)

        if not results.detections:
            return None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        h, w, _ = image_rgb.shape

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        # Safety bounds
        x, y = max(0, x), max(0, y)

        face = image_rgb[y:y+height, x:x+width]

        return face


# ==============================
# EXTRACT FACE EMBEDDING
# ==============================

def extract_face_embedding(image_rgb: np.ndarray):

    if app is None or cv2 is None:
        return None

    try:
        # 🔥 Step 1: Enhance image
        image_rgb = enhance_image(image_rgb)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # 🔥 Step 2: Try InsightFace directly
        faces = app.get(image_bgr)

        if faces:
            embedding = faces[0].embedding
            return embedding.astype(float).tolist()

        # 🔥 Step 3: Fallback → MediaPipe detection
        face_crop = detect_face_mediapipe(image_rgb)

        if face_crop is None:
            return None

        # 🔥 Step 4: Retry InsightFace on cropped face
        face_crop = cv2.resize(face_crop, (640, 640))
        face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)

        faces = app.get(face_bgr)

        if not faces:
            return None

        embedding = faces[0].embedding
        return embedding.astype(float).tolist()

    except Exception:
        return None
