import streamlit as st
from functools import wraps
import os
import uuid
import cv2


# ---------------- AUTH ---------------- #

def require_login(func):
    """Decorator to require login for Streamlit pages."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if (
            "login_status" not in st.session_state
            or not st.session_state["login_status"]
        ):
            st.write("You don't have access to this page")
            return
        return func(*args, **kwargs)
    return wrapper


# ---------------- UI HELPERS ---------------- #

def show_success(message: str):
    st.success(message)


def show_error(message: str):
    st.error(message)


def show_warning(message: str):
    st.warning(message)


# ---------------- IMAGE CACHE (CRITICAL FIX) ---------------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "resources", "images")

os.makedirs(IMAGE_DIR, exist_ok=True)


def save_image(img_np):
    """
    Save numpy image safely in RGB.
    Fixes blue color bug permanently.
    Returns absolute path to saved file.
    """
    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(IMAGE_DIR, filename)

    # 🔥 FORCE BGR → RGB if needed
    if img_np is not None and len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    cv2.imwrite(path, img_np)
    return path


def show_image(img_np, caption=None):
    """
    Safe wrapper to display any numpy image in Streamlit.
    """
    path = save_image(img_np)
    st.image(path, width="stretch", caption=caption)
    return path
