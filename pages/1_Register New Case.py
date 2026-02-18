import uuid
import numpy as np
import streamlit as st
import json

from pages.helper.data_models import RegisteredCases
from pages.helper import db_queries
from pages.helper.utils import image_obj_to_numpy, extract_face_embedding
from pages.helper.supabase_storage import upload_image

st.set_page_config(page_title="Register New Case")

# Ensure DB exists
db_queries.create_db()

# ---------------- LOGIN CHECK ---------------- #
if "login_status" not in st.session_state or not st.session_state["login_status"]:
    st.error("You don't have access to this page. Please login.")
    st.stop()

user = st.session_state.user

st.title("Register New Case")

image_col, form_col = st.columns(2)

image_obj = None
image_numpy = None
image_path = None
face_mesh = None
save_flag = False

# ---------------- IMAGE UPLOAD ---------------- #
with image_col:
    image_obj = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        key="new_case_upload"
    )

    if image_obj:
        with st.spinner("Processing image..."):
            st.image(image_obj, width=250)

            image_numpy = image_obj_to_numpy(image_obj)
            face_mesh = extract_face_embedding(image_numpy)

            if face_mesh:
                # Upload image to Supabase
                file_bytes = image_obj.getvalue()
                image_path = upload_image(file_bytes, image_obj.name)
                st.success("Image uploaded successfully")
            else:
                st.error("Face not detected. Please upload a clear face image.")

# ---------------- FORM ---------------- #
with form_col.form(key="new_case_form"):
    name = st.text_input("Name")
    father_name = st.text_input("Father's Name")
    age = st.number_input("Age", min_value=3, max_value=100, value=10, step=1)

    color = st.text_input("Color (Skin / Hair / Eye)")
    height = st.text_input("Height (in cm)")

    mobile_number = st.text_input("Mobile Number")
    address = st.text_input("Address")
    adhaar_card = st.text_input("Aadhaar Card")
    birthmarks = st.text_input("Birth Mark")
    last_seen = st.text_input("Last Seen")

    complainant_name = st.text_input("Complainant Name")
    complainant_phone = st.text_input("Complainant Phone")

    submit_bt = st.form_submit_button("Save Case")

    if submit_bt:
        if not image_obj:
            st.error("Please upload an image.")
        elif not face_mesh:
            st.error("Face not detected. Cannot save case.")
        elif not image_path:
            st.error("Image upload failed.")
        else:
            new_case_details = RegisteredCases(
                submitted_by=user,
                name=name,
                father_name=father_name,
                age=str(age),
                color=color,
                height=height,
                complainant_mobile=mobile_number,
                complainant_name=complainant_name,
                face_mesh=json.dumps(face_mesh),
                image_path=image_path,  # Supabase public URL
                adhaar_card=adhaar_card,
                birth_marks=birthmarks,
                address=address,
                last_seen=last_seen,
                status="NF",
                matched_with="",
            )

            db_queries.register_new_case(new_case_details)
            save_flag = True

# ---------------- SUCCESS MESSAGE ---------------- #
if save_flag:
    st.success("Case registered successfully!")