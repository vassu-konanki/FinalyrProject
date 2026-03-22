import streamlit as st

st.set_page_config(page_title="Register New Case")

import json

from pages.helper.data_models import RegisteredCases
from pages.helper import db_queries
from pages.helper.utils import image_obj_to_numpy, extract_face_embedding
from pages.helper.supabase_storage import upload_image

# Ensure DB exists
db_queries.create_db()

# ---------------- LOGIN CHECK ---------------- #
if "login_status" not in st.session_state or not st.session_state["login_status"]:
    st.error("You don't have access to this page. Please login.")
    st.stop()

user = st.session_state.user

st.title("Register New Case")

image_col, form_col = st.columns(2)

# ---------------- SESSION STATE INIT ---------------- #
if "face_mesh" not in st.session_state:
    st.session_state.face_mesh = None

if "image_path" not in st.session_state:
    st.session_state.image_path = None

if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False


# ---------------- IMAGE UPLOAD ---------------- #
with image_col:
    image_obj = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        key="new_case_upload"
    )

    if image_obj:
        with st.spinner("🔍 Processing image..."):

            st.image(image_obj, width=250)

            # Convert image
            image_numpy = image_obj_to_numpy(image_obj)

            # Extract face embedding
            face_mesh = extract_face_embedding(image_numpy)

            if face_mesh:
                try:
                    # ✅ FIXED PART (IMPORTANT)
                    file_bytes = image_obj.read()   # ✔ correct way
                    filename = image_obj.name       # ✔ pass name separately

                    image_path = upload_image(file_bytes, filename)

                    if image_path:
                        # Save in session
                        st.session_state.face_mesh = face_mesh
                        st.session_state.image_path = image_path
                        st.session_state.image_uploaded = True

                        st.success("✅ Face detected & image uploaded successfully!")
                    else:
                        st.session_state.image_uploaded = False
                        st.error("❌ Image upload failed.")

                except Exception as e:
                    st.session_state.image_uploaded = False
                    st.error(f"❌ Upload error: {str(e)}")

            else:
                st.session_state.face_mesh = None
                st.session_state.image_uploaded = False
                st.error("❌ Face not detected. Please upload a clear front-facing image.")


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

        # ---------------- VALIDATIONS ---------------- #
        if not st.session_state.image_uploaded:
            st.error("❌ Please upload a valid face image.")
            st.stop()

        if not st.session_state.face_mesh:
            st.error("❌ Face data missing. Please re-upload image.")
            st.stop()

        if not st.session_state.image_path:
            st.error("❌ Image upload failed. Try again.")
            st.stop()

        # ---------------- CREATE OBJECT ---------------- #
        new_case_details = RegisteredCases(
            submitted_by=user,
            name=name,
            father_name=father_name,
            age=str(age),
            color=color,
            height=height,
            complainant_mobile=mobile_number,
            complainant_name=complainant_name,
            face_mesh=json.dumps(st.session_state.face_mesh),
            image_path=st.session_state.image_path,
            adhaar_card=adhaar_card,
            birth_marks=birthmarks,
            address=address,
            last_seen=last_seen,
            status="NF",
            matched_with="",
        )

        # ---------------- SAVE ---------------- #
        try:
            db_queries.register_new_case(new_case_details)

            # Reset session
            st.session_state.face_mesh = None
            st.session_state.image_path = None
            st.session_state.image_uploaded = False

            st.success("🎉 Case registered successfully!")

        except Exception as e:
            st.error(f"❌ Failed to save case: {str(e)}")
