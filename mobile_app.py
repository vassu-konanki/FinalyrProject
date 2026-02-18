import uuid
import json
import numpy as np
import streamlit as st

from pages.helper import db_queries
from pages.helper.data_models import PublicSubmissions
from pages.helper.utils import image_obj_to_numpy, extract_face_embedding
from pages.helper.supabase_storage import upload_image

st.set_page_config("Mobile UI", initial_sidebar_state="expanded")

# -------------------------------
# SIDEBAR MENU
# -------------------------------
st.sidebar.title("mobile app")

menu = st.sidebar.radio(
    "Navigation",
    ["Register New Case", "Registered Cases", "All Cases", "Help"]
)

# -------------------------------
# PAGE 1: REGISTERED CASES
# (Police uploaded, Not Found only)
# -------------------------------
if menu == "Registered Cases":
    st.title("Registered Missing Persons (Not Found)")

    cases = db_queries.get_all_cases()

    not_found_cases = [case for case in cases if case.status == "NF"]

    if not_found_cases:
        for case in not_found_cases:
            st.markdown("---")
            col1, col2 = st.columns([1, 2])

            with col1:
                if case.image_path and str(case.image_path).startswith("http"):
                    st.image(case.image_path, width=150)
                else:
                    st.warning("Image not available")

            with col2:
                st.write(f"**Name:** {case.name}")
                st.write(f"**Age:** {case.age}")
                st.write(f"**Status:** {case.status}")
                st.write(f"**Last Seen:** {case.last_seen}")
                st.write(f"**Phone:** {case.complainant_mobile}")
    else:
        st.info("No not-found registered cases.")

# -------------------------------
# PAGE 2: ALL CASES
# (Shows all police cases)
# -------------------------------
elif menu == "All Cases":
    st.title("Missing Persons")
    st.subheader("Registered Missing Persons")

    cases = db_queries.get_all_cases()

    if cases:
        for case in cases:
            st.markdown("---")
            col1, col2 = st.columns([1, 2])

            with col1:
                if case.image_path and str(case.image_path).startswith("http"):
                    st.image(case.image_path, width=150)
                else:
                    st.warning("Image not available")

            with col2:
                st.write(f"**Name:** {case.name}")
                st.write(f"**Age:** {case.age}")
                st.write(f"**Status:** {case.status}")
                st.write(f"**Last Seen:** {case.last_seen}")
                st.write(f"**Phone:** {case.complainant_mobile}")
    else:
        st.info("No registered cases yet.")

# -------------------------------
# PAGE 3: REGISTER NEW CASE
# (Public submission)
# -------------------------------
elif menu == "Register New Case":
    st.title("Make a Submission")

    image_col, form_col = st.columns(2)
    image_obj = None
    save_flag = 0
    image_numpy = None
    image_path = None
    face_mesh = None

    with image_col:
        image_obj = st.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png"], key="user_submission"
        )

        if image_obj:
            with st.spinner("Processing..."):
                st.image(image_obj, width=200)

                image_numpy = image_obj_to_numpy(image_obj)
                face_mesh = extract_face_embedding(image_numpy)

                file_bytes = image_obj.getvalue()
                image_path = upload_image(file_bytes, image_obj.name)

    if image_obj:
        with form_col.form(key="new_user_submission"):
            name = st.text_input("Your Name")
            mobile_number = st.text_input("Your Mobile Number")
            email = st.text_input("Your Email")
            address = st.text_input("Address / Location last seen")

            color = st.text_input("Color (Skin / Hair / Eye)")
            height = st.text_input("Height (in cm)")
            birth_marks = st.text_input("Birth Marks")

            submit_bt = st.form_submit_button("Submit")

            if submit_bt:
                public_submission_details = PublicSubmissions(
                    submitted_by=name,
                    mobile=mobile_number,
                    email=email,
                    location=address,
                    color=color,
                    height=height,
                    face_mesh=json.dumps(face_mesh),
                    image_path=image_path,
                    birth_marks=birth_marks,
                    status="NF",
                )

                db_queries.new_public_case(public_submission_details)
                save_flag = 1

        if save_flag == 1:
            st.success("Successfully Submitted")

# -------------------------------
# PAGE 4: HELP
# -------------------------------
elif menu == "Help":
    st.title("Help")
    st.write("""
    **How to use this app:**
    
    1. View missing persons in **Registered Cases**.
    2. If you see someone, go to **Register New Case**.
    3. Upload the photo and submit details.
    4. Police will verify the match.
    """)