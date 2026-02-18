import streamlit as st
from pages.helper import db_queries
from pages.helper.streamlit_helpers import require_login


def show_image(image_col, image_path):
    """
    Unified image loader:
    - Works for Supabase URLs
    - Works for local images
    """
    if image_path:
        # If Supabase URL
        if str(image_path).startswith("http"):
            image_col.image(image_path, width=200)
        else:
            try:
                image_col.image(image_path, width=200)
            except:
                image_col.warning("Couldn't load image")
    else:
        image_col.warning("Image not available")


def case_viewer(case):
    matched_with_details = None

    if case.matched_with:
        matched_with_details = db_queries.get_public_case_detail(case.matched_with)

    data_col, image_col, matched_with_col = st.columns(3)

    status = "Found" if case.status == "F" else "Not Found"

    data_col.write(f"Name: {case.name}")
    data_col.write(f"Age: {case.age}")
    data_col.write(f"Status: {status}")
    data_col.write(f"Last Seen: {case.last_seen}")
    data_col.write(f"Phone: {case.complainant_mobile}")

    # 🔥 Fixed image loader
    show_image(image_col, case.image_path)

    if matched_with_details:
        pub = matched_with_details[0]
        matched_with_col.write(f"Location: {pub.location}")
        matched_with_col.write(f"Submitted By: {pub.submitted_by}")
        matched_with_col.write(f"Mobile: {pub.mobile}")
        matched_with_col.write(f"Birth Marks: {pub.birth_marks}")

    st.write("---")


def public_case_viewer(case):
    data_col, image_col, _ = st.columns(3)

    status = "Found" if case.status == "F" else "Not Found"

    data_col.write(f"Status: {status}")
    data_col.write(f"Location: {case.location}")
    data_col.write(f"Mobile: {case.mobile}")
    data_col.write(f"Birth Marks: {case.birth_marks}")
    data_col.write(f"Submitted on: {case.submitted_on}")
    data_col.write(f"Submitted by: {case.submitted_by}")

    # 🔥 Fixed image loader
    show_image(image_col, case.image_path)

    st.write("---")


if "login_status" not in st.session_state:
    st.write("You don't have access to this page")

elif st.session_state["login_status"]:
    user = st.session_state.user

    st.title("View Submitted Cases")

    status_col, date_col = st.columns(2)
    status = status_col.selectbox(
        "Filter", options=["All", "Not Found", "Found", "Public Cases"]
    )
    date = date_col.date_input("Date")

    st.write("---")

    if status == "Public Cases":
        cases_data = db_queries.fetch_public_cases(False, status)
        for case in cases_data:
            public_case_viewer(case)
    else:
        cases_data = db_queries.fetch_registered_cases(user, status)
        for case in cases_data:
            case_viewer(case)

else:
    st.write("You don't have access to this page")