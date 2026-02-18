import yaml
import base64
import streamlit as st
from yaml import SafeLoader
import streamlit_authenticator as stauth

from pages.helper import db_queries
from pages.helper.db_queries import create_db

# ---------------- CREATE TABLES ---------------- #
create_db()

# ---------------- UI HELPERS ---------------- #

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string.decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------- SESSION INIT ---------------- #

if "login_status" not in st.session_state:
    st.session_state["login_status"] = False

# ---------------- LOAD CONFIG ---------------- #

try:
    with open("login_config.yml") as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("Configuration file 'login_config.yml' not found")
    st.stop()

# ---------------- AUTH SETUP ---------------- #

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

authenticator.login(location="main")

# ---------------- SAFE AUTH HANDLING ---------------- #

auth_status = st.session_state.get("authentication_status")
username = st.session_state.get("username")

if auth_status is True:

    if not username:
        st.error("Session expired. Please login again.")
        st.stop()

    user_info = config["credentials"]["usernames"].get(username)

    if not user_info:
        st.error("User not found in login_config.yml")
        st.stop()

    authenticator.logout("Logout", "sidebar")

    st.session_state["login_status"] = True
    st.session_state["user"] = user_info.get("name", username)

    # ---------------- DASHBOARD HEADER ---------------- #

    name = user_info.get("name", "N/A")
    area = user_info.get("area", "N/A")
    city = user_info.get("city", "N/A")
    role = user_info.get("role", "N/A")

    st.write(
        f'<p style="color:grey; text-align:left; font-size:45px">{name}</p>',
        unsafe_allow_html=True,
    )

    st.write(
        f'<p style="color:grey; text-align:left; font-size:20px">{area}, {city}</p>',
        unsafe_allow_html=True,
    )

    st.write(
        f'<p style="color:grey; text-align:left; font-size:20px">{role}</p>',
        unsafe_allow_html=True,
    )

    st.write("---")

    # ---------------- METRICS ---------------- #

    found_cases = db_queries.get_registered_cases_count(name, "F")
    non_found_cases = db_queries.get_registered_cases_count(name, "NF")

    found_col, not_found_col = st.columns(2)
    found_col.metric("Found Cases Count", value=len(found_cases))
    not_found_col.metric("Not Found Cases Count", value=len(non_found_cases))

    st.write("---")

    # ---------------- DISPLAY ALL CASES ---------------- #

    st.title("All Registered Cases")

    cases = db_queries.get_all_cases()

    if cases:
        for case in cases:
            st.markdown("---")
            col1, col2 = st.columns([1, 2])

            with col1:
                if case.image_path and case.image_path.startswith("http"):
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
        st.info("No registered cases found.")

# ---------------- AUTH FAIL ---------------- #

elif auth_status is False:
    st.error("Username or password is incorrect")

else:
    st.warning("Please enter your username and password")
    st.session_state["login_status"] = False