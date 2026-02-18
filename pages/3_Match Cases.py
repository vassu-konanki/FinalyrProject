import streamlit as st
from pages.helper import db_queries, match_algo, train_model
from pages.helper.streamlit_helpers import require_login


# ---------------- IMAGE LOADER ----------------
def show_image(image_path, width=300):
    """
    Loads image from:
    - Supabase URL
    - Local path (fallback)
    """
    if image_path:
        if str(image_path).startswith("http"):
            st.image(image_path, width=width)
        else:
            try:
                st.image(image_path, width=width)
            except:
                st.warning("Couldn't load image")
    else:
        st.warning("Image not available")


# ---------------- CASE VIEWER ----------------
@require_login
def case_viewer(registered_case_id, public_case_id):
    try:
        case_details = db_queries.get_registered_case_detail(registered_case_id)[0]

        data_col, image_col = st.columns(2)

        for text, value in zip(
            ["Name", "Mobile", "Age", "Last Seen", "Birth marks"],
            case_details[:-1],
        ):
            data_col.write(f"{text}: {value}")

        image_path = case_details[-1]

        # Update status
        db_queries.update_found_status(registered_case_id, public_case_id)
        st.success(
            "Status Changed. Next time it will be only visible in confirmed cases"
        )

        # 🔥 FIXED IMAGE LOADING
        with image_col:
            show_image(image_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        st.error(f"Something went wrong: {str(e)}")


# ---------------- MAIN PAGE ----------------
if "login_status" not in st.session_state:
    st.write("You don't have access to this page")

elif st.session_state["login_status"]:
    user = st.session_state.user

    st.title("Check for match")

    col1, col2 = st.columns(2)
    refresh_bt = col1.button("Refresh")

    st.write("---")

    if refresh_bt:
        with st.spinner("Fetching Data and Training Model..."):
            train_model.train(user)
            matched_ids = match_algo.match()

            if matched_ids["status"]:
                if not matched_ids["result"]:
                    st.info("No match found")
                else:
                    for matched_id, submitted_case_id in matched_ids["result"].items():
                        case_viewer(matched_id, submitted_case_id[0])
                        st.write("---")
            else:
                st.info("No match found")

else:
    st.write("You don't have access to this page")