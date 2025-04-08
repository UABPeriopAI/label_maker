"""
The `main` function in the `app.py` file sets up a Streamlit UI for a label maker application with
tabs for spreadsheet upload, multiple document upload, and evaluating labeling performance.

"""
import streamlit as st

from LabeLMaker.streamlit_interface import (
    CategorizeHandler,
    EvaluateHandler,
)
from LabeLMaker.utils.page_renderer import UIHelper
from LabeLMaker_config.config import Config


def main():
    """
    The `main()` function sets up a Label maker application with tabs for spreadsheet upload, multiple
    document upload, and evaluating labeling performance.

    """
    st.set_page_config(page_title="Label maker", page_icon="🏷️")
    st.title("🏷️ Label maker 🤖")
    st.markdown(Config.HEADER_MARKDOWN)
    ui = UIHelper()
    cat_app = CategorizeHandler(ui, azure_key=Config.AZURE_DOCAI_COMPATIBLE_KEY)
    eval_app = EvaluateHandler(ui)
    # Tabs for single vs. multiple file uploads
    tab1, tab2, tab3 = st.tabs(
        ["Spreadsheet upload", "Multiple document upload", "Evaluate labeling performance"]
    )
    with tab1:
        cat_app.handle_single_upload()
    with tab2:
        cat_app.handle_multiple_upload()
    with tab3:
        eval_app.handle_evaluation()


if __name__ == "__main__":
    main()
