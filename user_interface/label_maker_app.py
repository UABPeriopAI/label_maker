"""
The `main` function in the `app.py` file sets up a Streamlit UI for a label maker application with
tabs for spreadsheet upload, multiple document upload, and evaluating labeling performance.

"""
import streamlit as st

from LabeLMaker.streamlit_interface import Categorize, Evaluate
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

    categorize = Categorize(ui)
    # If there are two categorization modes, let the user choose
    # Tabs for single vs. multiple file uploads
    tab1, tab2, tab3 = st.tabs(["Spreadsheet Upload", "Multiple Document Upload","Evaluate Classification Performance"])
    with tab1:
        categorize.handle_single_upload()
    with tab2:
        categorize.handle_multiple_upload()
    with tab3:
        evaluator = Evaluate(ui)
        evaluator.handle_evaluation()

if __name__ == "__main__":
    main()
