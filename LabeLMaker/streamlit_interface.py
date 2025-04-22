"""
Streamlit Interface

This module connects the Streamlit UI components with the underlying evaluation
and categorization workflows. It defines a BaseHandler with common UI helper functions,
as well as an Evaluate class and a Categorize class that wrap the corresponding
logic from evaluate_handler and categorizer_handler.
"""

import io
from pathlib import Path
from typing import Any, Tuple
import pandas as pd
import streamlit as st
# --- External/Internal imports ---
from aiweb_common.file_operations.upload_manager import StreamlitUploadManager
from LabeLMaker_config.config import Config
from LabeLMaker.utils.page_renderer import UIHelper
from LabeLMaker.utils.category import CategoryManager
from LabeLMaker.Evaluate.data_loader import DataLoader
from LabeLMaker.categorize_handler import StreamlitCategorizeHandler
from LabeLMaker.evaluate_handler import StreamlitEvaluateHandler
from LabeLMaker.utils.file_manager import FileManager


class BaseHandler:
    """
    Provides common helper functionality for both evaluation and categorization workflows.
    """
    def __init__(self, ui_helper: Any) -> None:
        """
        Initialize with a UI helper instance.
        
        Args:
            ui_helper: An object providing Streamlit helper methods (e.g. file_uploader, write, etc.).
        """
        self.ui = ui_helper

    def _load_data(self, uploaded_file) -> pd.DataFrame:
        """
        Loads CSV data by first converting the uploaded file to a DataFrame and then
        initializing the DataLoader with that DataFrame.
        """
        try:
            # Convert the file-like object into a DataFrame directly
            df = pd.read_csv(uploaded_file)
            # Initialize DataLoader with the DataFrame.
            loader = DataLoader(dataframe=df)
            return loader.df
        except Exception as e:
            raise Exception(f"Error processing CSV file: {e}")

    def _ensure_file(
        self,
        file: Any,
        upload_message: str,
        file_types: list,
        key: str,
        info_message: str,
        accept_multiple_files: bool = False,
    ) -> Any:
        """
        Ensure file(s) are uploaded. If not, prompt the user.
        
        Args:
            file: The current file variable (can be None).
            upload_message: Message shown for the uploader.
            file_types: Allowed file types.
            key: A unique key for the uploader.
            info_message: Message shown if no file is provided.
            accept_multiple_files: Whether to accept multiples.
        
        Returns:
            The file(s) uploaded or None if not provided.
        """
        if file is None:
            file = self.ui.file_uploader(
                label=upload_message,
                type=file_types,
                accept_multiple_files=accept_multiple_files,
                key=key,
            )
            if not file:
                self.ui.info(info_message)
        return file

    def generate_docx_report_download(self, doc: Any) -> bytes:
        """
        Convert a DOCX document into bytes for download.
        
        Args:
            doc: A DOCX document object.
        
        Returns:
            Byte representation of the DOCX document.
        """
        with io.BytesIO() as temp_stream:
            doc.save(temp_stream)
            temp_stream.seek(0)
            return temp_stream.read()


# --- Evaluation Handler for Streamlit UI ---
class Evaluate(BaseHandler):
    """
    Wraps the evaluation workflow.
    """
    def __init__(self, ui_helper: Any) -> None:
        """
        Initialize the evaluation handler.
        
        Args:
            ui_helper: An object providing Streamlit UI methods.
        """
        super().__init__(ui_helper)
        self.eval_handler = StreamlitEvaluateHandler(ui_helper)

    def handle_evaluation(self):
        self.eval_handler.handle_evaluation()

# --- Categorization Handler for Streamlit UI ---
class Categorize(BaseHandler):
    """
    Wraps the categorization workflow.
    """
    def __init__(self, ui_helper: Any, azure_key: str = None) -> None:
        """
        Initialize the categorization handler.
        
        Args:
            ui_helper: An object providing Streamlit UI methods.
            azure_key: Optional Azure key if needed.
        """
        super().__init__(ui_helper)
        self.fm = FileManager(azure_key)  # Used for any file operations
        self.config = Config
        self.cat_handler = StreamlitCategorizeHandler(azure_key=azure_key)

    def setup_workflow(self, df: pd.DataFrame) -> dict:
        """
        Gather workflow parameters from the UI.
        Includes unique ID setup, mode selection, text column selection, etc.
        
        Args:
            df: The input DataFrame.
        
        Returns:
            A dictionary of workflow parameters.
        """
        params = {}
        # Unique Identifier Setup
        if not self.ui.session_state.get("uniqueIdSetup_done"):
            self.ui.markdown("### Unique Identifier Setup")
            id_choice = self.ui.radio(
                "How would you like to specify a unique identifier?",
                options=["Create new ID column", "Use an existing column"],
                index=0,
                key="id_choice",
            )
            if id_choice == "Create new ID column":
                new_id_col = self.ui.text_input("Enter name for the new ID column", value="id", key="new_id_column")
                if self.ui.button("Create ID Column", key="create_id_column"):
                    if new_id_col not in df.columns:
                        df[new_id_col] = df.index.astype(str)
                        self.ui.success(f"New ID column '{new_id_col}' created.")
                    else:
                        self.ui.info(f"Column '{new_id_col}' already exists; using it.")
                    self.ui.session_state["single_file_df"] = df
                    self.ui.session_state["selected_id_column"] = new_id_col
                    self.ui.session_state["uniqueIdSetup_done"] = True
            else:
                selected_existing = self.ui.selectbox(
                    "Select the column to use as the unique identifier",
                    options=df.columns.tolist(),
                    key="existing_id_column",
                )
                if self.ui.button("Confirm ID Column", key="confirm_id_column"):
                    self.ui.session_state["selected_id_column"] = selected_existing
                    self.ui.session_state["uniqueIdSetup_done"] = True
        sel_id = self.ui.session_state.get("selected_id_column")
        if sel_id and sel_id not in df.columns:
            df[sel_id] = df.index.astype(str)
            self.ui.session_state["single_file_df"] = df
        params["index_column"] = sel_id
        self.ui.write(f"Using '{params['index_column']}' as the unique identifier column.")

        advanced_mode = self.ui.checkbox(
            "Advanced Mode",
            help="For users with ground truth labels. Check to run evaluation pipeline.",
            key="advanced_mode",
        )
        params["mode"] = "Evaluation" if advanced_mode else "Production"
        self.ui.info("Advanced Mode activated." if advanced_mode else "Software will automatically select labelling method")

        df_columns = df.columns.tolist()
        params["categorizing_column"] = self.ui.selectbox(
            "Select the column with text data you want to label",
            options=df_columns, key="categorizing_column"
        )
        if params["mode"] == "Evaluation":
            gt_col = self.ui.selectbox(
                "Select the column with ground truth labels",
                options=df_columns, key="ground_truth_column"
            )
            params["ground_truth_column"] = gt_col
            eval_techniques = self.ui.multiselect(
                "Select the evaluation approaches to run:",
                options=["Zero Shot", "Few Shot", "Many Shot"],
                default=["Zero Shot"],
                key="evaluation_techniques",
            )
            params["evaluation_techniques"] = eval_techniques
            if "Few Shot" in eval_techniques:
                few_shot_count = self.ui.number_input(
                    "Enter maximum examples per category (Few Shot)",
                    min_value=1, value=2, key="few_shot_count"
                )
                params["few_shot_count"] = few_shot_count
            if "Many Shot" in eval_techniques:
                many_shot_train_ratio = self.ui.number_input(
                    "Enter train proportion for Many Shot (0 to 1)",
                    min_value=0.0, max_value=1.0, value=0.8, key="many_shot_train_ratio"
                )
                params["many_shot_train_ratio"] = many_shot_train_ratio
        else:
            ex_options = ["None"] + df_columns
            ex_col = self.ui.selectbox(
                "Select the column containing your examples (if any)",
                options=ex_options, key="examples_column"
            )
            params["examples_column"] = None if ex_col == "None" else ex_col

        # Category definition using CategoryManager.
        
        default_col = params.get("ground_truth_column") if params["mode"] == "Evaluation" else (params.get("examples_column") or "")
        default_categories = ""
        if default_col and default_col in df.columns:
            unique_values = df[default_col].dropna().unique()
            if len(unique_values) <= Config.MAX_RECOMMENDED_GROUPS:
                default_categories = ",".join([str(val) for val in unique_values])
            else:
                self.ui.warning(
                    "There are more than 10 unique values in the column. Auto-population of categories may not be practical."
                )
        categories_dict = CategoryManager.define_categories(self.ui, "tab1", unique_values_str=default_categories)
        params["categories_dict"] = categories_dict
        return params

    def categorize_with_streamlit(self, df: pd.DataFrame, params: dict, zs_prompty: Path, fs_prompty: Path) -> pd.DataFrame:
        """
        Delegate categorization to the underlying StreamlitCategorizeHandler.
        
        Args:
            df: The input DataFrame.
            params: Dictionary of workflow parameters.
            zs_prompty: Path to the Zero Shot prompty file.
            fs_prompty: Path to the Few Shot prompty file.
        
        Returns:
            DataFrame with categorization results.
        """
        return self.cat_handler.streamlit_categorize(df, params, zs_prompty, fs_prompty)

    def process_multiple_files(self, uploaded_files: Any) -> Tuple[list, list]:
        """
        Process multiple file uploads using document analysis.
        
        Args:
            uploaded_files: List of uploaded file objects.
            
        Returns:
            A tuple of (filenames, texts) extracted from the files.
        """
        filenames = []
        document_analysis_client = Config.DOCUMENT_ANALYSIS_CLIENT if hasattr(Config, "AZURE_DOCAI_KEY") else None
        for file in uploaded_files:
            filenames.append(file.name)

        upload_manager = StreamlitUploadManager(uploaded_files, accept_multiple_files=True, document_analysis_client=document_analysis_client)
        self.ui.spinner('Reading in Files...')
        file_data, _ = upload_manager.process_upload()

        return filenames, file_data

    def display_results(self, results: Any) -> None:
        """
        Display categorization results.
        
        Args:
            results: The categorization results to display.
        """
        self.ui.write("Categorization Results:")
        self.ui.write(results)

    def handle_single_upload(self, zs_prompty: Path = Path(Config.ZS_PROMPTY), fs_prompty: Path = Path(Config.FS_PROMPTY)) -> None:
        """
        Handle a single file upload for categorization.
        
        The document is processed only once and cached in session state to prevent repeated reads on refresh.
        
        Args:
            zs_prompty: Path to the Zero Shot prompty file.
            fs_prompty: Path to the Few Shot prompty file.
        """
        file = self._ensure_file(
            file=None,
            upload_message="Choose a CSV file for evaluation",
            file_types=["csv"],
            key="cat_file_uploader",  
            info_message="Please upload a CSV file to proceed.",
        )
        if file is None:
            return

        # Process and cache the file only if not already cached.
        if "single_file_df" not in self.ui.session_state:
            try:
                document_analysis_client = Config.DOCUMENT_ANALYSIS_CLIENT if hasattr(Config, "AZURE_DOCAI_KEY") else None
                upload_manager = StreamlitUploadManager(file, accept_multiple_files=False, document_analysis_client=document_analysis_client)
                file_data, _ = upload_manager.process_upload()
                self.ui.session_state["single_file_df"] = file_data
                self.ui.session_state["uploaded_file_single"] = file
            except Exception as e:
                self.ui.error(f"Error processing the uploaded file: {e}")
                return
        df = self.ui.session_state["single_file_df"]
        self.ui.write(f"Uploaded file: {file.name}")
        ui_params = self.setup_workflow(df)
        if self.ui.button("Categorize", key="tab1_submit"):
            merged_df = self.categorize_with_streamlit(df, ui_params, zs_prompty, fs_prompty)
            csv_data = merged_df.to_csv(index=False).encode("utf-8")
            self.ui.download_button(
                label="Download Results",
                data=csv_data,
                file_name="AI_Generated_Categorization.csv",
                mime="text/csv",
            )

    def handle_multiple_upload(self, zs_prompty: Path = Path(Config.ZS_PROMPTY)) -> None:
        """
        Handle multiple file uploads for categorization.
        
        The processed file data is cached in session state so the files are not re-read on every refresh.
        
        Args:
            zs_prompty: Path to the Zero Shot prompty file.
        """
        if self.ui.button("Clear All", key="multi_clear_all"):
            for key in ("uploaded_files_multiple", "processed_files_multiple"):
                if key in self.ui.session_state:
                    del self.ui.session_state[key]
            self.ui.rerun()

        files = self._ensure_file(
            file=None,
            upload_message="Upload your DOCX or PDF files",
            file_types=["docx", "pdf"],
            key="multiple_file_uploader",
            info_message="Please upload DOCX or PDF files to proceed.",
            accept_multiple_files=True,
        )
        if files:
            # Cache processed files only once.
            if "processed_files_multiple" not in self.ui.session_state:
                self.ui.session_state["uploaded_files_multiple"] = files
                file_names = [file.name for file in files] if isinstance(files, list) else [files.name]
                self.ui.write(f"Uploaded files: {file_names}")
                with self.ui.spinner("Reading files super fast... please be patient"):
                    filenames, texts = self.process_multiple_files(files)
                self.ui.session_state["processed_files_multiple"] = (filenames, texts)

        if "processed_files_multiple" in self.ui.session_state:
            filenames, texts = self.ui.session_state["processed_files_multiple"]

            st.warning("We focus on using only the zero-shot modeling approach.")

            categories_dict = CategoryManager.define_categories(self.ui, "tab2")
            self.ui.write("Texts (excerpts) to label:")
            for text in texts[:5]:
                self.ui.write(text[:250])
            if self.ui.button("Categorize Multiple Files", key="tab2_submit"):
                from LabeLMaker.multifile_categorize_handler import MultiFileStreamlitCategorizeHandler
                handler = MultiFileStreamlitCategorizeHandler(azure_key=Config.AZURE_DOCAI_KEY)

                json_data = handler.multifile_categorization(filenames=filenames, texts=texts, categories_dict=categories_dict, zs_prompty=zs_prompty)

                st.download_button(
                    label="Download Results",
                    data=json_data,
                    file_name="AI_Generated_Categorization_multi.json",
                    mime="application/json"
                )


# For testing the UI independently:
if __name__ == "__main__":
    st.title("Evaluate and Categorize Handler Test")
    ui = UIHelper()  # Assuming UIHelper is implemented to wrap st.* calls.
    workflow_option = st.sidebar.radio("Choose Workflow", ("Evaluation", "Categorization"))
    if workflow_option == "Evaluation":
        eval_ui = StreamlitEvaluateHandler(ui)
        eval_ui.handle_evaluation()
    else:
        tab_option = st.sidebar.radio("Choose Categorization", ("Single File", "Multiple Files"))
        cat_handler = Categorize(ui)
        if tab_option == "Single File":
            cat_handler.handle_single_upload()
        else:
            cat_handler.handle_multiple_upload()