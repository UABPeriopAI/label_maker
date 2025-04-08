import io
import random
from pathlib import Path

import pandas as pd
from aiweb_common.file_operations.docx_creator import StreamlitDocxCreator
from aiweb_common.file_operations.upload_manager import (
    StreamlitUploadManager,
    create_document_analysis_client,
)

# Imports used exclusively by CategorizeHandler.
from app.v01.schemas import Example
from LabeLMaker.Categorize.fewshot import FewShotCategorizer
from LabeLMaker.Categorize.manyshot import ManyshotClassifier
from LabeLMaker.Categorize.zeroshot import ZeroShotCategorizer
from LabeLMaker.Evaluate.confidence_intervals import (
    compute_bootstrap_confidence_intervals,
)

# Imports used exclusively by EvaluateHandler.
from LabeLMaker.Evaluate.data_loader import DataLoader
from LabeLMaker.Evaluate.evaluator import Evaluator
from LabeLMaker.utils.category import CategoryManager
from LabeLMaker.utils.class_balance import ClassBalance
from LabeLMaker.utils.file_manager import FileManager
from LabeLMaker_config.config import Config


class BaseHandler:
    """
    Provides common helper functionality used in evaluation and categorization tasks,
    such as file upload and download functions.
    """

    def __init__(self, ui_helper):
        """
        The function initializes an object with a UI helper wrapper provided as a parameter.

        Args:
          ui_helper: The `ui_helper` parameter in the `__init__` method is expected to be an object that
        represents a UI helper wrapper, such as a Streamlit UI object. This parameter is used to interact
        with the user interface within the class or object where the `__init__` method is defined
        """
        self.ui = ui_helper  # The UI helper wrapper (e.g., Streamlit UI)

    def _ensure_file(
        self,
        file,
        upload_message,
        file_types,
        key,
        info_message,
        accept_multiple_files: bool = False,
    ):
        """
        The function `_ensure_file` checks if a file is provided, and if not, prompts the user to upload
        a file or displays an info message.

        Args:
          file: The `file` parameter in the `_ensure_file` method is used to represent the file that
        needs to be ensured or uploaded. If the `file` is `None`, the method will call the UI uploader
        to upload a file.
          upload_message: `upload_message` is a message that will be displayed to prompt the user to
        upload a file. It serves as an instruction or guidance for the user on what action to take.
          file_types: The `file_types` parameter in the `_ensure_file` method is used to specify the
        types of files that can be uploaded. It is typically a list of strings representing the allowed
        file extensions or MIME types. For example, `file_types=['.txt', '.csv', 'image/jpeg']`
          key: The `key` parameter in the `_ensure_file` method is used as a unique identifier for the
        file uploader component. It helps in associating the uploaded file with a specific key, which
        can be useful for tracking or handling the file data within the application.
          info_message: The `info_message` parameter in the `_ensure_file` method is a message that will
        be displayed if the `file` is still `None` after attempting to upload a file using the UI
        uploader. It serves as an informational message to the user indicating that no file was
        uploaded.
          accept_multiple_files (bool): The `accept_multiple_files` parameter in the `_ensure_file`
        method is a boolean flag that determines whether the file uploader should allow the user to
        upload multiple files at once. If set to `True`, the file uploader will enable the user to
        select and upload multiple files simultaneously. If set to `. Defaults to False

        Returns:
          the `file` variable, which may have been updated based on the conditions inside the function.
        If `file` is None initially and remains None after attempting to upload a file, an info message
        is displayed and None is returned.
        """

        if file is None:
            file = self.ui.file_uploader(
                label=upload_message,
                type=file_types,
                accept_multiple_files=accept_multiple_files,
                key=key,
            )
            if file is None:
                self.ui.info(info_message)
        return file

    def generate_docx_report_download(self, doc):
        """
        The function generates a DOCX report and converts it into bytes for file download.

        Args:
          doc: The `doc` parameter in the `generate_docx_report_download` function is expected to be a
        DOCX Document object that you want to convert into bytes for file download.

        Returns:
          The function `generate_docx_report_download` returns the DOCX document converted into bytes to
        allow file download.
        """
        """
        Convert the DOCX Document into bytes to allow file download.
        """
        temp_stream = io.BytesIO()
        doc.save(temp_stream)
        temp_stream.seek(0)
        return temp_stream.read()


class EvaluateHandler(BaseHandler):
    """
    Handles evaluation workflows.
    Includes CSV file upload & preview, evaluation setup,
    single/multi‐column evaluation including method comparisons,
    and DOCX report creation.
    """

    def __init__(self, ui_helper):
        """
        The above function is a constructor that initializes an object with a UI helper.

        Args:
          ui_helper: The `ui_helper` parameter in the `__init__` method is typically used to pass in an
        object that provides utility functions or methods related to the user interface. This can
        include functions for displaying information, handling user input, managing UI components, and
        more. By passing in a `ui_helper
        """
        super().__init__(ui_helper)

    # --------------------------
    # File Upload & Data Loader Methods
    def load_data(self, file):
        """
        The function `load_data` creates a `DataLoader` object with the specified file and returns it.

        Args:
          file: The `file` parameter in the `load_data` method is the file path or name that you want to
        load data from. This parameter is used to specify the location of the file that the `DataLoader`
        will load data from.

        Returns:
          An instance of the DataLoader class with the specified file.
        """
        loader = DataLoader(file=file)
        return loader

    def preview_data(self, loader):
        """
        The function `preview_data` displays a preview of the CSV data loaded by the `loader` object,
        including the first few rows and the total number of rows in the dataframe.

        Args:
          loader: The `loader` parameter in the `preview_data` method is likely an object that contains
        a DataFrame (`df`) attribute. This DataFrame is used to display a preview of the data in the CSV
        file being loaded. The method displays the first few rows of the DataFrame and the total number
        of rows in
        """
        self.ui.subheader("CSV Preview:")
        self.ui.write(loader.df.head())
        self.ui.write(f"Total rows in dataframe: {len(loader.df)}")

    # --------------------------
    # Evaluation Setup Helpers
    def select_evaluation_column(self, loader):
        """
        The function `select_evaluation_column` prompts the user to select a column from a DataFrame for
        use as the ground truth column.

        Args:
          loader: The `loader` parameter in the `select_evaluation_column` method seems to be an object
        that has a property `df` which is a DataFrame. The method allows the user to select a column
        from the DataFrame `loader.df` as the ground truth column. The user interface (UI) component
        used

        Returns:
          the column selected as the ground truth column from the DataFrame loaded by the loader.
        """
        ground_truth_col = self.ui.selectbox(
            "Select Ground Truth Column", loader.df.columns, key="eval_gt_column"
        )
        return ground_truth_col

    def check_class_balance(self, loader, class_col, label):
        """
        The function `check_class_balance` computes the class balance for a specified column in a
        DataFrame and displays the results using a UI subheader and table.

        Args:
          loader: The `loader` parameter is likely an object that helps load or manage data, such as a
        data loader object used in machine learning frameworks like PyTorch or TensorFlow. It is being
        used to access the data frame (`loader.df`) containing the dataset.
          class_col: The `class_col` parameter in the `check_class_balance` function is used to specify
        the column in the dataset that contains the class labels or categories for which you want to
        check the balance. This column will be used by the `ClassBalance` to compute the balance of the
        classes in the dataset
          label: The `label` parameter is a string that represents the label or title of the class
        balance being computed. It is used to provide context or information about the specific class
        balance being displayed in the user interface.
        """
        balancer = ClassBalance(loader.df, class_col)
        balance_df = balancer.compute_balance()
        self.ui.subheader(f"Class Balance for {label} ({class_col})")
        self.ui.write(balance_df)

    # --------------------------
    # Evaluation & Visualization Helper (for single–column evaluation)
    def evaluate_model(self, df, pred_col, ground_truth_col, n_bootstraps=1000, alpha=0.05):
        """
        The `evaluate_model` function evaluates predictions by comparing them with ground truth,
        calculates metrics, generates a classification report, computes bootstrap confidence intervals,
        and plots a confusion matrix.

        Args:
          df: The `df` parameter in the `evaluate_model` function is a pandas DataFrame that contains
        the data for evaluation. It should include columns for predictions (`pred_col`) and ground truth
        (`ground_truth_col`) that will be used for evaluating the model's performance.
          pred_col: The `pred_col` parameter in the `evaluate_model` function refers to the column in
        the DataFrame `df` that contains the predicted values for the model. It is used to evaluate the
        predictions by comparing them with the ground truth values.
          ground_truth_col: The `ground_truth_col` parameter in the `evaluate_model` function refers to
        the column in the DataFrame `df` that contains the ground truth values for the predictions. This
        column is used to compare the predicted values with the actual ground truth values during the
        evaluation process.
          n_bootstraps: The `n_bootstraps` parameter in the `evaluate_model` function specifies the
        number of bootstrap samples to generate when computing bootstrap confidence intervals. It
        determines how many resamples of the data will be created to estimate the sampling distribution
        of a statistic. In this case, it is set to a default. Defaults to 1000
          alpha: The `alpha` parameter in the `evaluate_model` function represents the significance
        level used for computing Bootstrap Confidence Intervals. It is typically set to a value between
        0 and 1, such as 0.05, to determine the confidence level of the intervals. In this case, an `

        Returns:
          metrics_df, report_df, bs_df, cm_fig
        """
        """
        Evaluate predictions by comparing with ground truth. Drops rows with NaN predictions.
        Returns:
            metrics_df, report_df, bs_df, cm_fig
        """
        if pred_col in df.columns and ground_truth_col in df.columns:
            valid = df[pred_col].notna()
            y_true = df.loc[valid, ground_truth_col].astype(str).tolist()
            y_pred = df.loc[valid, pred_col].astype(str).tolist()
            evaluator = Evaluator(y_true, y_pred)
            evaluator.calculate_metrics()
            metrics_df = evaluator.display_metrics()
            report_df = pd.DataFrame(evaluator.metrics["Classification Report"]).transpose()
            # Compute Bootstrap Confidence Intervals.
            bs_results = compute_bootstrap_confidence_intervals(
                y_true, y_pred, n_bootstraps=n_bootstraps, alpha=alpha
            )
            bs_display = []
            for metric, values in bs_results.items():
                if values["Value"] is not None:
                    bs_display.append(
                        {
                            "Metric": metric,
                            "Value": f"{values['Value']:.4f}",
                            "Bootstrap Mean": f"{values['Bootstrap Mean']:.4f}",
                            "95% CI": f"({values['95% CI'][0]:.4f}, {values['95% CI'][1]:.4f})",
                        }
                    )
                else:
                    bs_display.append(
                        {
                            "Metric": metric,
                            "Value": "Undefined",
                            "Bootstrap Mean": "Undefined",
                            "95% CI": "Undefined",
                        }
                    )
            bs_df = pd.DataFrame(bs_display)
            cm_fig = evaluator.plot_confusion_matrix()
            return metrics_df, report_df, bs_df, cm_fig
        else:
            self.ui.error("Selected evaluation columns not found in the dataframe.")
            return None, None, None, None

    def compare_methods(self, df, ground_truth_col, selected_methods):
        """
        The `compare_methods` function compares selected prediction methods on a common data subset and
        returns evaluation results and confusion matrices for each method.

        Args:
          df: The `df` parameter in the `compare_methods` function is a pandas DataFrame containing the data
        for evaluation. It is used to compare selected prediction methods on a common data subset.
          ground_truth_col: The `ground_truth_col` parameter in the `compare_methods` function refers to the
        column in the DataFrame `df` that contains the ground truth values for the evaluation. This column
        should represent the true values that the selected prediction methods are compared against. It is
        typically the actual values that the models are
          selected_methods: The `selected_methods` parameter in the `compare_methods` function is a list of
        prediction methods that you want to compare. These methods should correspond to the columns in the
        DataFrame `df` that contain the predicted categories for each method. The function will evaluate
        these selected methods on a common subset of the

        Returns:
          The function `compare_methods` returns three values:
        1. `common_df`: Subset of the input dataframe `df` used for evaluation.
        2. `results`: A dictionary with each selected method's evaluation metrics.
        3. `confusion_matrices`: A dictionary with each selected method's confusion matrix figure.
        """
        method_columns = {method: f"Predicted Category ({method})" for method in selected_methods}
        valid_methods = {m: col for m, col in method_columns.items() if col in df.columns}

        if not valid_methods:
            self.ui.info("No selected method prediction columns exist in the data.")
            return None, {}, {}

        common_df = df.dropna(subset=list(valid_methods.values()))
        self.ui.write(f"Common evaluation subset size: {len(common_df)}")
        results = {}
        confusion_matrices = {}
        for method, col in valid_methods.items():
            y_true = common_df[ground_truth_col].astype(str).tolist()
            y_pred = common_df[col].astype(str).tolist()
            evaluator = Evaluator(y_true, y_pred)
            evaluator.calculate_metrics()
            results[method] = evaluator.metrics
            confusion_matrices[method] = evaluator.plot_confusion_matrix()
        return common_df, results, confusion_matrices

    # --------------------------
    # Main Evaluation Workflow
    def handle_evaluation(self, file=None):
        """
        The `handle_evaluation` function manages the evaluation workflow by uploading a CSV file, loading
        and previewing data, selecting evaluation methods, running evaluation, displaying results, and
        generating a downloadable DOCX report.

        Args:
          file: The `file` parameter in the `handle_evaluation` method is used to specify the CSV file that
        will be used for evaluation. If a file is not provided, the method will prompt the user to upload a
        CSV file. The method then proceeds to load and preview the data, select the ground truth

        Returns:
          The `handle_evaluation` method returns `None` if the `file` is not provided, indicating that the
        evaluation workflow cannot proceed without a CSV file.

        Handles the evaluation workflow by:
          - Uploading the CSV file (if not provided),
          - Loading and previewing the data,
          - Selecting the ground truth column and checking class balance,
          - Letting the user select one or more evaluation methods,
          - Running evaluation on demand,
          - Displaying the results, and
          - Generating a downloadable DOCX report.

        """

        # File upload (if not already provided)
        file = self._ensure_file(
            file,
            upload_message="Choose CSV file",
            file_types="csv",
            key="eval_file_uploader",
            info_message="Please upload a CSV file to proceed.",
        )
        if file is None:
            return

        loader = self.load_data(file)
        self.preview_data(loader)
        ground_truth_col = self.select_evaluation_column(loader)
        self.check_class_balance(loader, ground_truth_col, "Ground Truth")
        self.ui.header("Evaluation")
        selected_methods = self.ui.multiselect(
            "Select evaluation methods",
            ["Zero Shot", "Few Shot", "Many Shot"],
            default=["Zero Shot", "Few Shot", "Many Shot"],
            key="eval_methods",
        )

        if self.ui.button("Evaluate"):
            with self.ui.spinner("Evaluating..."):
                if not selected_methods:
                    self.ui.error("Please select at least one method for evaluation.")
                    return

                common_df, comparison_results, confusion_matrices = self.compare_methods(
                    loader.df, ground_truth_col, selected_methods
                )
                if comparison_results:
                    for method, metrics in comparison_results.items():
                        self.ui.subheader(method)
                        self.ui.write(metrics)

                    docx_maker = StreamlitDocxCreator(comparison_results, confusion_matrices)
                    doc = docx_maker.create_docx_report()
                    docx_content = self.generate_docx_report_download(doc)
                    self.ui.download_button(
                        label="Download DOCX Report",
                        data=docx_content,
                        file_name="evaluation_report.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                else:
                    self.ui.error("No evaluation metrics available for the selected methods.")
        else:
            self.ui.info("Click the Evaluate button to start evaluation.")


class CategorizeHandler(BaseHandler):
    """
    Handles file uploads and categorization workflows.
    Includes both single and multiple file upload, workflow setup and execution,
    and processing of categorization results using zero-shot, few-shot, or many-shot methods.
    """

    def __init__(self, ui_helper, azure_key=None):
        """
        The function initializes a class instance with a UI helper and an optional Azure key for file
        management and configuration.

        Args:
          ui_helper: The `ui_helper` parameter is an object that provides helper
        functions or utilities related to user interface operations. It is being passed to the constructor
        of the current class as an argument.
          azure_key: The `azure_key` parameter is an optional parameter that can be passed to the `__init__`
        method of a class. It is used to provide an Azure key that may be required for certain operations
        within the class. If a value is provided for `azure_key`, it will be used to
        """
        super().__init__(ui_helper)
        self.fm = FileManager(azure_key)
        self.config = Config

    def get_document_analysis_client(self):
        """
        This function returns a valid document analysis client based on the configuration provided.

        Returns:
          A valid document analysis client based on the configuration is being returned if the condition `if
        hasattr(self.config, "AZURE_DOCAI_KEY") and self.config.AZURE_DOCAI_KEY` is met. Otherwise, `None`
        is returned.
        """
        if hasattr(self.config, "AZURE_DOCAI_KEY") and self.config.AZURE_DOCAI_KEY:
            return create_document_analysis_client(self.config.AZURE_DOCAI_KEY)
        return None

    def _get_default_categories(self, df, col):
        """
        The function `_get_default_categories` retrieves unique values from a specified column in a
        DataFrame, with a warning if there are more than 10 unique values.

        Args:
          df: The `df` parameter in the `_get_default_categories` method is typically a pandas DataFrame
        that contains the data from which you want to extract unique values for a specific column (`col`).
        The method checks if the specified column exists in the DataFrame and then retrieves the unique
        values from that column to generate a
          col: The `col` parameter in the `_get_default_categories` method is used to specify the column
        name from the DataFrame (`df`) for which you want to retrieve default categories. The method checks
        if the column name is valid (not empty and exists in the DataFrame), then it retrieves unique
        non-null values

        Returns:
          An empty string is being returned if the conditions in the function are not met.
        """
        if col and col.strip() != "" and col in df.columns:
            unique_values = df[col].dropna().unique()
            if len(unique_values) > self.config.MAX_RECOMMENDED_GROUPS:
                self.ui.warning(
                    "There are more than 10 unique values in the column. "
                    "Auto-population of categories may not be practical."
                )
                return ""
            return ",".join([str(val) for val in unique_values])
        return ""

    def _prepare_ground_truth_examples(
        self, df, id_col, text_col, gt_col, few_shot_count=2, many_shot_train_ratio=0.8
    ):
        """
        The function `_prepare_ground_truth_examples` prepares few-shot and many-shot examples with
        corresponding IDs from a DataFrame based on specified parameters.

        Args:
          df: DataFrame containing the data
          id_col: The `id_col` parameter in the `_prepare_ground_truth_examples` function refers to the
        column in the DataFrame `df` that contains unique identifiers for each record. This column is used
        to identify and track individual records within the dataset.
          text_col: The `text_col` parameter in the `_prepare_ground_truth_examples` method refers to the
        column in the DataFrame `df` that contains the text data for each example. This column will be used
        to extract the text information for creating examples during the preparation of ground truth
        examples.
          gt_col: The `gt_col` parameter in the `_prepare_ground_truth_examples` method refers to the column
        in the DataFrame `df` that contains the ground truth labels for the examples. This column is used
        for grouping the examples based on their ground truth labels during the preparation process.
          few_shot_count: The `few_shot_count` parameter in the `_prepare_ground_truth_examples` function
        determines the number of examples that will be included in the few-shot learning set. It specifies
        how many randomly sampled examples will be selected for each unique label in the dataset to create a
        few-shot learning subset. Defaults to 2
          many_shot_train_ratio: The `many_shot_train_ratio` parameter in the
        `_prepare_ground_truth_examples` function determines the ratio of examples that will be used for
        training in the "many-shot" category. For example, if `many_shot_train_ratio=0.8`, it means that 80%
        of the examples for

        Returns:
          The function `_prepare_ground_truth_examples` returns four values: `few_shot_examples`,
        `few_shot_ids`, `many_shot_examples`, and `many_shot_test_ids`.
        """
        few_shot_examples = []
        few_shot_ids = set()
        many_shot_examples = []
        many_shot_test_ids = set()
        df_gt = df[[id_col, text_col, gt_col]].copy()
        df_gt[gt_col] = df_gt[gt_col].astype(str).str.lower()
        grouped = df_gt.groupby(gt_col)
        for label, group in grouped:
            records = group.to_dict(orient="records")
            count = min(few_shot_count, len(records))
            if count > 0:
                sampled = random.sample(records, count)
                for rec in sampled:
                    few_shot_examples.append(
                        Example(text_with_label=str(rec[text_col]), label=str(rec[gt_col]))
                    )
                    few_shot_ids.add(str(rec[id_col]))
            if len(records) > 1:
                shuffled = records.copy()
                random.shuffle(shuffled)
                train_size = max(1, int(many_shot_train_ratio * len(records)))
                train_examples = shuffled[:train_size]
                test_examples = shuffled[train_size:]
                for rec in train_examples:
                    many_shot_examples.append(
                        Example(text_with_label=str(rec[text_col]), label=str(rec[gt_col]))
                    )
                for rec in test_examples:
                    many_shot_test_ids.add(str(rec[id_col]))
        return few_shot_examples, few_shot_ids, many_shot_examples, many_shot_test_ids

    def setup_workflow(self, df):
        """
        The `setup_workflow` function in Python sets up parameters for a workflow, including selecting
        columns, creating unique identifiers, and choosing evaluation approaches.

        Args:
          df: The `setup_workflow` function is designed to set up parameters for a workflow based on user
        interactions with a UI. The function takes a DataFrame `df` as input and performs the following
        tasks:

        Returns:
          The `setup_workflow` function returns a dictionary `params` containing various parameters set
        based on user inputs and selections made during the workflow setup process. The function collects
        information such as the mode (Evaluation or Production), unique identifier column, categorizing
        column, ground truth column, evaluation techniques selected, and related parameters for Few Shot and
        Many Shot evaluation approaches. Additionally, it includes information about default categories and
        descriptions
        """
        params = {}
        advanced_mode = self.ui.checkbox(
            "Advanced Mode",
            help="For users with ground truth labels. Check to run evaluation pipeline.",
            key="advanced_mode",
        )
        if advanced_mode:
            self.ui.info(
                "Advanced Mode is activated: Users can select classification mode (e.g., zero shot, few shot, or many shot)"
            )
        else:
            self.ui.info(
                "Advanced Mode is not activated: Classification will be determined by number of examples provided"
            )
        params["mode"] = "Evaluation" if advanced_mode else "Production"

        if not self.ui.session_state.get("uniqueIdSetup_done"):
            self.ui.markdown("### Unique Identifier Setup")
            id_choice = self.ui.radio(
                "How would you like to specify a unique identifier?",
                options=["Create new ID column", "Use an existing column"],
                index=0,
                key="id_choice",
            )
            if id_choice == "Create new ID column":
                new_id_col = self.ui.text_input(
                    "Enter name for the new ID column", value="id", key="new_id_column"
                )
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
        params["index_column"] = self.ui.session_state.get("selected_id_column")
        self.ui.write(f"Using '{params['index_column']}' as the unique identifier column.")

        df_columns = df.columns.tolist()
        params["categorizing_column"] = self.ui.selectbox(
            "Select the column with text data you want to label",
            options=df_columns,
            key="categorizing_column",
        )
        if "ground_truth_column" not in self.ui.session_state:
            self.ui.session_state["ground_truth_column"] = df_columns[0]
        gt_col = self.ui.selectbox(
            "Select the column with ground truth labels",
            options=df_columns,
            key="ground_truth_column",
            index=df_columns.index(self.ui.session_state["ground_truth_column"]),
        )
        params["ground_truth_column"] = gt_col

        if params["mode"] == "Evaluation":
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
                    min_value=1,
                    value=2,
                    key="few_shot_count",
                )
                params["few_shot_count"] = few_shot_count
            if "Many Shot" in eval_techniques:
                many_shot_train_ratio = self.ui.number_input(
                    "Enter train proportion for Many Shot (0 to 1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    key="many_shot_train_ratio",
                )
                params["many_shot_train_ratio"] = many_shot_train_ratio

        default_categories = self._get_default_categories(df, gt_col)
        categories_dict, categories_with_descriptions = CategoryManager.define_categories(
            self.ui, "tab1", unique_values_str=default_categories
        )
        params["categories_dict"] = categories_dict
        params["categories_with_descriptions"] = categories_with_descriptions
        return params

    def categorize_data(self, df, params, zs_prompty, fs_prompty):
        """
        The function categorizes data based on specified parameters, using different techniques such as
        Zero Shot, Few Shot, and Many Shot categorization.

        Args:
          df: The `df` parameter in the `categorize_data` function is typically a pandas DataFrame
        containing the data that needs to be categorized. This DataFrame should include columns with the
        data to be categorized, as well as any additional columns required for the categorization
        process such as unique identifiers, ground truth labels
          params: The `params` dictionary contains the following keys and their corresponding values:
          zs_prompty: The `zs_prompty` parameter in the code snippet refers to the path of the Zero Shot
        categorization model that will be used in the workflow. This model is utilized for performing
        Zero Shot categorization, where the model predicts categories for data points without any prior
        training on those specific categories.
          fs_prompty: The `fs_prompty` parameter in the provided code snippet is used as the path to the
        Few-Shot Categorizer model for categorizing data. This path is utilized when running the
        workflow for Few Shot categorization within the `categorize_data` method. The Few-Shot Categor

        Returns:
          The `categorize_data` function returns a DataFrame `merged_df` that contains the original data
        along with additional columns for predicted category and rationale based on the categorization
        process specified in the function. The specific columns added to `merged_df` depend on the
        categorization mode chosen (Evaluation or Production) and the evaluation techniques used (Zero
        Shot, Few Shot, Many Shot).
        """
        index_column = params["index_column"]
        cat_col = params["categorizing_column"]
        mode_choice = params["mode"]

        if index_column not in df.columns:
            self.ui.info(
                f"Unique ID column '{index_column}' was not found. Creating new column '{index_column}'."
            )
            df[index_column] = df.index.astype(str)
            self.ui.session_state["selected_id_column"] = index_column

        categories_dict = params.get("categories_dict", {})
        text_to_label = df[cat_col].astype(str).tolist()
        unique_ids = df[index_column].astype(str).tolist()

        if mode_choice == "Evaluation":
            categorization_request = CategoryManager.create_request(
                unique_ids, text_to_label, categories_dict
            )
            gt_column = params["ground_truth_column"]
            eval_techniques = params.get("evaluation_techniques", ["Zero Shot"])
            predictions = {}
            few_shot_count = params.get("few_shot_count", 2)
            many_shot_train_ratio = params.get("many_shot_train_ratio", 0.8)
            (
                few_shot_examples,
                few_shot_ids,
                many_shot_examples,
                many_shot_test_ids,
            ) = self._prepare_ground_truth_examples(
                df, index_column, cat_col, gt_column, few_shot_count, many_shot_train_ratio
            )
            for technique in eval_techniques:
                if technique == "Zero Shot":
                    with self.ui.spinner("Running workflow (Zero Shot categorization)..."):
                        categorizer = ZeroShotCategorizer(
                            prompty_path=zs_prompty, category_request=categorization_request
                        )
                        results = categorizer.process()
                    predictions["Zero Shot"] = results
                elif technique == "Few Shot":
                    few_shot_request = CategoryManager.create_request(
                        unique_ids, text_to_label, categories_dict, few_shot_examples
                    )
                    with self.ui.spinner("Running workflow (Few Shot categorization)..."):
                        categorizer = FewShotCategorizer(
                            prompty_path=fs_prompty, category_request=few_shot_request
                        )
                        results = categorizer.process()
                    filtered_results = [r for r in results if str(r[0]) not in few_shot_ids]
                    predictions["Few Shot"] = filtered_results
                elif technique == "Many Shot":
                    many_shot_request = CategoryManager.create_request(
                        unique_ids, text_to_label, categories_dict, many_shot_examples
                    )
                    with self.ui.spinner("Running workflow (Many Shot categorization)..."):
                        categorizer = ManyshotClassifier(
                            categorization_request=many_shot_request,
                            min_class_count=self.config.MIN_SAMPLES_MANY_SHOT,
                        )
                        results = categorizer.process()
                    results = [r for r in results if str(r[0]) in many_shot_test_ids]
                    predictions["Many Shot"] = results

            merged_df = df.copy()
            for tech, results in predictions.items():
                tech_pred_df = pd.DataFrame(
                    [(row[0], row[2], row[3]) for row in results],
                    columns=[index_column, f"Predicted Category ({tech})", f"Rationale ({tech})"],
                )
                merged_df[index_column] = merged_df[index_column].astype(str)
                tech_pred_df[index_column] = tech_pred_df[index_column].astype(str)
                merged_df = pd.merge(merged_df, tech_pred_df, on=index_column, how="left")
            return merged_df
        else:
            gt_column = params["ground_truth_column"]
            all_examples = [
                (str(text), str(gt))
                for text, gt in zip(df[cat_col], df[gt_column].astype(str).str.lower())
                if gt
            ]
            categorization_request = CategoryManager.create_request(
                unique_ids, text_to_label, categories_dict, all_examples
            )
            num_examples = len(all_examples)
            self.ui.write(f"Number of examples provided: {num_examples}")
            categorized_results = []
            if num_examples == 0:
                with self.ui.spinner("Running workflow (Zero-Shot Production)..."):
                    categorizer = ZeroShotCategorizer(
                        prompty_path=zs_prompty, category_request=categorization_request
                    )
                    categorized_results = categorizer.process()
            elif num_examples < Config.MIN_SAMPLES_MANY_SHOT:
                with self.ui.spinner("Running workflow (Few-Shot Production)..."):
                    categorizer = FewShotCategorizer(
                        prompty_path=fs_prompty, category_request=categorization_request
                    )
                    categorized_results = categorizer.process()
            else:
                label_counts = df[gt_column].value_counts()
                min_class_samples = label_counts.min() if not label_counts.empty else 0
                with self.ui.spinner("Running workflow (Many-Shot Production)..."):
                    categorizer = ManyshotClassifier(
                        categorization_request=categorization_request,
                        min_class_count=min_class_samples,
                    )
                    categorized_results = categorizer.process()
            results_df = pd.DataFrame(
                [(row[0], row[2], row[3]) for row in categorized_results],
                columns=[index_column, "Category", "Rationale"],
            )
            df[index_column] = df[index_column].astype(str)
            results_df[index_column] = results_df[index_column].astype(str)
            merged_df = pd.merge(df, results_df, on=index_column, how="left")
            final_columns = list(df.columns) + ["Category", "Rationale"]
            merged_df = merged_df[final_columns]
            return merged_df

    def handle_single_upload(
        self, zs_prompty=Path(Config.ZS_PROMPTY), fs_prompty=Path(Config.FS_PROMPTY)
    ):
        """
        The `handle_single_upload` function handles the uploading, processing, and categorization of a
        single CSV or XLSX file.

        Args:
          zs_prompty: The `zs_prompty` parameter in the `handle_single_upload` method is set to the value of
        `Path(Config.ZS_PROMPTY)`. This means that the `zs_prompty` parameter is a Path object initialized
        with the value of `Config.ZS_PROMPTY`.
          fs_prompty: The `fs_prompty` parameter in the `handle_single_upload` method is a Path object that
        is set to the value of `Config.FS_PROMPTY`. It is used as a default value for the parameter if no
        value is provided when calling the method. The purpose of this parameter is to

        Returns:
          The `handle_single_upload` method returns either None if the uploaded file is None or if there is
        an error processing the uploaded file. Otherwise, it does not explicitly return a value, but it
        performs various operations such as setting session state variables, processing the uploaded file,
        categorizing the data, and providing a download button for the results.
        """
        file = self._ensure_file(
            file=None,
            upload_message="Upload your file",
            file_types=["csv", "xlsx"],
            key="single_file_uploader",
            info_message="Please upload a CSV or XLSX file to proceed.",
        )
        if file is None:
            return
        self.ui.session_state["uploaded_file_single"] = file
        try:
            document_analysis_client = self.get_document_analysis_client()
            upload_manager = StreamlitUploadManager(
                file, accept_multiple_files=False, document_analysis_client=document_analysis_client
            )
            file_data, extension = upload_manager.process_upload()
            self.ui.session_state["single_file_df"] = file_data
        except Exception as e:
            self.ui.error(f"Error processing the uploaded file: {e}")
            return

        if self.ui.session_state.get("single_file_df") is not None:
            df = self.ui.session_state["single_file_df"]
            self.ui.write(f"Uploaded file: {file.name}")
            params = self.setup_workflow(df)
            if self.ui.button("Categorize", key="tab1_submit"):
                merged_df = self.categorize_data(df, params, zs_prompty, fs_prompty)
                csv_data = merged_df.to_csv(index=False).encode("utf-8")
                self.ui.download_button(
                    label="Download Results",
                    data=csv_data,
                    file_name="AI_Generated_Categorization.csv",
                    mime="text/csv",
                )

    def process_multiple_files(self, uploaded_files):
        """
        The function `process_multiple_files` processes uploaded files using a document analysis client and
        returns the filenames and corresponding texts.

        Args:
          uploaded_files: Uploaded_files is a list of files that have been uploaded by the user. Each file
        in the list represents a file that the user has uploaded for processing.

        Returns:
          The `process_multiple_files` function returns two lists: `filenames` and `texts`. The `filenames`
        list contains the names of the uploaded files, while the `texts` list contains the processed data
        from the uploaded files.
        """
        filenames = []
        texts = []
        document_analysis_client = self.get_document_analysis_client()
        for file in uploaded_files:
            upload_manager = StreamlitUploadManager(
                file, accept_multiple_files=False, document_analysis_client=document_analysis_client
            )
            file_data, _ = upload_manager.process_upload()
            if file_data:
                filenames.append(file.name)
                texts.append(file_data)
        return filenames, texts

    def process_results(self, results):
        """
        The function `process_results` displays categorization results with optional formatting options.

        Args:
          results: The `results` parameter in the `process_results` method is used to pass the results that
        need to be processed or displayed. It could be any data structure or information that the method
        needs to work with, such as a list, dictionary, string, etc.
        """
        self.ui.write("Categorization Results:")
        self.ui.write(results)

    def handle_multiple_upload(
        self, zs_prompty=Path(Config.ZS_PROMPTY), fs_prompty=Path(Config.FS_PROMPTY)
    ):
        """
        The function `handle_multiple_upload` manages the upload and processing of multiple files,
        categorizes text excerpts, and processes the categorization results using either a few-shot or
        zero-shot categorizer.

        Args:
          zs_prompty: The `zs_prompty` parameter in the `handle_multiple_upload` function is a Path object
        that represents the path to the ZS_PROMPTY configuration in the Config module. It is used as a
        default value for the `prompty_path` parameter when initializing a ZeroShotCategorizer object in
          fs_prompty: The `fs_prompty` parameter in the `handle_multiple_upload` function seems to be a Path
        object that is initialized with the value from `Config.FS_PROMPTY`. It is used as the `prompty_path`
        parameter when creating instances of `FewShotCategorizer` or `ZeroShotCategorizer`
        """
        # Provide a "Clear All" option as before.
        if self.ui.button("Clear All", key="multi_clear_all"):
            if "uploaded_files_multiple" in self.ui.session_state:
                del self.ui.session_state["uploaded_files_multiple"]
            if "processed_files_multiple" in self.ui.session_state:
                del self.ui.session_state["processed_files_multiple"]
            self.ui.rerun()

        # Use the common file uploader helper for multiple files.
        files = self._ensure_file(
            file=None,
            upload_message="Upload your files",
            file_types=["docx", "pdf"],
            key="multiple_file_uploader",
            info_message="Please upload DOCX or PDF files to proceed.",
        )
        if files:
            self.ui.session_state["uploaded_files_multiple"] = files
            file_names = [file.name for file in files] if isinstance(files, list) else [files.name]
            self.ui.write(f"Uploaded files: {file_names}")
            filenames, texts = self.process_multiple_files(files)
            self.ui.session_state["processed_files_multiple"] = (filenames, texts)

        if self.ui.session_state.get("processed_files_multiple") is not None:
            filenames, texts = self.ui.session_state["processed_files_multiple"]
            text_to_label = texts
            categories_dict, examples = CategoryManager.define_categories(
                self.ui, "tab2", get_file_examples=True
            )
            self.ui.write("Texts (excerpts) to label:")
            for text in text_to_label[:5]:
                self.ui.write(text[:250])
            if self.ui.button("Categorize", key="tab2_submit"):
                categorized_results = []
                if examples:
                    categorization_request = CategoryManager.create_request(
                        filenames, text_to_label, categories_dict, examples
                    )
                    few_shot_categorizer = FewShotCategorizer(
                        prompty_path=fs_prompty, category_request=categorization_request
                    )
                    categorized_results = few_shot_categorizer.process()
                else:
                    categorization_request = CategoryManager.create_request(
                        filenames, text_to_label, categories_dict
                    )
                    zero_shot_categorizer = ZeroShotCategorizer(
                        prompty_path=zs_prompty, category_request=categorization_request
                    )
                    categorized_results = zero_shot_categorizer.process()
                self.process_results(categorized_results, replace_newlines=True, fill_missing=True)


# --------------------------
# For Standalone Testing
if __name__ == "__main__":
    import streamlit as st

    from LabeLMaker.utils.page_renderer import UIHelper

    st.title("Evaluate and Categorize Handler Test")
    ui = UIHelper()

    # Uncomment one of the following to test the corresponding workflow:

    # Evaluate workflow:
    evaluator_handler = EvaluateHandler(ui)
    evaluator_handler.handle_evaluation()

    # Single file categorization:
    # categorize_handler = CategorizeHandler(ui)
    # categorize_handler.handle_single_upload()

    # Multiple files categorization:
    # categorize_handler = CategorizeHandler(ui)
    # categorize_handler.handle_multiple_upload()
