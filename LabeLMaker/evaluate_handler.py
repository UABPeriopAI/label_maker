"""
Evaluation Handlers

This module provides three classes:
  • BaseEvaluateHandler: Contains core evaluation methods (data loading, model evaluation,
    metric calculation, bootstrap confidence interval computation, confusion matrix plotting).
  • StreamlitEvaluateHandler: Inherits from BaseEvaluateHandler and integrates with a Streamlit UI.
  • FastAPIEvaluateHandler: Inherits from BaseEvaluateHandler and exposes a FastAPI–friendly method.
  
Note:
  This module assumes that the underlying evaluation components (DataLoader, Evaluator,
  compute_bootstrap_confidence_intervals, etc.) are available.
"""

import io
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from LabeLMaker.Evaluate.confidence_intervals import compute_bootstrap_confidence_intervals

# --- External Dependencies (assumed present) ---
from LabeLMaker.Evaluate.data_loader import DataLoader
from LabeLMaker.Evaluate.evaluator import Evaluator
from LabeLMaker_config.config import Config


class BaseEvaluateHandler:
    """
    Provides core evaluation functionality.
    """

    def __init__(self, azure_key: str = None) -> None:
        self.azure_key = azure_key
        self.config = Config

    def load_data(self, file: Any) -> pd.DataFrame:
        """
        Load CSV evaluation data.
        """
        # We assume the uploaded file (or file path) is CSV.
        return pd.read_csv(file)

    def evaluate_model(
        self,
        df: pd.DataFrame,
        pred_col: str,
        ground_truth_col: str,
        n_bootstraps: int = 1000,
        alpha: float = 0.05,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
        """
        Evaluate model predictions and compute associated metrics.
        Returns a tuple:
          (metrics_df, classification_report_df, bootstrap_df, confusion_matrix_fig)
        """
        if pred_col not in df.columns or ground_truth_col not in df.columns:
            raise ValueError("Prediction or ground truth column not found in DataFrame.")

        valid = df[pred_col].notna()
        y_true = df.loc[valid, ground_truth_col].astype(str).tolist()
        y_pred = df.loc[valid, pred_col].astype(str).tolist()

        evaluator = Evaluator(y_true, y_pred)
        evaluator.calculate_metrics()
        metrics_df = evaluator.display_metrics()
        report_df = pd.DataFrame(evaluator.metrics["Classification Report"]).transpose()
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

    def compare_methods(
        self,
        df: pd.DataFrame,
        ground_truth_col: str,
        selected_methods: list,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """
        Compare prediction methods (e.g. Zero Shot, Few Shot, Many Shot)
        by evaluating predictions in multiple columns.
        Returns (common_df, results, confusion_matrices)
        """
        # Map the method to the corresponding DataFrame column name.
        method_columns = {method: f"Predicted Category ({method})" for method in selected_methods}
        valid_methods = {m: col for m, col in method_columns.items() if col in df.columns}
        if not valid_methods:
            raise ValueError("No selected method prediction columns exist in DataFrame.")

        common_df = df.dropna(subset=list(valid_methods.values()))
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


class StreamlitEvaluateHandler(BaseEvaluateHandler):
    """
    Integrates the evaluation workflow with a Streamlit UI.
    """

    def __init__(self, ui_helper: Any, azure_key: str = None) -> None:
        super().__init__(azure_key=azure_key)
        self.ui = ui_helper

    def generate_docx_report_download(self, doc: Any) -> bytes:
        """
        Convert a DOCX document to bytes for download.
        """
        with io.BytesIO() as temp_stream:
            doc.save(temp_stream)
            temp_stream.seek(0)
            return temp_stream.read()

    def handle_evaluation(self, file: Any = None) -> None:
        """
        Execute the evaluation workflow:
          – File upload
          – Data preview
          – Ground truth column selection
          – Display class balance & evaluation methods (via LabeLMaker utility)
          – Evaluate predictions and offer DOCX report for download
        """
        # Step 1: File upload using the provided UI helper.
        file = self.ui.file_uploader(
            "Upload a CSV file for Evaluation", type=["csv"], accept_multiple_files=False, key="eval_file_uploader"
        )
        if file is None:
            self.ui.info("Please upload a CSV file to proceed.")
            return

        df = self.load_data(file)
        self.ui.subheader("CSV Preview:")
        self.ui.write(df.head())
        self.ui.write(f"Total rows: {len(df)}")

        # Step 2: Ground truth column selection.
        ground_truth_col = self.ui.selectbox(
            "Select the Ground Truth Column", df.columns, key="eval_gt_column"
        )
        self.ui.subheader(f"Class Balance for {ground_truth_col}")
        from LabeLMaker.utils.class_balance import ClassBalance  # assumed external utility

        balancer = ClassBalance(df, ground_truth_col)
        self.ui.write(balancer.compute_balance())

        selected_methods = self.ui.multiselect(
            "Select Evaluation Methods",
            ["Zero Shot", "Few Shot", "Many Shot"],
            default=["Zero Shot"],
            key="eval_methods",
        )

        if self.ui.button("Evaluate"):
            try:
                common_df, results, confusion_matrices = self.compare_methods(
                    df, ground_truth_col, selected_methods
                )
                # Display results for each method.
                for method, metrics in results.items():
                    self.ui.subheader(method)
                    self.ui.write(metrics)
                # Generate a DOCX report.
                from aiweb_common.file_operations.docx_creator import StreamlitDocxCreator

                docx_maker = StreamlitDocxCreator(results, confusion_matrices)
                doc = docx_maker.create_docx_report()
                docx_content = self.generate_docx_report_download(doc)
                self.ui.download_button(
                    label="Download DOCX Report",
                    data=docx_content,
                    file_name="evaluation_report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            except Exception as e:
                self.ui.error(str(e))


class FastAPIEvaluateHandler(BaseEvaluateHandler):
    """
    Provides a FastAPI–friendly evaluation method.
    """

    def __init__(self, azure_key: str = None) -> None:
        super().__init__(azure_key=azure_key)

    def fastapi_evaluate(self, data: pd.DataFrame, request: Any) -> Dict[str, Any]:
        """
        Execute evaluation and return results as a JSON–serializable dictionary.
        Expects that request defines:
          • ground_truth_column
          • pred_column
          • Optional: n_bootstraps, alpha
        """
        ground_truth_col = request.ground_truth_column
        pred_col = request.pred_column
        n_bootstraps = getattr(request, "n_bootstraps", 1000)
        alpha = getattr(request, "alpha", 0.05)
        try:
            metrics_df, report_df, bs_df, cm_fig = self.evaluate_model(
                data, pred_col, ground_truth_col, n_bootstraps=n_bootstraps, alpha=alpha
            )
            response = {
                "metrics": metrics_df.to_dict(),
                "classification_report": report_df.to_dict(),
                "bootstrap_confidence_intervals": bs_df.to_dict(),
                "confusion_matrix": str(cm_fig),  # Customize serialization as needed.
            }
            return response
        except Exception as e:
            return {"error": str(e)}


# For module-level testing, you might run:
if __name__ == "__main__":
    print("This module contains evaluation classes for Streamlit and FastAPI.")
