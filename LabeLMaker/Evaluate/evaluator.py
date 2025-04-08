"""
The `Evaluator` class in Python calculates and displays evaluation metrics such as accuracy,
precision, recall, F1 score, confusion matrix, and classification report for classification tasks.
"""
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class Evaluator:
    """
    This Python class `Evaluator` provides methods to calculate and display evaluation metrics for
    classification tasks, including precision, recall, F1 score, accuracy, confusion matrix, and
    classification report.
    """

    def __init__(self, y_true: List, y_pred: List) -> None:
        """
        Initializes the Evaluator with true and predicted labels.
        Parameters:
            y_true (List): Ground truth labels.
            y_pred (List): Predicted labels.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.metrics: Dict[str, Any] = {}

    @staticmethod
    def _format_numeric(value: Any) -> Any:
        """
        Format a numeric value to 5 significant figures. If the value is not numeric,
        it is returned unchanged.
        """
        if isinstance(value, (int, float)):
            # Convert to float and then format.
            return float(format(value, ".4g"))
        return value

    @classmethod
    def _format_dict(cls, d: Dict[Any, Any]) -> Dict[Any, Any]:
        """
        Recursively format all numeric entries in a dictionary to 5 significant figures.
        """
        formatted = {}
        for k, v in d.items():
            if isinstance(v, dict):
                # Recursively process nested dictionaries.
                formatted[k] = cls._format_dict(v)
            else:
                formatted[k] = cls._format_numeric(v)
        return formatted

    def calculate_metrics(self, average_options: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculates evaluation metrics.
        Parameters:
            average_options (List[str], optional): Averaging methods
                (e.g., ['macro', 'weighted']).
        Returns:
            Dict[str, Any]: Dictionary of calculated metrics.
        """
        if average_options is None:
            average_options = ["macro", "weighted"]

        # Calculate and format accuracy.
        self.metrics["Accuracy"] = self._format_numeric(accuracy_score(self.y_true, self.y_pred))

        # Calculate precision, recall, and f1 scores for each averaging option.
        for avg in average_options:
            self.metrics[f"Precision ({avg})"] = self._format_numeric(
                precision_score(self.y_true, self.y_pred, average=avg, zero_division=0)
            )
            self.metrics[f"Recall ({avg})"] = self._format_numeric(
                recall_score(self.y_true, self.y_pred, average=avg, zero_division=0)
            )
            self.metrics[f"F1 Score ({avg})"] = self._format_numeric(
                f1_score(self.y_true, self.y_pred, average=avg, zero_division=0)
            )

        # Compute and save the confusion matrix without formatting.
        self.metrics["Confusion Matrix"] = confusion_matrix(self.y_true, self.y_pred)

        # Compute and then recursively format the classification report.
        raw_report = classification_report(
            self.y_true, self.y_pred, output_dict=True, zero_division=0
        )
        self.metrics["Classification Report"] = self._format_dict(raw_report)

        return self.metrics

    def display_metrics(self) -> pd.DataFrame:
        """
        Returns calculated metrics as a DataFrame.
        """
        metrics_to_display = {
            k: v
            for k, v in self.metrics.items()
            if k not in ["Confusion Matrix", "Classification Report"]
        }
        df = pd.DataFrame(list(metrics_to_display.items()), columns=["Metric", "Value"])
        return df

    def plot_confusion_matrix(self, class_labels: Optional[List[str]] = None) -> plt.Figure:
        """
        Plots the confusion matrix.
        Parameters:
            class_labels (List[str], optional): Labels for the classes.
        Raises:
            ValueError: If confusion matrix is not calculated.
        """
        cm = self.metrics.get("Confusion Matrix")
        if cm is None:
            raise ValueError("Confusion Matrix not calculated. Call calculate_metrics() first.")
        if class_labels is None:
            class_labels = sorted(set(self.y_true) | set(self.y_pred))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.close(fig)  # Close the figure to prevent it from displaying automatically
        return fig
