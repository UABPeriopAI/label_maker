"""
Categorizer Handlers

This module defines methods to perform content categorization using different techniques.
It provides a base handler (BaseCategorizeHandler) for core logic and specialized handlers
for Streamlit and FastAPI integration.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional
import pandas as pd

from app.v01.schemas import Example
from LabeLMaker.Categorize.fewshot import FewShotCategorizer
from LabeLMaker.Categorize.manyshot import ManyshotClassifier
from LabeLMaker.Categorize.zeroshot import ZeroShotCategorizer
from LabeLMaker.utils.category import CategoryManager
from LabeLMaker_config.config import Config


class BaseCategorizeHandler:
    """
    Abstract base class that implements categorization logic.
    
    In evaluation mode, it accepts a list of evaluation techniques.
    In production mode it inspects the provided ground truth examples and then:
      - If there are enough training examples for a given category, uses Many Shot.
      - Else if there are a few examples available, uses Few Shot.
      - Otherwise, falls back to Zero Shot.
    
    UI/transport-specific request parsing should be done in child classes.
    """

    def __init__(self, azure_key: Optional[str] = None) -> None:
        self.config = Config  # Expose configuration constants.
        self.azure_key = azure_key

    def _prepare_ground_truth_examples(
        self,
        df: pd.DataFrame,
        id_col: str,
        text_col: str,
        gt_col: str,
        few_shot_count: int = Config.MIN_SAMPLES_FEW_SHOT,
        many_shot_train_ratio: float = Config.MANY_SHOT_TRAIN_RATIO,
    ) -> Tuple[List[Example], Set[str], List[Example], Set[str]]:
        few_shot_examples: List[Example] = []
        few_shot_ids: Set[str] = set()
        many_shot_examples: List[Example] = []
        many_shot_test_ids: Set[str] = set()

        df_gt = df[[id_col, text_col, gt_col]].copy()
        df_gt[gt_col] = df_gt[gt_col].astype(str).str.lower()

        # Group by the ground truth label.
        grouped = df_gt.groupby(gt_col)
        for _, group in grouped:
            records = group.to_dict(orient="records")
            # Select a few-shot sample up to the provided count.
            count = min(few_shot_count, len(records))
            if count > 0:
                sampled = random.sample(records, count)
                for rec in sampled:
                    few_shot_examples.append(
                        Example(text_with_label=str(rec[text_col]), label=str(rec[gt_col]))
                    )
                    few_shot_ids.add(str(rec[id_col]))

            # Prepare many-shot data if there are multiple records.
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

    def categorize_data(
        self,
        df: pd.DataFrame,
        mode: str,
        index_column: Optional[str],
        text_column: str,
        ground_truth_column: str,
        examples_column: str,
        categories_dict: Dict[str, Any],
        zs_prompty: Path,
        fs_prompty: Path,
        evaluation_techniques: Optional[List[str]] = None,
        few_shot_count: int = Config.FEW_SHOT_COUNT,
        many_shot_train_ratio: float = Config.MANY_SHOT_TRAIN_RATIO,
    ) -> pd.DataFrame:
        """
        The heart of the abstract categorization logic.
        
        If mode is "evaluation", it prepares ground truth examples and then applies
        the chosen evaluation techniques.
        
        Otherwise (production mode), it selects among zero, few, or many shot modes depending
        on whether a ground truth column is provided and on the number of examples available.
        """
        # If no dedicated index is passed in, add one.
        if not index_column:
            df["index"] = df.index.astype(str)
            index_column = "index"

        # Prepare a list of texts and corresponding unique ids.
        text_to_label = df[text_column].astype(str).tolist()
        unique_ids = df[index_column].astype(str).tolist()

        if mode.lower() == "evaluation" and evaluation_techniques:
            # Evaluation mode: run all requested techniques (e.g., Zero Shot, Few Shot, Many Shot)
            categorization_request = CategoryManager.create_request(
                unique_ids, text_to_label, categories_dict
            )
            predictions = {}

            (
                few_shot_examples,
                few_shot_ids,
                many_shot_examples,
                many_shot_test_ids,
            ) = self._prepare_ground_truth_examples(
                df,
                index_column,
                text_column,
                ground_truth_column,
                few_shot_count,
                many_shot_train_ratio,
            )

            for tech in evaluation_techniques:
                if tech == "Zero Shot":
                    zs_categorizer = ZeroShotCategorizer(
                        prompty_path=zs_prompty, category_request=categorization_request
                    )
                    results = zs_categorizer.process()
                    predictions["Zero Shot"] = results
                elif tech == "Few Shot":
                    fs_request = CategoryManager.create_request(
                        unique_ids, text_to_label, categories_dict, few_shot_examples
                    )
                    fs_categorizer = FewShotCategorizer(
                        prompty_path=fs_prompty, category_request=fs_request
                    )
                    results = fs_categorizer.process()
                    # Remove any examples already in the few-shot gold set.
                    predictions["Few Shot"] = [r for r in results if str(r[0]) not in few_shot_ids]
                elif tech == "Many Shot":
                    ms_request = CategoryManager.create_request(
                        unique_ids, text_to_label, categories_dict, many_shot_examples
                    )
                    ms_categorizer = ManyshotClassifier(
                        categorization_request=ms_request,
                        min_class_count=self.config.MIN_SAMPLES_MANY_SHOT,
                    )
                    results = ms_categorizer.process()
                    predictions["Many Shot"] = [r for r in results if str(r[0]) in many_shot_test_ids]
                else:
                    raise ValueError(f"Unsupported technique '{tech}' in evaluation mode.")

            # Merge the results into the dataframe: one additional set of columns per technique.
            merged_df = df.copy()
            for technique, results in predictions.items():
                tech_pred_df = pd.DataFrame(
                    [(row[0], row[2], row[3]) for row in results],
                    columns=[index_column,
                             f"Predicted Category ({technique})",
                             f"Rationale ({technique})"],
                )
                merged_df[index_column] = merged_df[index_column].astype(str)
                tech_pred_df[index_column] = tech_pred_df[index_column].astype(str)
                merged_df = pd.merge(merged_df, tech_pred_df, on=index_column, how="left")
            return merged_df

        else:
            print("gt col - ", examples_column)
            print("df cols - ", df.columns)
            # Production mode: choose the most appropriate technique based on the provided examples.
            if examples_column and examples_column in df.columns:
                (
                    few_shot_examples,
                    few_shot_ids,
                    many_shot_examples,
                    many_shot_test_ids,
                ) = self._prepare_ground_truth_examples(
                    df,
                    index_column,
                    text_column,
                    examples_column,
                    few_shot_count,
                    many_shot_train_ratio,
                )
                print("FS Examples - ", few_shot_examples)
                print("MS Examples - ", few_shot_examples)
                # Prefer Many Shot if we have enough training examples.
                if len(many_shot_examples) >= self.config.MIN_SAMPLES_MANY_SHOT:
                    ms_request = CategoryManager.create_request(
                        unique_ids, text_to_label, categories_dict, many_shot_examples
                    )
                    ms_categorizer = ManyshotClassifier(
                        categorization_request=ms_request,
                        min_class_count=self.config.MIN_SAMPLES_MANY_SHOT,
                    )
                    predictions = ms_categorizer.process()
                    # In production, we assume predictions are for test examples only.
                    results = [r for r in predictions if str(r[0]) in many_shot_test_ids]
                # Else, try Few Shot if any examples exist.
                elif len(few_shot_examples) > 0:
                    fs_request = CategoryManager.create_request(
                        unique_ids, text_to_label, categories_dict, few_shot_examples
                    )
                    fs_categorizer = FewShotCategorizer(
                        prompty_path=fs_prompty, category_request=fs_request
                    )
                    predictions = fs_categorizer.process()
                    results = predictions
                else:
                    # Fallback to Zero Shot if no ground truth examples are available.
                    zs_request = CategoryManager.create_request(
                        unique_ids, text_to_label, categories_dict
                    )
                    zs_categorizer = ZeroShotCategorizer(
                        prompty_path=zs_prompty, category_request=zs_request
                    )
                    results = zs_categorizer.process()
            else:
                # No ground truth column provided: use Zero Shot as default.
                zs_request = CategoryManager.create_request(
                    unique_ids, text_to_label, categories_dict
                )
                zs_categorizer = ZeroShotCategorizer(
                    prompty_path=zs_prompty, category_request=zs_request
                )
                results = zs_categorizer.process()

            # In production we assume a single set of predicted results.
            results_df = pd.DataFrame(
                [(row[0], row[2], row[3]) for row in results],
                columns=[index_column, "Category", "Rationale"],
            )
            # Ensure columns are strings for a proper merge.
            df[index_column] = df[index_column].astype(str)
            results_df[index_column] = results_df[index_column].astype(str)
            merged_df = pd.merge(df, results_df, on=index_column, how="left")

            final_columns = list(df.columns) + ["Category", "Rationale"]
            return merged_df[final_columns]


class StreamlitCategorizeHandler(BaseCategorizeHandler):
    """
    Thin wrapper for Streamlit usage.
    Expects that UI parameters (collected via the UI) are passed in a dictionary.
    """

    def __init__(self, azure_key: Optional[str] = None) -> None:
        super().__init__(azure_key=azure_key)

    def streamlit_categorize(
        self,
        df: pd.DataFrame,
        ui_params: Dict[str, Any],
        zs_prompty: Path,
        fs_prompty: Path,
    ) -> pd.DataFrame:
        """
        Extract values from the Streamlit UI dictionary and pass them to BaseCategorizeHandler.
        """
        return self.categorize_data(
            df=df,
            mode=ui_params.get("mode", "production"),
            index_column=ui_params.get("index_column"),
            text_column=ui_params.get("categorizing_column"),
            ground_truth_column=ui_params.get("ground_truth_column", ""),
            examples_column=ui_params.get("examples_column",""),
            categories_dict=ui_params.get("categories_dict", {}),
            zs_prompty=zs_prompty,
            fs_prompty=fs_prompty,
            evaluation_techniques=ui_params.get("evaluation_techniques"),
            few_shot_count=ui_params.get("few_shot_count", Config.FEW_SHOT_COUNT),
            many_shot_train_ratio=ui_params.get(
                "many_shot_train_ratio", Config.MANY_SHOT_TRAIN_RATIO
            ),
        )


class FastAPICategorizeHandler(BaseCategorizeHandler):
    """
    Provides categorization functionality for FastAPI endpoints.
    """

    def __init__(self, azure_key: Optional[str] = None) -> None:
        super().__init__(azure_key=azure_key)

    def fastapi_categorize(
        self, data: pd.DataFrame, request: Any, zs_prompty: Path, fs_prompty: Path
    ) -> pd.DataFrame:
        """
        Extract values from the FastAPI request and pass them to BaseCategorizeHandler.
        (Here we assume the request object carries attributes like index_column, text_column, etc.)
        """
        index_column = request.index_column
        text_column = request.text_column
        # ex_label_column may be empty or None.
        gt_column = request.ex_label_column if getattr(request, "ex_label_column", None) else ""
        examples_column=gt_column
        categories_dict = {cat.name: cat.description for cat in request.categories}

        return self.categorize_data(
            df=data,
            mode=request.mode,
            index_column=index_column,
            text_column=text_column,
            ground_truth_column=gt_column,
            examples_column=examples_column,
            categories_dict=categories_dict,
            zs_prompty=zs_prompty,
            fs_prompty=fs_prompty,
            evaluation_techniques=getattr(request, "model", None),  # Could be a list of techniques.
            few_shot_count=int(request.few_shot_count),
            many_shot_train_ratio=float(request.many_shot_train_ratio),
        )