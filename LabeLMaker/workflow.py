"""
The `BaseCategorizeHandler` class contains methods for categorizing data using different techniques
such as Zero Shot, Few Shot, and Many Shot, with the ability to handle both evaluation and
production modes.
"""
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

from app.v01.schemas import Example
from LabeLMaker.Categorize.fewshot import FewShotCategorizer
from LabeLMaker.Categorize.manyshot import ManyshotClassifier
from LabeLMaker.Categorize.zeroshot import ZeroShotCategorizer
from LabeLMaker.utils.category import CategoryManager
from LabeLMaker_config.config import Config


class BaseCategorizeHandler:
    """
    The `BaseCategorizeHandler` class in Python contains methods for preparing ground truth examples
    and categorizing data based on different modes and evaluation techniques.
    """

    def __init__(self, azure_key=None):
        self.config = Config
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
        """
        Same utility method used in both Streamlit and FastAPI versions.
        """
        few_shot_examples = []
        few_shot_ids = set()
        many_shot_examples = []
        many_shot_test_ids = set()

        df_gt = df[[id_col, text_col, gt_col]].copy()
        df_gt[gt_col] = df_gt[gt_col].astype(str).str.lower()
        grouped = df_gt.groupby(gt_col)

        for _, group in grouped:
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

    def categorize_data(
        self,
        df: pd.DataFrame,
        mode: str,
        index_column: str,
        text_column: str,
        ground_truth_column: str,
        categories_dict: Dict[str, Any],
        zs_prompty: Path,
        fs_prompty: Path,
        evaluation_techniques: List[str] = None,
        few_shot_count: int = Config.FEW_SHOT_COUNT,
        many_shot_train_ratio: float = Config.MANY_SHOT_TRAIN_RATIO,
    ) -> pd.DataFrame:
        """
        The main categorization logic, shared by both Streamlit and FastAPI.
        mode should be either 'Evaluation'/'evaluation' or 'Production'/'production'.
        """
        if not index_column:
            df["index"] = df.index.astype(str)
            index_column = "index"

        text_to_label = df[text_column].astype(str).tolist()
        unique_ids = df[index_column].astype(str).tolist()

        # If the user is in evaluation mode
        if mode.lower() == "evaluation":
            # Prepare request
            categorization_request = CategoryManager.create_request(
                unique_ids, text_to_label, categories_dict
            )
            predictions = {}

            # Prepare ground-truth examples
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

            for tech in evaluation_techniques or ["Zero Shot"]:
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
                    # Remove few-shot training examples from final predictions
                    filtered_results = [r for r in results if str(r[0]) not in few_shot_ids]
                    predictions["Few Shot"] = filtered_results

                elif tech == "Many Shot":
                    ms_request = CategoryManager.create_request(
                        unique_ids, text_to_label, categories_dict, many_shot_examples
                    )
                    ms_categorizer = ManyshotClassifier(
                        categorization_request=ms_request,
                        min_class_count=self.config.MIN_SAMPLES_MANY_SHOT,
                    )
                    results = ms_categorizer.process()
                    # Only keep test partition
                    results = [r for r in results if str(r[0]) in many_shot_test_ids]
                    predictions["Many Shot"] = results

            merged_df = df.copy()
            for technique, results in predictions.items():
                tech_pred_df = pd.DataFrame(
                    [(row[0], row[2], row[3]) for row in results],
                    columns=[
                        index_column,
                        f"Predicted Category ({technique})",
                        f"Rationale ({technique})",
                    ],
                )
                # Ensure both DataFrames have string values in the index_column
                merged_df[index_column] = merged_df[index_column].astype(str)
                tech_pred_df[index_column] = tech_pred_df[index_column].astype(str)

                merged_df = pd.merge(merged_df, tech_pred_df, on=index_column, how="left")

            return merged_df

        else:
            # Production mode
            if ground_truth_column and ground_truth_column.strip():
                # We can see if few shot or many shot is possible

                all_examples = [
                    (str(txt), str(gt).lower())
                    for txt, gt in zip(
                        df[text_column], df[ground_truth_column].astype(str).str.lower()
                    )
                    if gt
                ]

                categorization_request = CategoryManager.create_request(
                    unique_ids, text_to_label, categories_dict, all_examples
                )

                # Decide which approach to run
                label_counts = df[ground_truth_column].value_counts()
                min_class_samples = label_counts.min() if not label_counts.empty else 0

                if min_class_samples >= self.config.MIN_SAMPLES_MANY_SHOT:
                    manyshot = ManyshotClassifier(
                        categorization_request=categorization_request,
                        min_class_count=min_class_samples,
                    )
                    categorized_results = manyshot.process()
                else:
                    fewshot = FewShotCategorizer(
                        prompty_path=fs_prompty, category_request=categorization_request
                    )
                    categorized_results = fewshot.process()

            else:
                # No ground truth column at all => zero shot
                categorization_request = CategoryManager.create_request(
                    unique_ids, text_to_label, categories_dict
                )
                zs_categorizer = ZeroShotCategorizer(
                    prompty_path=zs_prompty, category_request=categorization_request
                )
                categorized_results = zs_categorizer.process()

            # Merge results
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


class FastAPICategorizeHandler(BaseCategorizeHandler):
    def __init__(self, azure_key=None):
        super().__init__(azure_key=azure_key)

    def fastapi_categorize(self, data, request, zs_prompty, fs_prompty) -> pd.DataFrame:
        # Gather parameters for fastapi_categorize
        index_column = request.index_column
        text_column = request.text_column
        gt_column = (
            request.ex_label_column if request.ex_label_column else None
        )  # Optional ground truth column
        categories_dict = {
            cat.name: cat.description for cat in request.categories
        }  # Construct categories dict

        # Call the shared base logic
        return self.categorize_data(
            df=data,
            mode=request.mode,
            index_column=index_column,
            text_column=text_column,
            ground_truth_column=gt_column,
            categories_dict=categories_dict,
            zs_prompty=zs_prompty,
            fs_prompty=fs_prompty,
            evaluation_techniques=request.model,
            few_shot_count=int(request.few_shot_count),
            many_shot_train_ratio=float(request.many_shot_train_ratio),
        )
