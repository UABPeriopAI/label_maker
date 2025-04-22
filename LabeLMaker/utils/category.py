import pandas as pd

from app.v01.schemas import Categories, CategorizationRequest, Example
from LabeLMaker.utils.file_manager import FileManager


class CategoryManager:
    @staticmethod
    def define_categories(ui_helper, key_prefix, unique_values_str=None):
        """
        ui_helper: a helper object wrapping Streamlit calls so that category logic stays separate.
        """
        ui_helper.markdown("---")
        if unique_values_str:
            unique_values = [
                val.strip()
                for val in unique_values_str.split(",")
                if val.strip().lower() not in ["nan", "none"]
            ]
            num_categories = len(unique_values)
        else:
            num_categories = int(
                ui_helper.number_input(
                    "Enter the number of categories",
                    min_value=2,
                    value=2,
                    step=1,
                    key=f"{key_prefix}_categories",
                )
            )
            unique_values = None

        categories_dict = {}

        for i in range(num_categories):
            with ui_helper.expander(f"Category {i+1}", expanded=True):
                if unique_values:
                    category_value = ui_helper.text_input(
                        f"Enter label for category {i+1}",
                        value=unique_values[i].title(),
                        key=f"{key_prefix}_text_input_{i+1}",
                    )
                else:
                    category_value = ui_helper.text_input(
                        f"Enter label for category {i+1}", key=f"{key_prefix}_text_input_{i+1}"
                    )
                category_description = ui_helper.text_input(
                    f"Enter description for category {i+1} (optional but recommended)",
                    "",
                    key=f"{key_prefix}_desc_input_{i+1}",
                )
                categories_dict[category_value.lower()] = category_description or ""

        return categories_dict

    @staticmethod
    def create_request(index_list, df_text, categories_dict, examples=None):
        categories = [
            Categories(name=name, description=desc) for name, desc in categories_dict.items()
        ]

        # Convert examples from tuple to Example objects if necessary.
        if examples:
            new_examples = []
            for ex in examples:
                # If ex is already an Example instance, leave it as is.
                if isinstance(ex, Example):
                    new_examples.append(ex)
                # Otherwise, assume it's a tuple (text, label) and convert it.
                elif isinstance(ex, (list, tuple)) and len(ex) == 2:
                    new_examples.append(Example(text_with_label=ex[0], label=ex[1]))
                else:
                    raise ValueError(
                        "Example must be an Example object or a tuple of (text, label)."
                    )
            examples = new_examples

        cat_req = CategorizationRequest(
            unique_ids=index_list, text_to_label=df_text, categories=categories, examples=examples
        )
        return cat_req
