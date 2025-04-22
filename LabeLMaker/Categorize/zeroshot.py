from pathlib import Path

from app.v01.schemas import Categories, CategorizationRequest
from LabeLMaker.Categorize.categorizer import LabeLMaker


class ZeroShotCategorizer(LabeLMaker):
    def __init__(self, prompty_path: Path, category_request: CategorizationRequest):
        print("Initializing Zero-shot categorizer")
        super().__init__(prompty_path, category_request)

    def _get_filename(self):
        return "zeroshot_categorizer_output.txt"

    def _get_mime_type(self):
        return "text/plain"

    def _validate_prompt_template(self, prompt_template):
        """
        The function `_validate_prompt_template` checks if mandatory variables for zero-shot prompts are
        present in a given prompt template.

        :param prompt_template: It looks like the code snippet you provided is a Python function for
        validating a prompt template against a list of expected variables. The function checks if certain
        variables ('item' and 'categories_with_descriptions') are present in the prompt template string
        """
        # Validate mandatory variables for zero-shot prompts
        expected_variables = ["item", "categories_with_descriptions"]
        for var in expected_variables:
            if f"{var}" not in prompt_template:
                raise ValueError(
                    f"Expected variable {{var}} not found in the zero-shot prompt template."
                )

    def _prepare_prompt_inputs(self):
        """
        The `_prepare_prompt_inputs` function prepares a list of categories with their descriptions,
        handling different types of categories appropriately.
        :return: A list of dictionaries containing the category name and description for each category in
        the categorization request. If a category does not have a description, it will be set to "No
        description provided".
        """
        categories_with_descriptions = []
        for category in self.categorzation_request.categories:
            if isinstance(category, Categories):
                categories_with_descriptions.append(
                    {
                        "category": category.name,
                        "description": (
                            category.description
                            if category.description
                            else "No description provided"
                        ),
                    }
                )
            else:
                raise TypeError(
                    f"Expected an instance of Categories, but got {type(category)} instead"
                )
        prompt_inputs = {"categories_with_descriptions": categories_with_descriptions}
        return prompt_inputs
