import traceback
from pathlib import Path

from app.v01.schemas import Categories, CategorizationRequest, Example
from LabeLMaker.Categorize.categorizer import LabeLMaker


class FewShotCategorizer(LabeLMaker):
    def __init__(self, prompty_path: Path, category_request: CategorizationRequest):
        print("Initializing Few-shot categorizer")
        super().__init__(prompty_path, category_request)
        self._validate_examples()

    def _get_filename(self):
        return "fewshot_categorizer_output.txt"

    def _get_mime_type(self):
        return "text/plain"

    def _validate_prompt_template(self, prompt_template):
        """
        The function `_validate_prompt_template` checks if certain expected variables are present in a given
        prompt template for few-shot prompts.

        :param prompt_template: It looks like you are trying to validate mandatory variables in a few-shot
        prompt template. The expected variables are 'item', 'categories_with_descriptions', and 'examples'.
        The code snippet you provided checks if these variables are present in the prompt template. If any
        of the expected variables are missing, a
        """
        # Validate mandatory variables for few-shot prompts
        expected_variables = ["item", "categories_with_descriptions", "examples"]
        for var in expected_variables:
            if f"{var}" not in prompt_template:
                raise ValueError(
                    f"Expected variable {{var}} not found in the few-shot prompt template."
                )

    def _validate_examples(self):
        """
        This function validates that there is at least one example for each proposed category in a
        categorization request.
        """
        # Ensure that there is at least one example for each proposed category
        category_examples = {category.name: 0 for category in self.categorzation_request.categories}
        for example in self.categorzation_request.examples or []:
            if isinstance(example, Example):
                if example.label in category_examples:
                    category_examples[example.label] += 1
            else:
                raise TypeError(f"Expected an instance of Example, but got {type(example)} instead")

        for category, count in category_examples.items():
            if count == 0:
                raise ValueError(
                    f"No examples provided for category: '{category}'. Each category must have at least one example."
                )

    def _prepare_prompt_inputs(self):
        """
        The `_prepare_prompt_inputs` function prepares categories with descriptions and examples for a
        prompt input.
        :return: The `_prepare_prompt_inputs` method returns a dictionary containing the following keys and
        values:
        - 'categories_with_descriptions': a list of dictionaries where each dictionary contains a category
        name and its description (or a default message if no description is provided)
        - 'examples': examples prepared by the `_prepare_examples` method if available, otherwise it is set
        to None
        """
        # Prepare categories with descriptions like in ZeroShotCategorizer
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

        # Prepare examples if available
        examples = self._prepare_examples() if self.categorzation_request.examples else None

        # Combine prompt inputs
        prompt_inputs = {
            "categories_with_descriptions": categories_with_descriptions,
            "examples": examples,
        }

        return prompt_inputs

    def _prepare_examples(self):
        """
        The `_prepare_examples` function prepares example prompt text by converting them into a list of
        dictionaries, ensuring they are instances of the `Example` class.
        :return: The method `_prepare_examples` is returning a list of dictionaries containing the text with
        label and label of each example in the categorization request. The method also includes a debug
        print statement to show the prepared examples before returning them.
        """
        # Prepare the example prompt text as a list of dictionaries
        prompt_examples = []
        for example in self.categorzation_request.examples:
            if isinstance(example, Example):
                prompt_examples.append(
                    {"text_with_label": example.text_with_label, "label": example.label}
                )
            else:
                raise TypeError(f"Expected an instance of Example, but got {type(example)} instead")
        return prompt_examples

    def categorize_item(self, item_args):
        """
        The `categorize_item` function updates item arguments with prompt inputs, invokes a chain, handles
        exceptions, and returns the result or None.

        :param item_args: The `item_args` parameter in the `categorize_item` method seems to be a dictionary
        containing arguments related to an item. These arguments are updated with prompt inputs obtained
        from the `_prepare_prompt_inputs` method before being passed to the `chain.invoke` method for
        processing
        :return: The `categorize_item` method returns the result of invoking the `chain` with the
        `item_args` after updating it with prompt inputs. If the result is `None`, a `ValueError` is raised.
        If an exception occurs during the processing, an error message is printed along with the exception
        details, and `None` is returned.
        """
        # Get the prompt inputs from _prepare_prompt_inputs and update item_args
        prompt_inputs = self._prepare_prompt_inputs()
        item_args.update(prompt_inputs)

        try:
            result = self.chain.invoke(item_args)
            if result is None:
                raise ValueError(f"Chain returned None for input: {item_args}")
            return result
        except Exception as e:
            print(f"Error processing item: {item_args.get('item', '')}")
            print(f"Exception: {e}")
            traceback.print_exc()
            return None
