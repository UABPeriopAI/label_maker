"""
The Categorizer Module orchestrates the activity associated with categorizing the text.
It contains the logic for common aspects of categorization shared across zero and few
shot cases.
"""
import re
import traceback
from abc import ABC
from pathlib import Path

import streamlit as st
from aiweb_common.WorkflowHandler import WorkflowHandler

from app.v01.schemas import CategorizationRequest
from LabeLMaker.utils.normalize_text import normalize_text
from LabeLMaker_config.config import Config


class BaseCategorizer(WorkflowHandler, ABC):
    """
    The `BaseCategorizer` class is a subclass of `WorkflowHandler` and an abstract base class (ABC) with
    an empty constructor.
    """

    def __init__(self):
        pass


class LabeLMaker(BaseCategorizer):

    """
    The function initializes attributes for a prompt generator class with specified parameters.

    Args:
        prompty_path (Path): The `prompty_path` parameter is expected to be of type `Path` and represents
    the path to a file or directory related to the prompt.
        categorzation_request (CategorizationRequest): It looks like there is a typo in the parameter name
    `categorzation_request`. It should be `categorization_request` instead.
        llm_interface: The `llm_interface` parameter in the `__init__` method is a default parameter with
    a default value of `Config.LLM_INTERFACE`. This means that if no value is provided for
    `llm_interface` when creating an instance of the class, it will default to the value specified in
    """

    def __init__(
        self,
        prompty_path: Path,
        categorzation_request: CategorizationRequest,
        llm_interface=Config.LLM_INTERFACE,
    ):
        super().__init__()
        self.prompty_path = prompty_path
        self.categorzation_request = categorzation_request
        self.prompt_template = self.load_prompty()
        self.llm = llm_interface
        self.chain = self.create_chain()
        self.prompt_inputs = self._prepare_prompt_inputs()
        self.valid_categories = [
            category.name for category in self.categorzation_request.categories
        ]

    def _prepare_prompt_inputs(self):
        # This method should be overridden by child classes for specific validation
        raise NotImplementedError("Subclasses must implement prepare_prompt_inputs.")

    def _validate_prompt_template(self, prompt_template):
        # This method should be overridden by child classes for specific validation
        raise NotImplementedError("Subclasses must implement _validate_prompt_template.")

    def create_chain(self):
        """
        The `create_chain` function returns the result of combining the `prompt_template` and `llm`
        attributes.
        :return: The `create_chain` method is returning the result of the bitwise OR operation between
        `self.prompt_template` and `self.llm`.
        """
        return self.prompt_template | self.llm

    def categorize_item(self, item_args):
        """
        The `categorize_item` function categorizes an item based on input arguments and handles exceptions.

        :param item_args: The `item_args` parameter seems to be a dictionary that is being passed to the
        `categorize_item` method. It likely contains information or data related to an item that needs to be
        categorized
        :return: The `categorize_item` method returns the result of invoking the `chain` with the
        `item_args` provided. If the result is `None`, it raises a `ValueError` indicating that the chain
        returned `None` for the input. If an exception occurs during the processing of the item, it prints
        an error message with the item details and the exception message, then prints the traceback
        """
        try:
            # Update item_args with all prompt inputs
            item_args.update(self.prompt_inputs)
            print("ITEM ARGS - ", item_args)
            result = self.chain.invoke(item_args)
            if result is None:
                raise ValueError(f"Chain returned None for input: {item_args}")
            return result
        except Exception as e:
            print(f"Error processing item: {item_args.get('item', '')}")
            print(f"Exception: {e}")
            traceback.print_exc()
            return None

    def categorize_text_with_retries(
        self, text_to_categorize: str, max_retries: int = Config.MAX_RETRIES
    ):
        """
        The function `categorize_text_with_retries` categorizes text with retry logic to ensure a valid
        category is obtained.

        :param text_to_categorize: The `text_to_categorize` parameter is a string that represents the text
        that needs to be categorized. It is the input text that will be processed and categorized by the
        `categorize_text_with_retries` method
        :type text_to_categorize: str
        :param max_retries: The `max_retries` parameter in the `categorize_text_with_retries` method
        specifies the maximum number of retries allowed when attempting to categorize the text. If the
        categorization process fails to produce a valid category within the specified number of retries, the
        method will return "Uncategorized"
        :type max_retries: int
        :return: The `categorize_text_with_retries` method returns a tuple containing the `rationale` and
        `category` of the text after attempting to categorize it with retry logic. If a valid category is
        not obtained after the maximum number of retries, it sets the category to "Uncategorized".
        """
        # Common method to categorize text with retry logic
        retry_count = 0
        rationale = None
        category = None

        while retry_count < max_retries:
            item_args = {"item": text_to_categorize}
            content = self.categorize_item(item_args)
            if content is None:
                print(f"Warning: categorize_item() returned None for input: {text_to_categorize}")
            else:
                # Add this line
                print(f"Content returned from chain: {content}")
                text_content = self.check_content_type(content)
                rationale = self.extract_rationale(text_content)
                category = self.extract_category(text_content)
                # And this line
                print(f"Extracted rationale: {rationale}, category: {category}")
                if self._is_valid_category(category):
                    break
                else:
                    print(
                        f"Invalid category '{category}' received. Retrying ({retry_count + 1}/{max_retries})..."
                    )
            retry_count += 1

        if not self._is_valid_category(category):
            print(
                f"Failed to get a valid category for text: '{text_to_categorize}' after {max_retries} retries."
            )
            category = "Uncategorized"
        return rationale, category

    @staticmethod
    def extract_rationale(content: str):
        """
        This Python function extracts the rationale from a given content using a regular expression pattern.

        :param content: It looks like you have provided the code snippet for a function called
        `extract_rationale` that extracts the rationale from a given content string using a regular
        expression pattern. The rationale is expected to be found after the text "Rationale:" and before the
        text "Category:" or the end of the string
        :type content: str
        :return: The function `extract_rationale` returns the rationale extracted from the input `content`
        string based on the provided regex pattern. If a match is found, it returns the extracted rationale
        text after stripping any leading or trailing whitespaces. If no match is found, it returns `None`.
        """
        # Adjust the regex pattern to match the actual content
        # TODO consider having model output JSON and parse that JSON here
        rationale_pattern = r"Rationale:\s*(.*?)\s*(?:Category:|$)"
        rationale_match = re.search(rationale_pattern, content, re.DOTALL | re.IGNORECASE)
        rationale = rationale_match.group(1).strip() if rationale_match else None
        return rationale

    @staticmethod
    def extract_category(content: str):
        """
        The function `extract_category` takes a string input and extracts the category information following
        the "Category:" keyword.

        :param content: Thank you for providing the code snippet. It looks like you are trying to extract
        the category from a given content string using a regular expression pattern
        :type content: str
        :return: The function `extract_category` returns the category extracted from the input `content`
        string. If the string contains a pattern "Category: " followed by any characters, the function will
        extract and return those characters as the category. If the pattern is not found in the input
        string, the function will return `None`.
        """
        category_pattern = r"Category:\s*(.*)"
        category_match = re.search(category_pattern, content, re.IGNORECASE)
        category = category_match.group(1).strip() if category_match else None
        return category

    def _is_valid_category(self, category: str):
        """
        The function `_is_valid_category` checks if a given category is valid within a list of valid
        categories.

        :param category: The `_is_valid_category` method takes a parameter `category` of type `str`. It
        checks if the `category` is present in the `valid_categories` attribute of the class and returns a
        boolean value indicating whether the category is valid or not
        :type category: str
        :return: a boolean value indicating whether the input category is found in the list of valid
        categories stored in the object.
        """
        category_found = category in self.valid_categories
        return category_found

    def process(self):
        """
        The `process` function categorizes text data with retries and returns the results.
        :return: The `process` method is returning a list of tuples where each tuple contains the original
        text, the category assigned to that text, and the rationale for the categorization.
        """
        categorized_results = []

        index_list = self.categorzation_request.unique_ids
        text_list = self.categorzation_request.text_to_label
        total_list_length = len(text_list)
        progress_bar = st.progress(1, text="Operation in progress. Please wait...")

        for i, (idx, text) in enumerate(zip(index_list, text_list)):
            normalized_text = normalize_text(text)
            progress = (i + 1) / total_list_length
            progress_bar.progress(progress, text="Operation in progress. Please wait...")

            rationale, category = self.categorize_text_with_retries(normalized_text)
            categorized_results.append((idx, normalized_text, category, rationale))
        return categorized_results
