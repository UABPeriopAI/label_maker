from LabeLMaker_config.config import Config
from typing import Optional
from LabeLMaker.Categorize.zeroshot import ZeroShotCategorizer
from LabeLMaker.utils.category import CategoryManager
import json

class BaseMultiFileCategorizeHandler:
    """
    Abstract base class that implements categorization logic.
    """

    def __init__(self, azure_key: Optional[str] = None) -> None:
        self.config = Config  # Expose configuration constants.
        self.azure_key = azure_key

    def multifile_categorization(self, filenames, texts, categories_dict, zs_prompty=None):
        categorization_request = CategoryManager.create_request(filenames, texts, categories_dict)
        zero_shot_categorizer = ZeroShotCategorizer(prompty_path=zs_prompty, category_request=categorization_request)
        categorized_results = zero_shot_categorizer.process()
        #self.display_results(categorized_results)
        
        return json.dumps(categorized_results)


class MultiFileStreamlitCategorizeHandler(BaseMultiFileCategorizeHandler):
    """
    Thin wrapper for Streamlit usage.
    Expects that UI parameters (collected via the UI) are passed in a dictionary.
    """

    def __init__(self, azure_key: Optional[str] = None) -> None:
        super().__init__(azure_key=azure_key)
        


class MultiFileFastAPICategorizeHandler(BaseMultiFileCategorizeHandler):
    """
    Provides categorization functionality for FastAPI endpoints.
    """

    def __init__(self, azure_key: Optional[str] = None) -> None:
        super().__init__(azure_key=azure_key)