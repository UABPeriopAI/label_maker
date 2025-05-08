"""
Configuration Module containing hardcoded varaibles. Config is a class for easy calling
"""

from pathlib import Path

from aiweb_common.WorkflowHandler import manage_sensitive
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from langchain_openai import OpenAI, OpenAIEmbeddings


class Config:
    """
    Configuration class for the Generative Categorizer application.

    This class centralizes all the configurable settings and file paths used throughout
    the application, and it is automatically included in the mkdocs web documentation.

    Attributes:
        Development Directories:
            - BASE_DIR (Path): The root directory of the project (two levels up from this file).
            - CONFIG_DIR (Path): Directory containing configuration files for the application.
            - LOGS_DIR (Path): Directory where application log files are stored.

        Data Directories:
            - DATA_DIR (Path): Root directory for data storage.
            - RAW_DATA (Path): Directory where raw input data is kept.
            - INTERMEDIATE_DIR (Path): Directory for intermediate data products during processing.
            - RESULTS_DIR (Path): Directory where output results are stored.

        Configurable Variables:
            - MAX_RETRIES (int): The maximum number of retry attempts for operations.
            - MAX_RECOMMENDED_GROUPS (int): The maximum number of recommended groups to display.
            - MIN_SAMPLES_MANY_SHOT (int): Minimum number of samples required for many-shot learning.
            - MIN_SAMPLES_FEW_SHOT (int): Minimum number of samples required for few-shot learning.
            - MIN_LOGISTIC_SAMPLES_PER_CLASS (int): Minimum number of samples required per class in
                logistic regression.

        Automated Evaluation Settings:
            - MANY_SHOT_TRAIN_RATIO (float): Ratio of training samples to be used in many-shot
                scenarios.
            - FEW_SHOT_COUNT (int): Number of examples per category for few-shot learning.

        Assets:
            - ASSETS_DIR (Path): Directory containing static assets (like prompt files).
            - ZS_PROMPTY (Path): File path for the zero-shot prompt template.
            - FS_PROMPTY (Path): File path for the few-shot prompt template.

        LLM Specific Settings:
            - OPENAI_COMPATIBLE_KEY: your OpenAI compatible API key here
            - OPENAI_COMPATIBLE_ENDPOINT: your OpenAI compatible endpoint here
            - LLM_INTERFACE (OpenAI): Configured OpenAI interface using the GPT-4 model.
            document_analysis_client (DocumentAnalysisClient): Configured Azure client for
                document analysis.
            - EMBEDDING_CLIENT (OpenAIEmbeddings): Configured OpenAI client for computing text
                embeddings.

        Page Rendering Configuration:
            - HEADER_MARKDOWN (str): Markdown template used as a header for rendered pages,
                                   including important notices and policy information.

    For additional details on configuration and usage, please refer to the mkdocs documentation.
    """

    OPENAI_COMPATIBLE_KEY = ""  # Enter your OpenAI compatible key here
    OPENAI_COMPATIBLE_ENDPOINT = ""  # Enter your OpenAI compatible endpoint here
    # Note: Can delete the azure keys if don't want (or care for) OCR on PDR upload.
    AZURE_DOCAI_COMPATIBLE_KEY = ""  # Enter your Azure DocAI compatible key here
    AZURE_DOCAI_COMPATIBLE_ENDPOINT = ""  # Enter your Azure DocAI compatible endpoint here

    # Development Directories
    BASE_DIR = Path(__file__).parent.parent.absolute()
    CONFIG_DIR = Path(BASE_DIR, "LabeLMaker_config")
    LOGS_DIR = Path(BASE_DIR, "logs")

    # Data Directories
    DATA_DIR = Path("/data/DATASCI")
    RAW_DATA = Path(DATA_DIR, "raw")
    INTERMEDIATE_DIR = Path(DATA_DIR, "intermediate")
    RESULTS_DIR = Path(DATA_DIR, "results")

    # Configurable variables
    MAX_RETRIES = 3
    MAX_RECOMMENDED_GROUPS = 10
    MIN_SAMPLES_MANY_SHOT = 25
    MIN_SAMPLES_FEW_SHOT = 1
    MIN_LOGISTIC_SAMPLES_PER_CLASS = 100
    # Automated selection for eval mode
    MANY_SHOT_TRAIN_RATIO = 0.8
    FEW_SHOT_COUNT = 2  # number of examples per category

    # Assets
    ASSETS_DIR = Path(BASE_DIR, "assets")
    ZS_PROMPTY = Path(ASSETS_DIR, "gencat_zeroshot.prompty")
    FS_PROMPTY = Path(ASSETS_DIR, "gencat_fewshot.prompty")

    # LLM specific
    LLM_INTERFACE = OpenAI(
        base_url=OPENAI_COMPATIBLE_ENDPOINT,
        model="gpt-4o-mini",
        api_key=OPENAI_COMPATIBLE_KEY,
    )

    document_analysis_client = DocumentAnalysisClient(
        endpoint=AZURE_DOCAI_COMPATIBLE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_DOCAI_COMPATIBLE_KEY),
    )

    EMBEDDING_CLIENT = OpenAIEmbeddings(
        api_key=OPENAI_COMPATIBLE_KEY,
        base_url=OPENAI_COMPATIBLE_ENDPOINT,
        model="text-embedding-3-small",
    )

    # Page rendering configuration (separate file?)

    HEADER_MARKDOWN = """
    
    ---                
                
    **Categorize data using AI**
    Brought to you by the Perioperative Data Science Team at UAB            
    _Not recommended for use with protected patient data_

    
    ---
                
    """
