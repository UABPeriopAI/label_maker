# models/data_loader.py
from typing import List, Optional

import pandas as pd


class DataLoader:
    def __init__(
        self, file: Optional[str] = None, dataframe: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initializes the DataLoader with a file path or an existing DataFrame.

        Parameters:
            file (str, optional): Path to the CSV file to load.
            dataframe (pd.DataFrame, optional): An existing DataFrame.

        Raises:
            ValueError: If neither file nor dataframe is provided.
        """
        if file is not None:
            self.df = self.load_csv_file(file)
        elif dataframe is not None:
            self.df = dataframe
        else:
            raise ValueError("Either 'file' or 'dataframe' must be provided.")

    def load_csv_file(self, file: str) -> pd.DataFrame:
        """
        Loads a CSV file into a pandas DataFrame.

        Parameters:
            file (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        return pd.read_csv(file, encoding="utf-8")

    def preprocess_text_columns(self, columns: List[str]) -> "DataLoader":
        """
        Preprocesses text columns by stripping whitespace and converting to lowercase.

        Parameters:
            columns (List[str]): List of column names to preprocess.

        Returns:
            DataLoader: Returns self for method chaining.
        """
        for col in columns:
            self.df[col] = self.df[col].astype(str).str.strip().str.lower()
        return self

    def drop_duplicates(self, subset: List[str], keep: str = "first") -> "DataLoader":
        """
        Drops duplicate rows based on specified columns.

        Parameters:
            subset (List[str]): Columns to consider for identifying duplicates.
            keep (str, optional): Which duplicates to keep ('first', 'last', or False).

        Returns:
            DataLoader: Returns self for method chaining.
        """
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return self
