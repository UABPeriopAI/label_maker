from typing import Optional

import pandas as pd


class ClassBalance:
    def __init__(self, df: pd.DataFrame, class_column: str) -> None:
        """
        Initializes the ClassBalance with a DataFrame and class column name.

        Parameters:
            df (pd.DataFrame): DataFrame containing the class column.
            class_column (str): Name of the class column.
        """
        self.df = df
        self.class_column = class_column
        self.balance_df: Optional[pd.DataFrame] = None

    def compute_balance(self) -> pd.DataFrame:
        """
        Computes class balance.

        Returns:
            pd.DataFrame: DataFrame containing counts and percentages of each class.
        """
        counts = self.df[self.class_column].value_counts()
        percentages = (counts / len(self.df)) * 100
        self.balance_df = pd.DataFrame({"Count": counts, "Percentage": percentages})
        return self.balance_df

    def display_balance(self) -> None:
        """
        Displays class balance.

        Raises:
            ValueError: If balance data has not been computed.
        """
        if self.balance_df is None:
            raise ValueError("Balance data not computed. Call compute_balance() first.")
        print(f"=== Class Balance for '{self.class_column}' ===")
        print(self.balance_df)
