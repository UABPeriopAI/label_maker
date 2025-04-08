from typing import Any, List, Optional

from pydantic import BaseModel


class Categories(BaseModel):
    name: str
    description: Optional[str] = None


class Example(BaseModel):
    text_with_label: str
    label: str


class CategorizationRequest(BaseModel):
    unique_ids: Optional[List[str]] = None
    text_to_label: List[str]
    categories: List[Categories]
    examples: Optional[List[Example]] = None


# backend will parse these into two dataframes:
# 1. train_df (optional)
# 2. unlabeled_df (required)

# front end should always drop duplicates based on the "text..." column

# e.g., to slice out examples from a dataframe
# make a new dataframe with only the rows where "label" is not empty
# examples_df = examples_df[examples_df['category'].notna()]
# or use .dropna
# examples_df = examples_df.dropna(subset=['category'])
# in the end, this dataframe should have
# only the columns "text_to_categorize" and "category"
