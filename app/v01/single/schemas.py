from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.v01.single.validators import validate_input_bytes


# TODO: Add this to the call
class Mode(str, Enum):
    evaluation = "Evaluation"
    production = "Production"


class Model(str, Enum):
    zero_shot = "Zero Shot"
    few_shot = "Few Shot"
    many_shot = "Many Shot"


class CategoriesList(BaseModel):
    name: str
    description: Optional[str] = None


class RequestInput(BaseModel):
    index_column: Optional[str] = None
    text_column: str
    ex_label_column: Optional[str] = None
    categories: List[CategoriesList]
    mode: Mode
    model: Optional[List[Model]] = None
    few_shot_count: Optional[str] = "1"
    many_shot_train_ratio: Optional[str] = "0.8"
    index_column: Optional[str] = None
    text_column: str
    ex_label_column: Optional[str] = None
    categories: List[CategoriesList]
    mode: Mode
    model: Optional[List[Model]] = None
    few_shot_count: Optional[str] = "1"
    many_shot_train_ratio: Optional[str] = "0.8"


class FileInRequest(BaseModel):
    file_encoded: str = Field(..., description="Base64-encoded CSV or XLSX content.")
    extension: str = Field(..., description="File extension which must be either '.csv' or '.xlsx'")

    @field_validator("file_encoded")
    @classmethod
    def check_mime_type(cls, v, values, **kwargs):
        # Your current logic to validate the base64 file
        return validate_input_bytes(cls, v, values)


class RequestCategorization(RequestInput, FileInRequest):
    """This class extends the RequestCategorization class to handle iteration requests."""

    pass
