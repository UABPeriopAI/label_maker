from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.v01.multiple.validators import validate_input_bytes

class CategoriesList(BaseModel):
    name: str
    description: Optional[str] = None


class RequestInput(BaseModel):
    categories: List[CategoriesList]


class FileItem(BaseModel):
    filename: str = Field(..., description="Original file name (e.g., 'data.pdf')")
    file_encoded: str = Field(..., description="Base64-encoded DOCX or PDF content.")
    extension: str = Field(..., description="File extension which must be either '.pdf' or '.docx'")

    @field_validator("file_encoded")
    @classmethod
    def check_mime_type(cls, v, values, **kwargs):
        # Your current logic to validate the base64 file
        return validate_input_bytes(cls, v, values)


class RequestMultiFileCategorization(RequestInput):
    """This class extends the RequestCategorization class to handle iteration requests."""

    files: List[FileItem] = Field(
        ...,
        description="List of DOCX/PDF files (each base64-encoded)."
    )