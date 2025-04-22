from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import List

from pydantic import BaseModel
from app.v01.single.validators import validate_input_bytes


class Model(str, Enum):
    zero_shot = "Zero Shot"
    few_shot = "Few Shot"
    many_shot = "Many Shot"

class RequestInput(BaseModel):
    models: List[Model]
    ground_truth: str

class FileInRequest(BaseModel):
    file_encoded: str = Field(...,
        description="Base64-encoded CSV or XLSX content."
    )
    extension: str = Field(...,
        description="File extension which must be either '.csv' or '.xlsx'"
    )
    
    @field_validator("file_encoded")
    @classmethod
    def check_mime_type(cls, v, values, **kwargs):
        # Your current logic to validate the base64 file
        return validate_input_bytes(cls, v, values)
    
class RequestEvaluation(RequestInput, FileInRequest):
    """This class extends the RequestEvaluation class to handle iteration requests."""
    pass

