from aiweb_common.file_operations.file_handling import (
    create_base64_file_validator,
)

import app.fastapi_config as form_api_config

validate_input_bytes = create_base64_file_validator(
    form_api_config.DOCX_EXPECTED_TYPE,
    form_api_config.PDF_EXPECTED_TYPE,
)
