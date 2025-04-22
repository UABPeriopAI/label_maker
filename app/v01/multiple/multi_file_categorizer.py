from pathlib import Path
from typing import List
import base64

from aiweb_common.fastapi.schemas import JSONResponse
from aiweb_common.file_operations.upload_manager import FastAPIUploadManager
from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.fastapi_config import MULTIPLE_META
from app.v01.multiple.schemas import FileItem, RequestMultiFileCategorization
from LabeLMaker.multifile_categorize_handler import MultiFileFastAPICategorizeHandler
from LabeLMaker_config.config import Config

router = APIRouter(tags=["Multi"])

def get_multi_file_categorization_response(request: RequestMultiFileCategorization, files: List[FileItem], background_tasks: BackgroundTasks) -> JSONResponse:
    
    handler = MultiFileFastAPICategorizeHandler(azure_key=Config.AZURE_DOCAI_KEY)
    zs_prompty = Path(Config.ZS_PROMPTY)

    try:
        document_analysis_client = Config.DOCUMENT_ANALYSIS_CLIENT if hasattr(Config, "AZURE_DOCAI_KEY") else None
        upload_manager = FastAPIUploadManager(background_tasks=background_tasks, document_analysis_client=document_analysis_client)
        results = []

        for file in files:
            data = upload_manager.read_and_validate_file(file.file_encoded, file.extension)
            results.append((file.filename, data))

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {e}")

    try:

        # Split the aggregated results for the multi_file_categorization call
        filenames = [item[0] for item in results]
        texts = [item[1] for item in results]

        categories_dict = {cat.name: cat.description for cat in request.categories}

        categorized_results = handler.multifile_categorization(filenames=filenames, texts=texts, categories_dict=categories_dict, zs_prompty=zs_prompty)

        encoded_data = base64.b64encode(categorized_results.encode("utf-8"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error categorizing data: {e}")

    return JSONResponse(
        encoded_json=encoded_data,
        filename="AI_Generated_Categorization_multi.json",
        media_type="application/json",
    )


@router.post("/cv/v01/labels/multiple", **MULTIPLE_META)
async def process_multiple_files(
    background_tasks: BackgroundTasks, request: RequestMultiFileCategorization
) -> JSONResponse:
    response = get_multi_file_categorization_response(request, request.files, background_tasks)

    return response
