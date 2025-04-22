import pandas as pd
from aiweb_common.fastapi.schemas import MSWordResponse
from aiweb_common.file_operations.upload_manager import FastAPIUploadManager
from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.fastapi_config import EVALUATE_META
from app.v01.evaluate.schemas import RequestEvaluation
from LabeLMaker_config.config import Config
from LabeLMaker.evaluate_handler import FastAPIEvaluateHandler

router = APIRouter(tags=["Evaluate"])


def get_evaluator_response(
    request: RequestEvaluation, file_encoded: str, background_tasks: BackgroundTasks
) -> MSWordResponse:
    
    handler = FastAPIEvaluateHandler(azure_key=Config.AZURE_DOCAI_KEY)

    try:
        upload_manager = FastAPIUploadManager(background_tasks=background_tasks)
        data = upload_manager.read_and_validate_file(file_encoded, request.extension)

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Uploaded file did not result in a valid DataFrame.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {e}")

    try:
        encoded_file = handler.fastapi_evaluate(data, request, background_tasks)
        response =  MSWordResponse(encoded_docx=encoded_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error categorizing data: {e}")

    return response


@router.post("/cv/v01/labels/evaluate", **EVALUATE_META)
async def process_evaluate(
    background_tasks: BackgroundTasks, request: RequestEvaluation
) -> MSWordResponse:
    response = get_evaluator_response(request, request.file_encoded, background_tasks)

    return response
