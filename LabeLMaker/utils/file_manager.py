import os
import tempfile

import pandas as pd
from aiweb_common.file_operations.upload_manager import (
    FastAPIUploadManager,
    StreamlitUploadManager
    )
from LabeLMaker_config.config import Config

class FileManager:
    def __init__(self, azure_key=None):
        self.document_analysis_client = None
        if azure_key:
            try:
                self.document_analysis_client = Config.document_analysis_client(azure_key)
            except Exception as e:
                raise Exception(f"Failed to create Document Analysis Client: {e}")

    def process_file_upload(self, uploaded_file):
        # Process a single file using StreamlitUploadManager
        upload_manager = StreamlitUploadManager(
            uploaded_file, document_analysis_client=self.document_analysis_client
        )
        df, extension = upload_manager.process_upload()
        return df, extension

    def process_uploaded_file(self, background_tasks, uploaded_file, ext):
        # Process a single file using StreamlitUploadManager
        upload_manager = FastAPIUploadManager(
            background_tasks, document_analysis_client=self.document_analysis_client
        )
        df, _ = upload_manager.process_file_bytes(uploaded_file, ext)
        return df

    def process_multiple_files(self, uploaded_files):
        filenames, texts = [], []
        for file in uploaded_files:
            manager = StreamlitUploadManager(
                file,
                accept_multiple_files=False,
                document_analysis_client=self.document_analysis_client,
            )
            file_data, _ = manager.process_upload()
            if file_data:
                filenames.append(file.name)
                texts.append(file_data)
        return filenames, texts

    def write_excel_output(self, df):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmpfile:
            with pd.ExcelWriter(tmpfile.name, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Sheet1")
            with open(tmpfile.name, "rb") as file:
                excel_bytes = file.read()
            os.remove(tmpfile.name)
        return excel_bytes
