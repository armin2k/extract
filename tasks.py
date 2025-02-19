# tasks.py
import os
import json
import logging
import math
from celery import Celery
from pdf_processor import extract_text
from ocr_utils import clean_ocr_text, wrap_pages_in_json, extract_company_info_from_text
from api_integration import analyze_with_api, analyze_document_in_batches
from db import SessionLocal
from models import BalanceSheet

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Create a Celery instance with a broker and result backend.
celery = Celery(
    'tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

@celery.task(bind=True)
def process_balance_sheet(self, filename, upload_path, provider, categories):
    try:
        # 1. Extract text from PDF using pdf_processor
        raw_pages = extract_text(upload_path)
        cleaned_pages = [clean_ocr_text(page, categories) for page in raw_pages]
        wrapped_json = wrap_pages_in_json(cleaned_pages)
        
        # 2. Analyze the document using API
        total_text_length = sum(len(page) for page in cleaned_pages)
        if total_text_length > 10000:
            result, batch_logs = analyze_document_in_batches(wrapped_json, provider, categories, batch_size=10000, overlap=500)
        else:
            result = analyze_with_api(wrapped_json, provider, categories)
            batch_logs = "Single API call used (no batch processing)."
        
        # 3. Extract company information from OCR text
        all_text = "\n".join(cleaned_pages)
        company_info = extract_company_info_from_text(all_text)
        company_name = company_info.get("company_name", "")
        cnpj = company_info.get("cnpj", "")
        logger.info("Extracted company info: %s", company_info)
        
        # 4. Save (or update) the analysis result in the database
        db = SessionLocal()
        existing_sheet = None
        if company_name or cnpj:
            existing_sheet = db.query(BalanceSheet).filter(
                BalanceSheet.company_name == company_name,
                BalanceSheet.cnpj == cnpj
            ).first()
        
        if existing_sheet:
            existing_sheet.data = json.dumps(result)
            existing_sheet.filename = filename
            db.commit()
            record_id = existing_sheet.id
            logger.info("Updated record for company %s.", company_name)
        else:
            new_sheet = BalanceSheet(
                filename=filename,
                data=json.dumps(result),
                company_name=company_name,
                cnpj=cnpj
            )
            db.add(new_sheet)
            db.commit()
            record_id = new_sheet.id
            logger.info("Added new record for company %s.", company_name)
        db.close()
        
        # Optionally, update the task state with processing details
        self.update_state(state="SUCCESS", meta={"record_id": record_id, "status": "Processing complete!", "current": 1, "total": 1, "processing_time": 1})
        return {"record_id": record_id, "meta": {"status": "Processing complete!", "current": 1, "total": 1, "processing_time": 1}}
    except Exception as e:
        logger.exception("Error in processing balance sheet task: %s", e)
        self.update_state(state="FAILURE", meta={"status": str(e)})
        return {"error": str(e)}