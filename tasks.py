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

logger = logging.getLogger(__name__)

# Create a Celery instance (adjust the broker URL as needed; here we assume Redis)
celery = Celery('tasks', broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'))

@celery.task
def process_balance_sheet(filename, upload_path, provider, categories):
    try:
        # 1. Extract text from PDF
        raw_pages = extract_text(upload_path)
        cleaned_pages = [clean_ocr_text(page, categories) for page in raw_pages]
        wrapped_json = wrap_pages_in_json(cleaned_pages)
        
        # 2. Analyze document using API
        total_text_length = sum(len(page) for page in cleaned_pages)
        if total_text_length > 10000:
            result, batch_logs = analyze_document_in_batches(wrapped_json, provider, categories, batch_size=10000, overlap=500)
        else:
            result = analyze_with_api(wrapped_json, provider, categories)
            batch_logs = "Single API call used (no batch processing)."
        
        # 3. Extract company info
        all_text = "\n".join(cleaned_pages)
        company_info = extract_company_info_from_text(all_text)
        company_name = company_info.get("company_name", "")
        cnpj = company_info.get("cnpj", "")
        
        # 4. Save or update database record (prevent duplicates)
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
            logger.info("Updated record for company %s", company_name)
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
            logger.info("Added new record for company %s", company_name)
        db.close()
        
        # Optionally, you could also save result files for download if needed.
        return {"record_id": record_id, "batch_logs": batch_logs, "wrapped_json": wrapped_json}
    except Exception as e:
        logger.exception("Error in processing balance sheet task: %s", e)
        return {"error": str(e)}