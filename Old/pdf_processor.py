import os
import logging
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter

load_dotenv()

POPPLER_PATH = os.getenv("POPPLER_PATH", "/opt/homebrew/bin")
SCALE_FACTOR = int(os.getenv("SCALE_FACTOR", 1))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def preprocess_image(img: Image.Image) -> Image.Image:
    """Convert image to grayscale, enhance contrast, and apply a median filter."""
    gray = img.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2)
    gray = gray.filter(ImageFilter.MedianFilter())
    return gray

def post_process_text(text: str) -> str:
    """Apply basic post-processing to OCR text."""
    return text

def extract_text(pdf_path: str) -> str:
    """Extract text from a PDF using PyPDF2; if that fails, fallback to OCR."""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            if len(text.strip()) > 100:
                return text
    except Exception as e:
        logger.error(f"Standard extraction failed: {e}")
    try:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=300)
        ocr_text = []
        custom_config = "--psm 6 --oem 3"
        for img in images:
            processed_img = preprocess_image(img)
            text_img = pytesseract.image_to_string(processed_img, lang='por', config=custom_config)
            text_img = post_process_text(text_img)
            ocr_text.append(text_img)
        return "\n".join(ocr_text)
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return ""