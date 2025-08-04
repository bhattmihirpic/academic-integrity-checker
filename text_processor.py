# text_processor.py

import os
import re
import io
from docx import Document  # For .docx files
from pdfminer.high_level import extract_text as pdf_extract_text  # Primary PDF extraction
from PyPDF2 import PdfReader  # Secondary PDF extraction
from pdf2image import convert_from_path  # For converting PDF pages to images
from PIL import Image  # For OCR image handling
import pytesseract  # For OCR

def extract_text_from_file(file_path: str) -> str:
    """
    Dispatch to the correct reader based on file extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return read_pdf_file(file_path)
    elif ext == '.docx':
        return read_word_file(file_path)
    elif ext in ['.txt', '.text']:
        return read_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def read_pdf_file(file_path: str) -> str:
    """
    Robust PDF text extraction:
      1. Try pdfminer.six
      2. Fallback to PyPDF2
      3. If still empty, use OCR via pytesseract
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    text = ""
    # 1. pdfminer.six extraction
    try:
        text = pdf_extract_text(file_path)
        print(f"[PDFMINER] Extracted {len(text)} chars from {os.path.basename(file_path)}")
    except Exception as e:
        print(f"[PDFMINER ERROR] {e}")

    # 2. PyPDF2 fallback
    if not text.strip():
        try:
            reader = PdfReader(file_path)
            pages = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)
            text = "\n".join(pages)
            print(f"[PyPDF2] Extracted {len(text)} chars from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"[PyPDF2 ERROR] {e}")

    # 3. OCR fallback
    if not text.strip():
        try:
            images = convert_from_path(file_path)
            ocr_pages = []
            for img in images:
                ocr_text = pytesseract.image_to_string(img)
                ocr_pages.append(ocr_text)
            text = "\n".join(ocr_pages)
            print(f"[OCR] Extracted {len(text)} chars via Tesseract from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"[OCR ERROR] {e}")

    # Normalize whitespace and remove empty lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized = "\n".join(lines)
    print(f"[PDF NORMALIZED] {len(normalized)} chars after cleanup")
    return normalized

def read_word_file(file_path: str) -> str:
    """
    Read text from a Word (.docx) file using python-docx.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        doc = Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        result = "\n".join(paragraphs)
        print(f"[DOCX] Extracted {len(result)} chars from {os.path.basename(file_path)}")
        return result
    except Exception as e:
        print(f"[DOCX ERROR] {e}")
        return ""

def read_text_file(file_path: str) -> str:
    """
    Read plain text from a .txt file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"[TXT] Extracted {len(content)} chars from {os.path.basename(file_path)}")
        return content
    except Exception as e:
        print(f"[TXT ERROR] {e}")
        return ""

def clean_up_text(text: str) -> str:
    """
    Clean raw text: collapse whitespace and remove unwanted characters.
    """
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.,!?;:\-\(\)]', '', text)
    return text.strip()

def get_text_info(text: str) -> dict:
    """
    Return character, word, and sentence statistics from cleaned text.
    """
    if not text:
        return {
            'total_characters': 0,
            'total_words': 0,
            'total_sentences': 0,
            'average_words_per_sentence': 0
        }
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    total_sentences = len(sentences)
    return {
        'total_characters': len(text),
        'total_words': len(words),
        'total_sentences': total_sentences,
        'average_words_per_sentence': round(len(words) / max(total_sentences, 1), 2)
    }
