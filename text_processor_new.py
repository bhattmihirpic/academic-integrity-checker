# text_processor.py

import os
import re
import io
from docx import Document  # For .docx files
from pdfminer.high_level import extract_text as pdf_extract_text  # Primary PDF extraction
from PyPDF2 import PdfReader  # Secondary PDF extraction
from pdf2image import convert_from_bytes  # For converting PDF pages to images
from PIL import Image  # For OCR image handling
import pytesseract  # For OCR
from werkzeug.datastructures import FileStorage

def extract_text_from_file_storage(file: FileStorage) -> str:
    """
    Extract text from a FileStorage object based on its extension.
    """
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.pdf':
        return read_pdf_from_storage(file)
    elif ext == '.docx':
        return read_word_from_storage(file)
    elif ext in ['.txt', '.text']:
        return read_text_from_storage(file)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def read_pdf_from_storage(file: FileStorage) -> str:
    """
    Robust PDF text extraction from FileStorage:
      1. Try pdfminer.six
      2. Fallback to PyPDF2
    """
    text = ""
    file_content = file.read()
    file.seek(0)  # Reset file pointer for potential reuse
    
    # 1. pdfminer.six extraction
    try:
        text = pdf_extract_text(io.BytesIO(file_content))
        print(f"[PDFMINER] Extracted {len(text)} chars from {file.filename}")
    except Exception as e:
        print(f"[PDFMINER ERROR] {e}")

    # 2. PyPDF2 fallback
    if not text.strip():
        try:
            reader = PdfReader(io.BytesIO(file_content))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text())
            text = "\n".join(pages)
            print(f"[PYPDF2] Extracted {len(text)} chars from {file.filename}")
        except Exception as e:
            print(f"[PYPDF2 ERROR] {e}")

    return text

def read_word_from_storage(file: FileStorage) -> str:
    """Read text from a .docx file stored in FileStorage"""
    doc = Document(io.BytesIO(file.read()))
    file.seek(0)  # Reset file pointer
    
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

def read_text_from_storage(file: FileStorage) -> str:
    """Read text from a .txt file stored in FileStorage"""
    return file.read().decode('utf-8')
