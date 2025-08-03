import fitz  # This helps read PDF files
from docx import Document  # This helps read Word documents
import os
import re

def extract_text_from_file(file_path):
    """Detect file type and dispatch to correct reader"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return read_pdf_file(file_path)
    elif ext == '.docx':
        return read_word_file(file_path)
    elif ext in ['.txt', '.text']:
        return read_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def read_pdf_file(file_path):
    """Read text from a PDF file using PyMuPDF (fitz)"""
    try:
        doc = fitz.open(file_path)  # ✅ Correct usage of PyMuPDF
        text = ""
        
        for page in doc:
            text += page.get_text()
            text += "\n"
        
        doc.close()
        return text.strip()
    except Exception as e:
        return f"[PDF ERROR] Unable to extract text: {str(e)}"

def read_word_file(file_path):
    """Read text from a Word (.docx) file"""
    try:
        doc = Document(file_path)
        text = []

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text)
        
        return '\n'.join(text)
    except Exception as e:
        return f"[DOCX ERROR] Unable to extract text: {str(e)}"

def read_text_file(file_path):
    """Read plain text from a .txt file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"[TXT ERROR] Unable to read text file: {str(e)}"

def clean_up_text(text):
    """Clean raw text: remove extra whitespace and unwanted characters"""
    if not text:
        return ""
    
    # Remove redundant whitespace (newlines, multiple spaces)
    text = re.sub(r'\s+', ' ', text)
    # Keep letters, numbers, and basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\(\)]', '', text)

    return text.strip()

def get_text_info(text):
    """Return character, word, sentence stats from cleaned text"""
    if not text:
        return {
            'total_characters': 0, 
            'total_words': 0, 
            'total_sentences': 0, 
            'average_words_per_sentence': 0
        }

    words = text.split()
    sentences = re.split(r'[.!?]', text)

    return {
        'total_characters': len(text),
        'total_words': len(words),
        'total_sentences': len([s for s in sentences if s.strip()]),
        'average_words_per_sentence': round(len(words) / max(len(sentences), 1), 2)
    }
