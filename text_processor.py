import fitz  # This helps read PDF files
from docx import Document  # This helps read Word documents
import os

def extract_text_from_file(file_path):
    """This function takes a file and pulls out all the text from it"""
    
    # First, let's figure out what type of file it is
    if file_path.lower().endswith('.pdf'):
        return read_pdf_file(file_path)
    elif file_path.lower().endswith('.docx'):
        return read_word_file(file_path)
    else:
        return read_text_file(file_path)

def read_pdf_file(file_path):
    """Read text from a PDF file"""
    try:
        doc = PyMuPDF.open(file_path)
        text = ""
        
        # Go through each page and get the text
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
            text += "\n"  # Add a line break between pages
        
        doc.close()
        return text.strip()
    except Exception as e:
        return f"Sorry, I couldn't read this PDF file: {str(e)}"

def read_word_file(file_path):
    """Read text from a Word document"""
    try:
        doc = Document(file_path)
        text = []
        
        # Go through each paragraph
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # Only add paragraphs that have text
                text.append(paragraph.text)
        
        return '\n'.join(text)
    except Exception as e:
        return f"Sorry, I couldn't read this Word document: {str(e)}"

def read_text_file(file_path):
    """Read text from a simple text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Sorry, I couldn't read this text file: {str(e)}"

def clean_up_text(text):
    """Make the text easier to analyze by cleaning it up"""
    import re
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove weird characters but keep important punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    return text.strip()

def get_text_info(text):
    """Get some basic information about the text"""
    if not text:
        return {}
    
    words = text.split()
    sentences = text.split('.')
    
    return {
        'total_characters': len(text),
        'total_words': len(words),
        'total_sentences': len([s for s in sentences if s.strip()]),
        'average_words_per_sentence': len(words) / max(len(sentences), 1)
    }
