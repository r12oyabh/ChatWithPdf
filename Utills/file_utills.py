"""
Utility functions for file processing and ID generation.
"""

import hashlib
import os
from datetime import datetime
from typing import List

import PyPDF2
from docx import Document
from fastapi import UploadFile, HTTPException

from Config.settings import settings
from Config.logger import logger
import mlflow

def generate_bot_id(team_name: str, bot_name: str) -> str:
    """
    Generate unique bot ID using team name, bot name, and timestamp.
    
    Args:
        team_name: Name of the team/organization
        bot_name: Name of the bot
        
    Returns:
        16-character hexadecimal bot ID
    """
    unique_string = f"{team_name}_{bot_name}_{datetime.now().isoformat()}"
    return hashlib.md5(unique_string.encode()).hexdigest()[:16]


def generate_namespace(bot_id: str) -> str:
    """
    Generate Pinecone namespace for a bot.
    
    Args:
        bot_id: Unique bot identifier
        
    Returns:
        Namespace string in format 'bot_{bot_id}'
    """
    return f"bot_{bot_id}"



def validate_file_extension(filename: str) -> bool:
    """
    Check if the file extension is allowed.
    Handles filenames with multiple dots and ignores spaces.
    """
    filename = filename.strip()  # remove leading/trailing spaces
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()  # split only on the last dot
    return ext in settings.ALLOWED_EXTENSIONS


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF file.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extracted text content
        
    Raises:
        Exception: If PDF cannot be read
    """
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        raise Exception(f"Error reading PDF file: {str(e)}")
    
    return text


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from DOCX file.
    
    Args:
        file_path: Path to DOCX file
        
    Returns:
        Extracted text content
        
    Raises:
        Exception: If DOCX cannot be read
    """
    try:
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        raise Exception(f"Error reading DOCX file: {str(e)}")
    
    return text


def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text from TXT file.
    
    Args:
        file_path: Path to TXT file
        
    Returns:
        Extracted text content
        
    Raises:
        Exception: If TXT cannot be read
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
        except Exception as e:
            raise Exception(f"Error reading TXT file: {str(e)}")
    except Exception as e:
        raise Exception(f"Error reading TXT file: {str(e)}")
    
    return text


def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from any supported file type.
    
    Args:
        file_path: Path to file
        
    Returns:
        Extracted text content
        
    Raises:
        ValueError: If file type is not supported
        Exception: If file cannot be read
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


async def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    """
    Save uploaded file to disk.
    
    Args:
        upload_file: FastAPI UploadFile object
        destination: Destination directory path
        
    Returns:
        Full path to saved file
        
    Raises:
        HTTPException: If file cannot be saved
    """
    try:
        # Create destination directory if it doesn't exist
        os.makedirs(destination, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{upload_file.filename}"
        file_path = os.path.join(destination, filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await upload_file.read()
            buffer.write(content)
        
        return file_path
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving file: {str(e)}"
        )


async def validate_and_save_files(files: List[UploadFile]) -> List[str]:
    """
    Validate and save multiple uploaded files.
    
    Args:
        files: List of FastAPI UploadFile objects
        
    Returns:
        List of saved file paths
        
    Raises:
        HTTPException: If validation fails or files cannot be saved
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    saved_paths = []
    
    try:
        for file in files:
            # Validate file extension
            if not validate_file_extension(file.filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {file.filename}. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
                )
            
            # Validate file size
            file.file.seek(0, 2)  # Seek to end
            file_size = file.file.tell()
            file.file.seek(0)  # Reset to beginning
            
            if file_size >settings.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large: {file.filename}. Max size: {settings.MAX_FILE_SIZE / (1024*1024)}MB"
                )           
            # Save file
            file_path = await save_upload_file(file, settings.UPLOAD_DIR)
            saved_paths.append(file_path)
        
        return saved_paths
        
    except HTTPException:
        # Clean up any saved files if validation fails
        for path in saved_paths:
            if os.path.exists(path):
                os.remove(path)
        raise
    except Exception as e:
        # Clean up any saved files if an error occurs
        for path in saved_paths:
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")


def cleanup_files(file_paths: List[str]) -> None:
    """
    Delete files from disk.
    
    Args:
        file_paths: List of file paths to delete
    """
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            # Log error but don't raise
            logger.info(f"Warning: Could not delete file {path}: {str(e)}")