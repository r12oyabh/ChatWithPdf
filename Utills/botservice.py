"""
Bot service for creating and managing chatbot instances.
"""

import os
from datetime import datetime
from typing import List, Dict
import mlflow

from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✅ Modern
from Config.logger import logger
from Config.settings import settings
from Utills.chroma import chromadb_service
from langchain_text_splitters import MarkdownHeaderTextSplitter  # ✅ Modern
from langchain_text_splitters import RecursiveJsonSplitter

from Utills.file_utills import (
    extract_text_from_file,
    generate_bot_id,generate_namespace,cleanup_files
)

class BotService:
    """Service class for bot creation and management."""
    def __init__(self):
        """Initialize bot service."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len
        )

    @mlflow.trace(name="Create_Bot_Pipeline", span_type="CHAIN")
    def create_bot(self, team_name: str, bot_name: str, file_paths: List[str]) -> Dict:
        """
        Create a new bot with its own knowledge base using ChromaDB collection.
        Args:
            team_name: Name of the team/organization
            bot_name: Name of the bot
            file_paths: List of file paths to upload as knowledge base
        Returns:
            Dictionary containing bot metadata
        Raises:
            Exception: If bot creation fails
        """
        try:
            logger.info(f"\nCreating bot '{bot_name}' for team '{team_name}'...")
            
            # Generate unique bot ID and namespace
            bot_id = generate_bot_id(team_name, bot_name)
            namespace = generate_namespace(bot_id)
            
            logger.info(f"Bot ID: {bot_id}")
            logger.info(f"Namespace: {namespace}")
            
            # Update trace metadata
            mlflow.update_current_trace(
                metadata={
                    "bot_id": bot_id,
                    "team_name": team_name,
                    "bot_name": bot_name,
                    "num_files": len(file_paths),
                    "operation": "create_bot"
                }
            )
            
            # STEP 1: Extract text from files
            with mlflow.start_span(name="Extract_Text_From_Files", span_type="PARSER") as extract_span:
                extract_span.set_inputs({
                    "num_files": len(file_paths),
                    "file_names": [os.path.basename(fp) for fp in file_paths]
                })
                
                all_text, file_names = self._extract_text_from_files(file_paths)
                
                extract_span.set_outputs({
                    "total_characters": len(all_text),
                    "processed_files": len(file_names)
                })
                extract_span.set_attributes({
                    "extraction_successful": True,
                    "parser_type": "multi_format"
                })
            
            logger.info(f"Extracted {len(all_text)} characters from {len(file_names)} file(s)")
            
            # STEP 2: Split text into chunks
            with mlflow.start_span(name="Chunk_Documents", span_type="PARSER") as chunk_span:
                chunk_span.set_inputs({
                    "text_length": len(all_text),
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap
                })
                
                chunks = self.text_splitter.split_text(all_text)
                
                chunk_span.set_outputs({
                    "num_chunks": len(chunks),
                    "avg_chunk_size": sum(len(c) for c in chunks) // len(chunks) if chunks else 0
                })
                chunk_span.set_attributes({
                    "splitter_type": "RecursiveCharacterTextSplitter",
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap
                })
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # STEP 3: Store vectors in ChromaDB
            with mlflow.start_span(name="Store_In_VectorDB", span_type="EMBEDDING") as store_span:
                store_span.set_inputs({
                    "num_chunks": len(chunks),
                    "user_id": bot_id,
                    "vector_db": "ChromaDB"
                })
                
                logger.info(f"Storing vectors in ChromaDB collection for user '{bot_id}'...")
                chromadb_service.store_documents(texts=chunks, user_id=bot_id)
                
                store_span.set_outputs({
                    "success": True,
                    "stored_chunks": len(chunks)
                })
                store_span.set_attributes({
                    "collection_name": f"user_{bot_id}",
                    "embedding_model": settings.EMBEDDING_MODEL
                })
            
            # Create bot metadata
            bot_metadata = {
                'bot_id': bot_id,
                'team_name': team_name,
                'bot_name': bot_name,
                'namespace': namespace,
                'created_at': datetime.now(),
                'file_names': file_names,
                'chunk_count': len(chunks),
                'total_characters': len(all_text)
            }
            
            logger.info(f"Bot '{bot_name}' created successfully!")
            
            # Clean up uploaded files after processing
            cleanup_files(file_paths)
            
            return bot_metadata
            
        except Exception as e:
            # Clean up files if bot creation fails
            cleanup_files(file_paths)
            raise Exception(f"Error creating bot: {str(e)}")
    
    @mlflow.trace
    def _extract_text_from_files(self, file_paths: List[str]) -> tuple[str, List[str]]:
        """
        Extract text from multiple files.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Tuple of (combined_text, file_names)
            
        Raises:
            Exception: If text extraction fails
        """
        all_text = ""
        file_names = []
        
        for file_path in file_paths:
            try:
                logger.info(f"Processing: {file_path}")
                text = extract_text_from_file(file_path)
                
                # Add file separator for clarity
                filename = os.path.basename(file_path)
                all_text += f"\n\n--- Content from {filename} ---\n\n{text}"
                file_names.append(filename)
                
            except Exception as e:
                raise Exception(f"Error extracting text from {file_path}: {str(e)}")
        
        if not all_text.strip():
            raise Exception("No text content extracted from files")
        
        return all_text, file_names
    
    @mlflow.trace
    def delete_bot(self, bot_id: str) -> None:
        """
        Delete a bot and its associated data.
        
        Args:
            bot_id: Unique bot identifier
            
        Raises:
            Exception: If deletion fails
        """
        try:
            chromadb_service.delete_user_collection(bot_id)
            logger.info(f"Bot '{bot_id}' deleted successfully")
        except Exception as e:
            raise Exception(f"Error deleting bot: {str(e)}")
    
    @mlflow.trace
    def bot_exists(self, bot_id: str) -> bool:
        """
        Check if a bot exists.
        
        Args:
            bot_id: Unique bot identifier
            
        Returns:
            True if bot exists, False otherwise
        """
        return chromadb_service.user_collection_exists(bot_id)

# Global bot service instance
bot_service = BotService()