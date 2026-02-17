"""
Bot service for creating and managing chatbot instances.
"""

import os
from datetime import datetime
from typing import List, Dict
import mlflow

from opentelemetry import trace

from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✅ Modern
from Config.logger import logger
from Config.settings import settings
from Utills.chroma import chromadb_service

from Utills.file_utills import (
    extract_text_from_file,
    generate_bot_id,generate_namespace,cleanup_files
)

# Initialize tracer
tracer = trace.get_tracer("bot.service")

class BotService:
    """Service class for bot creation and management."""
    def __init__(self):
        """Initialize bot service."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len
        )

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
            logger.info(f"\n🤖 Creating bot '{bot_name}' for team '{team_name}'...")
            
            # Generate unique bot ID and namespace
            bot_id = generate_bot_id(team_name, bot_name)
            namespace = generate_namespace(bot_id)
            
            logger.info(f"   Bot ID: {bot_id}")
            logger.info(f"   Namespace: {namespace}")
            
            # Get current span to set attributes
            current_span = trace.get_current_span()
            if current_span.is_recording():
                current_span.set_attribute("mlflow.spanType", "CHAIN")
                current_span.set_attribute("mlflow.trace.metadata.bot_id", bot_id)
                current_span.set_attribute("mlflow.trace.metadata.team_name", team_name)
                current_span.set_attribute("mlflow.trace.metadata.bot_name", bot_name)
                current_span.set_attribute("mlflow.trace.metadata.num_files", len(file_paths))
                current_span.set_attribute("mlflow.trace.metadata.operation", "create_bot")
            
            # STEP 1: Extract text from files
            with tracer.start_as_current_span("extract_text") as otel_span:
                otel_span.set_attribute("mlflow.spanType", "PARSER")
                otel_span.set_attribute("num_files", len(file_paths))
                otel_span.set_attribute("file_names", [os.path.basename(fp) for fp in file_paths])
                
                all_text, file_names = self._extract_text_from_files(file_paths)
                
                otel_span.set_attribute("total_characters", len(all_text))
                otel_span.set_attribute("processed_files", len(file_names))
                otel_span.set_attribute("extraction_successful", True)
            
            logger.info(f"   ✅ Extracted {len(all_text)} characters from {len(file_names)} file(s)")
            
            # STEP 2: Split text into chunks
            with tracer.start_as_current_span("chunk_text") as otel_span:
                otel_span.set_attribute("mlflow.spanType", "PARSER")
                otel_span.set_attribute("chunk_size", settings.chunk_size)
                otel_span.set_attribute("chunk_overlap", settings.chunk_overlap)
                otel_span.set_attribute("text_length", len(all_text))
                
                chunks = self.text_splitter.split_text(all_text)
                
                otel_span.set_attribute("num_chunks", len(chunks))
                avg_chunk_size = sum(len(c) for c in chunks) // len(chunks) if chunks else 0
                otel_span.set_attribute("avg_chunk_size", avg_chunk_size)
            
            logger.info(f"   ✅ Created {len(chunks)} chunks")
            
            # STEP 3: Store vectors in ChromaDB
            # We don't need a wrapper span here as chroma_service.store_documents has its own OTEL span
            logger.info(f"   🔄 Storing vectors in ChromaDB collection for user '{bot_id}'...")
            chromadb_service.store_documents(texts=chunks, user_id=bot_id)
            
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
            
            logger.info(f"\n✅ Bot '{bot_name}' created successfully!")
            
            # Clean up uploaded files after processing
            cleanup_files(file_paths)
            
            return bot_metadata
            
        except Exception as e:
            # Clean up files if bot creation fails
            cleanup_files(file_paths)
            raise Exception(f"Error creating bot: {str(e)}")
    
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
                logger.info(f"   📄 Processing: {file_path}")
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
    
    def delete_bot(self, bot_id: str) -> None:
        """
        Delete a bot and its associated data.
        
        Args:
            bot_id: Unique bot identifier
            
        Raises:
            Exception: If deletion fails
        """
        try:
            with tracer.start_as_current_span("delete_bot") as span:
                span.set_attribute("bot.id", bot_id)
                chromadb_service.delete_user_collection(bot_id)
                logger.info(f"✅ Bot '{bot_id}' deleted successfully")
        except Exception as e:
            raise Exception(f"Error deleting bot: {str(e)}")
    
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