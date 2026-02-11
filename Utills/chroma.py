"""
ChromaDBService with collection-based user isolation (similar to Pinecone namespaces).
"""
from typing import Optional
import chromadb
import mlflow
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from Config.settings import settings
from Config.logger import logger


class ChromaDBService:
    """Service class for ChromaDB vector database operations."""
    
    _instance: Optional['ChromaDBService'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize ChromaDB service (only once)."""
        if not self._initialized:
            self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                google_api_key=settings.GEMINI_API_KEY
            )
            ChromaDBService._initialized = True
    
    def _get_user_collection_name(self, user_id: str) -> str:
        """
        Generate collection name for a user (equivalent to Pinecone namespace).
        
        Args:
            user_id: User identifier
            
        Returns:
            Collection name for this user
        """
        return f"user_{user_id}"
    
    def _ensure_collection_exists(self, collection_name: str) -> None:
        """
        Ensure ChromaDB collection exists, create if it doesn't.
        
        Args:
            collection_name: Name of the collection to ensure exists
            
        Raises:
            Exception: If collection creation fails
        """
        try:
            existing_collections = [col.name for col in self.client.list_collections()]
            
            if collection_name not in existing_collections:
                logger.info(f"Creating new ChromaDB collection: {collection_name}")
                self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": settings.metric}
                )
                logger.info(f"✅ Collection '{collection_name}' created successfully")
            else:
                logger.info(f"✅ Collection '{collection_name}' already exists")
                
        except Exception as e:
            raise Exception(f"Error ensuring ChromaDB collection exists: {str(e)}")
    
    def get_collection_stats(self, user_id: str) -> dict:
        """
        Get ChromaDB collection statistics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing collection statistics
        """
        collection_name = self._get_user_collection_name(user_id)
        try:
            collection = self.client.get_collection(name=collection_name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception:
            return {"name": collection_name, "count": 0, "metadata": {}}
    
    @mlflow.trace(name="ChromaDB.create_vectorstore", span_type="RETRIEVER")
    def create_vectorstore(self, user_id: str) -> Chroma:
        """
        Create a Chroma instance for a specific user.
        
        Args:
            user_id: User identifier (equivalent to Pinecone namespace)
            
        Returns:
            Chroma instance
        """
        collection_name = self._get_user_collection_name(user_id)
        self._ensure_collection_exists(collection_name)
        
        # Get collection stats for tracing
        stats = self.get_collection_stats(user_id)
        
        vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        # Add metadata to current span for visibility
        with mlflow.start_span(name="Collection_Stats", span_type="RETRIEVER") as span:
            span.set_inputs({"user_id": user_id})
            span.set_outputs({
                "collection_name": collection_name,
                "document_count": stats.get('count', 0),
                "vector_db_type": "ChromaDB"
            })
            span.set_attributes({
                "embedding_model": settings.EMBEDDING_MODEL,
                "embedding_dimension": 768,
                "metric": settings.metric
            })
        
        return vectorstore
    
    @mlflow.trace(name="ChromaDB.store_documents", span_type="EMBEDDING")
    def store_documents(self, texts: list, user_id: str) -> Chroma:
        """
        Store text chunks in ChromaDB with user isolation.
        
        Args:
            texts: List of text chunks to store
            user_id: User identifier (equivalent to Pinecone namespace)
            
        Returns:
            Chroma instance
            
        Raises:
            Exception: If storage fails
        """
        try:
            collection_name = self._get_user_collection_name(user_id)
            self._ensure_collection_exists(collection_name)
            
            # Add embedding span for visibility
            with mlflow.start_span(name="Create_Embeddings", span_type="EMBEDDING") as embed_span:
                embed_span.set_inputs({
                    "num_texts": len(texts),
                    "user_id": user_id,
                    "total_characters": sum(len(text) for text in texts)
                })
                
                vectorstore = Chroma.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    client=self.client,
                    collection_name=collection_name
                )
                
                embed_span.set_outputs({
                    "success": True,
                    "documents_stored": len(texts)
                })
                embed_span.set_attributes({
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "collection_name": collection_name,
                    "vector_db": "ChromaDB"
                })
            
            logger.info(f"✅ Stored {len(texts)} documents for user '{user_id}'")
            return vectorstore
        except Exception as e:
            raise Exception(f"Error storing documents in ChromaDB: {str(e)}")
    
    def delete_user_collection(self, user_id: str) -> None:
        """
        Delete all vectors for a user (equivalent to delete_namespace).
        
        Args:
            user_id: User identifier
            
        Raises:
            Exception: If deletion fails
        """
        try:
            collection_name = self._get_user_collection_name(user_id)
            self.client.delete_collection(name=collection_name)
            logger.info(f"✅ Collection '{collection_name}' deleted successfully")
        except Exception as e:
            raise Exception(f"Error deleting collection: {str(e)}")
    
    def user_collection_exists(self, user_id: str) -> bool:
        """
        Check if a user has any vectors (equivalent to namespace_exists).
        
        Args:
            user_id: User identifier
            
        Returns:
            True if user collection has vectors, False otherwise
        """
        try:
            collection_name = self._get_user_collection_name(user_id)
            existing_collections = [col.name for col in self.client.list_collections()]
            
            if collection_name not in existing_collections:
                return False
            
            collection = self.client.get_collection(name=collection_name)
            return collection.count() > 0
        except Exception:
            return False


# Global ChromaDB service instance
chromadb_service = ChromaDBService()