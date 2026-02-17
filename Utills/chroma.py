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

from opentelemetry import trace, metrics


# Initialize tracer and meter
tracer = trace.get_tracer("chroma.service")
meter = metrics.get_meter("chroma.service")

# Metrics
doc_storage_counter = meter.create_counter(
    name="chroma_documents_stored_total",
    description="Total number of documents stored in ChromaDB",
    unit="1"
)
doc_retrieval_counter = meter.create_counter(
    name="chroma_retrievals_total",
    description="Total number of retrieval operations",
    unit="1"
)

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
    
    def create_vectorstore(self, user_id: str) -> Chroma:
        """
        Create a Chroma instance for a specific user.
        """
        doc_retrieval_counter.add(1, {"user_id": user_id})
        
        with tracer.start_as_current_span("chroma_create_vectorstore") as otel_span:
            otel_span.set_attribute("mlflow.spanType", "RETRIEVER")
            otel_span.set_attribute("user.id", user_id)
            
            collection_name = self._get_user_collection_name(user_id)
            self._ensure_collection_exists(collection_name)
            
            # Get collection stats for tracing
            stats = self.get_collection_stats(user_id)
            
            vectorstore = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            
            otel_span.set_attribute("collection.name", collection_name)
            otel_span.set_attribute("document.count", stats.get('count', 0))
            otel_span.set_attribute("vector_db_type", "ChromaDB")
        
        return vectorstore
    
    def store_documents(self, texts: list, user_id: str) -> Chroma:
        """
        Store text chunks in ChromaDB with user isolation.
        """
        doc_storage_counter.add(len(texts), {"user_id": user_id})
        
        with tracer.start_as_current_span("chroma_store_documents") as otel_span:
            otel_span.set_attribute("mlflow.spanType", "EMBEDDING")
            otel_span.set_attribute("user.id", user_id)
            otel_span.set_attribute("num_texts", len(texts))
            otel_span.set_attribute("total_characters", sum(len(text) for text in texts))

            try:
                collection_name = self._get_user_collection_name(user_id)
                self._ensure_collection_exists(collection_name)
                
                vectorstore = Chroma.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    client=self.client,
                    collection_name=collection_name
                )
                
                otel_span.set_attribute("status", "success")
                logger.info(f"✅ Stored {len(texts)} documents for user '{user_id}'")
                return vectorstore
            except Exception as e:
                otel_span.set_attribute("status", "error")
                otel_span.record_exception(e)
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