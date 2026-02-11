"""
Pinecone service for vector database operations.
"""
from typing import Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from Config.settings import settings
from Config.logger import logger


class PineconeService:
    """Service class for Pinecone vector database operations."""
    
    _instance: Optional['PineconeService'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Pinecone service (only once)."""
        if not self._initialized:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                google_api_key=settings.GEMINI_API_KEY
            )
            self._ensure_index_exists()
            self.index = self.pc.Index(settings.INDEX_NAME)
            PineconeService._initialized = True
    
    def _ensure_index_exists(self) -> None:
        """
        Ensure Pinecone index exists, create if it doesn't.
        
        Raises:
            Exception: If index creation fails
        """
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if settings.INDEX_NAME not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {settings.INDEX_NAME}")
                self.pc.create_index(
                    name=settings.INDEX_NAME,
                    dimension=settings.DIMENSION,
                    metric=settings.metric,
                    spec=ServerlessSpec(
                        cloud=settings.cloud,
                        region=settings.region
                    )
                )
                logger.info(f"✅ Index '{settings.INDEX_NAME}' created successfully")
            else:
                logger.info(f"✅ Index '{settings.INDEX_NAME}' already exists")
                
        except Exception as e:
            raise Exception(f"Error ensuring Pinecone index exists: {str(e)}")
    
    def get_index_stats(self) -> dict:
        """
        Get Pinecone index statistics.
        
        Returns:
            Dictionary containing index statistics
        """
        return self.index.describe_index_stats()
    
    def create_vectorstore(self, namespace: str) -> PineconeVectorStore:
        """
        Create a PineconeVectorStore instance for a specific namespace.
        
        Args:
            namespace: Pinecone namespace
            
        Returns:
            PineconeVectorStore instance
        """
        return PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="text",
            namespace=namespace
        )
    
    def store_documents(self, texts: list, namespace: str) -> PineconeVectorStore:
        """
        Store text chunks in Pinecone with namespace isolation.
        
        Args:
            texts: List of text chunks to store
            namespace: Pinecone namespace for isolation
            
        Returns:
            PineconeVectorStore instance
            
        Raises:
            Exception: If storage fails
        """
        try:
            vectorstore = PineconeVectorStore.from_texts(
                texts=texts,
                embedding=self.embeddings,
                index_name=settings.INDEX_NAME,
                namespace=namespace
            )
            return vectorstore
        except Exception as e:
            raise Exception(f"Error storing documents in Pinecone: {str(e)}")
    
    def delete_namespace(self, namespace: str) -> None:
        """
        Delete all vectors in a namespace.
        
        Args:
            namespace: Pinecone namespace to delete
            
        Raises:
            Exception: If deletion fails
        """
        try:
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"✅ Namespace '{namespace}' deleted successfully")
        except Exception as e:
            raise Exception(f"Error deleting namespace: {str(e)}")
    
    def namespace_exists(self, namespace: str) -> bool:
        """
        Check if a namespace has any vectors.
        
        Args:
            namespace: Pinecone namespace to check
            
        Returns:
            True if namespace has vectors, False otherwise
        """
        try:
            stats = self.index.describe_index_stats()
            namespaces = stats.get('namespaces', {})
            return namespace in namespaces and namespaces[namespace].get('vector_count', 0) > 0
        except Exception:
            return False


# Global Pinecone service instance
pinecone_service = PineconeService()