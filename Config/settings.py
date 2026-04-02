from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import ConfigDict,Field
from typing import List

# Load environment variables from .env
load_dotenv()
 
class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str
    PINECONE_API_KEY: str
    OPEN_API_KEY:str
    AZURE_OPENAI_API_VERSION:str
    AZURE_OPENAI_DEPLOYMENT:str
    AZURE_OPENAI_ENDPOINT:str
    ML_FLOW_TRACKING_URL:str
    ML_FLOW_EXPERIMENT_NAME:str

    # Pinecone index configuration
    INDEX_NAME: str
    DIMENSION: int

    # Embedding and LLM models
    EMBEDDING_MODEL: str
    GEMINI_MODEL: str
    TEMPERATURE: float

    ALLOWED_EXTENSIONS: List[str] = Field(
        default_factory=lambda: ["pdf", "docx", "txt"]
    )
    MAX_FILE_SIZE: int = 10485760
    UPLOAD_DIR:str 
    metric:str
    chunk_size:int
    chunk_overlap:int
    cloud:str
    region:str
    CHROMA_DB_PATH:str
    COLLECTION_NAME:str
    LOG_DIR:str
    openai_embedding_model: str
    api_version: str
    endpoint: str
    location: str
    key: str

    # Pydantic v2 way to configure BaseSettings
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

# Instantiate the settings object
settings = Settings()