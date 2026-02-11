"""
Pydantic schemas for request and response validation.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class BotCreateRequest(BaseModel):
    """Request schema for creating a new bot."""
    
    team_name: str = Field(..., min_length=1, max_length=100, description="Name of the team/organization")
    bot_name: str = Field(..., min_length=1, max_length=100, description="Name of the bot")
    
    @field_validator("team_name", "bot_name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that names don't contain special characters."""
        if not v.replace(" ", "").replace("-", "").replace("_", "").isalnum():
            raise ValueError("Name can only contain alphanumeric characters, spaces, hyphens, and underscores")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "team_name": "Engineering Team",
                "bot_name": "HR Assistant"
            }
        }


class BotCreateResponse(BaseModel):
    """Response schema for bot creation."""
    
    bot_id: str = Field(..., description="Unique identifier for the bot")
    team_name: str = Field(..., description="Team name")
    bot_name: str = Field(..., description="Bot name")
    namespace: str = Field(..., description="Pinecone namespace")
    created_at: datetime = Field(..., description="Creation timestamp")
    file_names: List[str] = Field(..., description="List of uploaded file names")
    chunk_count: int = Field(..., description="Number of text chunks created")
    total_characters: int = Field(..., description="Total characters processed")
    message: str = Field(default="Bot created successfully")
    
    class Config:
        json_schema_extra = {
            "example": {
                "bot_id": "abc123def456",
                "team_name": "Engineering Team",
                "bot_name": "HR Assistant",
                "namespace": "bot_abc123def456",
                "created_at": "2024-02-10T10:30:00",
                "file_names": ["policy.pdf"],
                "chunk_count": 45,
                "total_characters": 12500,
                "message": "Bot created successfully"
            }
        }


class ChatRequest(BaseModel):
    """Request schema for chatting with a bot."""
    
    bot_id: str = Field(..., min_length=1, description="Unique bot identifier")
    question: str = Field(..., min_length=1, max_length=2000, description="User's question")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation history")
    
    class Config:
        json_schema_extra = {
            "example": {
                "bot_id": "abc123def456",
                "question": "What is the vacation policy?",
                "session_id": "user_session_001"
            }
        }


class SourceDocument(BaseModel):
    """Schema for source document metadata."""
    
    page_content: str = Field(..., description="Content of the document chunk")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ChatResponse(BaseModel):
    """Response schema for chat interactions."""
    
    bot_id: str = Field(..., description="Bot identifier")
    question: str = Field(..., description="User's question")
    answer: str = Field(..., description="Bot's answer")
    source_documents: List[SourceDocument] = Field(default_factory=list, description="Source documents used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "bot_id": "abc123def456",
                "question": "What is the vacation policy?",
                "answer": "Employees are entitled to 15 days of PTO per year.",
                "source_documents": [],
                "timestamp": "2024-02-10T10:35:00"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Detailed error message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid file format",
                "timestamp": "2024-02-10T10:40:00"
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(default="healthy", description="Service health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-02-10T10:45:00"
            }
        }