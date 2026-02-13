"""
API routes for chat interactions.
"""
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from Schema.schema import(ChatRequest,ChatResponse,ErrorResponse)
from Utills.chatservice import chat_service
from Utills.botservice import bot_service
from Config.logger import logger
from langchain.agents.middleware import PIIMiddleware

#ROUTING API ENDPOINT FOR THE CHAT  

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post(
    "/",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with a bot",
    description="""
    Send a question to a specific bot and receive an answer based on its knowledge base.
    
    - **bot_id**: Unique identifier of the bot (required)
    - **question**: The question to ask the bot (required)
    - **session_id**: Optional session ID for maintaining conversation history
    
    The bot will answer based only on the documents uploaded during its creation.
    If the answer is not found in the knowledge base, the bot will indicate this.
    """,
    responses={
        200: {
            "description": "Successful response from bot",
            "model": ChatResponse
        },
        400: {
            "description": "Bad request - invalid input",
            "model": ErrorResponse
        },
        404: {
            "description": "Bot not found",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    }
)
async def chat_with_bot(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    logger.info("Chat request received | bot_id=%s | session_id=%s",request.bot_id,request.session_id)
    
    """
    Chat with a specific bot.
    
    Args:
        request: ChatRequest containing bot_id, question, and optional session_id
        background_tasks: FastAPI background tasks
        
    Returns:
        ChatResponse with answer and source documents
        
    Raises:
        HTTPException: If bot not found or chat fails
    """
    try:
        # Validate bot exists
        if not bot_service.bot_exists(request.bot_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bot with ID '{request.bot_id}' not found"
            )
        
        # Get chat response
        response = chat_service.chat(
            bot_id=request.bot_id,
            question=request.question,
            session_id=request.session_id
        )

        # Add evaluation to background tasks
        background_tasks.add_task(
            chat_service.evaluate_chat_response,
            question=request.question,
            answer=response['answer'],
            source_documents=response['source_documents']
        )

        logger.info(
            "Chat response successful | bot_id=%s | session_id=%s (Evaluation scheduled in background)",
            request.bot_id,
            request.session_id,
        )

        return ChatResponse(**response)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during chat: {str(e)}"
        )


@router.delete(
    "/session/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Clear chat session",
    description="Clear the conversation history for a specific session.",
    responses={
        200: {"description": "Session cleared successfully"},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def clear_chat_session(session_id: str):
    logger.info("Clear session request received | session_id=%s", session_id)

    """
    Clear conversation history for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If clearing fails
    """
    try:
        chat_service.clear_session(session_id)

        logger.info("Session cleared successfully | session_id=%s", session_id)
        
        return {
            "message": f"Session '{session_id}' cleared successfully",
            "session_id": session_id
        }
        
    except Exception as e:
        logger.exception("Error clearing session | session_id=%s", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing session: {str(e)}"
        )


@router.get(
    "/sessions/count",
    status_code=status.HTTP_200_OK,
    summary="Get active sessions count",
    description="Get the number of currently active chat sessions.",
    responses={
        200: {"description": "Active sessions count"},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def get_active_sessions():
    logger.info("Active session count requested")
    """
    Get the number of active chat sessions.
    
    Returns:
        Dictionary with active session count
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        count = chat_service.get_session_count()
        logger.info("Active session count retrieved | count=%s", count)
        
        return {
            "active_sessions": count,
            "message": f"Currently {count} active session(s)"
        }
        
    except Exception as e:
        logger.exception("Error retrieving active session count")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting session count: {str(e)}"
        )