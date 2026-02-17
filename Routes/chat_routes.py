# """
# API routes for chat interactions.
# """
# from fastapi import APIRouter, HTTPException, status, BackgroundTasks
# from Schema.schema import(ChatRequest,ChatResponse,ErrorResponse)
# from Utills.chatservice import chat_service
# from Utills.botservice import bot_service
# from Config.logger import logger
# from langchain.agents.middleware import PIIMiddleware

# #ROUTING API ENDPOINT FOR THE CHAT  

# from opentelemetry import metrics

# router = APIRouter(prefix="/chat", tags=["Chat"])

# # Initialize Meter and Counter
# meter = metrics.get_meter("chat.routes")
# chat_counter = meter.create_counter(
#     name="chat_requests_total",
#     description="Total number of chat requests",
#     unit="1"
# )

# @router.post(
#     "/",
#     response_model=ChatResponse,
#     status_code=status.HTTP_200_OK,
#     summary="Chat with a bot",
#     description="""
#     Send a question to a specific bot and receive an answer based on its knowledge base.
    
#     - **bot_id**: Unique identifier of the bot (required)
#     - **question**: The question to ask the bot (required)
#     - **session_id**: Optional session ID for maintaining conversation history
    
#     The bot will answer based only on the documents uploaded during its creation.
#     If the answer is not found in the knowledge base, the bot will indicate this.
#     """,
#     responses={
#         200: {
#             "description": "Successful response from bot",
#             "model": ChatResponse
#         },
#         400: {
#             "description": "Bad request - invalid input",
#             "model": ErrorResponse
#         },
#         404: {
#             "description": "Bot not found",
#             "model": ErrorResponse
#         },
#         500: {
#             "description": "Internal server error",
#             "model": ErrorResponse
#         }
#     }
# )
# async def chat_with_bot(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
#     logger.info("Chat request received | bot_id=%s | session_id=%s",request.bot_id,request.session_id)
    
#     """
#     Chat with a specific bot.
    
#     Args:
#         request: ChatRequest containing bot_id, question, and optional session_id
#         background_tasks: FastAPI background tasks
        
#     Returns:
#         ChatResponse with answer and source documents
        
#     Raises:
#         HTTPException: If bot not found or chat fails
#     """
#     try:
#         # Validate bot exists
#         if not bot_service.bot_exists(request.bot_id):
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Bot with ID '{request.bot_id}' not found"
#             )
        
#         # Get chat response
#         response = chat_service.chat(
#             bot_id=request.bot_id,
#             question=request.question,
#             session_id=request.session_id
#         )

#         # Add evaluation to background tasks
#         background_tasks.add_task(
#             chat_service.evaluate_chat_response,
#             question=request.question,
#             answer=response['answer'],
#             source_documents=response['source_documents']
#         )

#         # Increment metrics counter
#         chat_counter.add(1, {"bot_id": request.bot_id})

#         logger.info(
#             "Chat response successful | bot_id=%s | session_id=%s (Evaluation scheduled in background)",
#             request.bot_id,
#             request.session_id,
#         )

#         return ChatResponse(**response)
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error during chat: {str(e)}"
#         )


# @router.delete(
#     "/session/{session_id}",
#     status_code=status.HTTP_200_OK,
#     summary="Clear chat session",
#     description="Clear the conversation history for a specific session.",
#     responses={
#         200: {"description": "Session cleared successfully"},
#         500: {"description": "Internal server error", "model": ErrorResponse}
#     }
# )
# async def clear_chat_session(session_id: str):
#     logger.info("Clear session request received | session_id=%s", session_id)

#     """
#     Clear conversation history for a session.
    
#     Args:
#         session_id: Session identifier
        
#     Returns:
#         Success message
        
#     Raises:
#         HTTPException: If clearing fails
#     """
#     try:
#         chat_service.clear_session(session_id)

#         logger.info("Session cleared successfully | session_id=%s", session_id)
        
#         return {
#             "message": f"Session '{session_id}' cleared successfully",
#             "session_id": session_id
#         }
        
#     except Exception as e:
#         logger.exception("Error clearing session | session_id=%s", session_id)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error clearing session: {str(e)}"
#         )


# @router.get(
#     "/sessions/count",
#     status_code=status.HTTP_200_OK,
#     summary="Get active sessions count",
#     description="Get the number of currently active chat sessions.",
#     responses={
#         200: {"description": "Active sessions count"},
#         500: {"description": "Internal server error", "model": ErrorResponse}
#     }
# )
# async def get_active_sessions():
#     logger.info("Active session count requested")
#     """
#     Get the number of active chat sessions.
    
#     Returns:
#         Dictionary with active session count
        
#     Raises:
#         HTTPException: If retrieval fails
#     """
#     try:
#         count = chat_service.get_session_count()
#         logger.info("Active session count retrieved | count=%s", count)
        
#         return {
#             "active_sessions": count,
#             "message": f"Currently {count} active session(s)"
#         }
        
#     except Exception as e:
#         logger.exception("Error retrieving active session count")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error getting session count: {str(e)}"
#         )


###############################

"""
API routes for chat interactions with full OpenTelemetry telemetry.
"""
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from Schema.schema import ChatRequest, ChatResponse, ErrorResponse
from Utills.chatservice import chat_service
from Utills.botservice import bot_service
from Config.logger import logger
from langchain.agents.middleware import PIIMiddleware
import time

# OpenTelemetry
from opentelemetry import trace, metrics
import json
from datetime import datetime

# Helper for JSON serialization of datetime
def json_serial(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

router = APIRouter(prefix="/chat", tags=["Chat"])

# Initialize tracer and meter
tracer = trace.get_tracer("chat.routes")
meter = metrics.get_meter("chat.routes")

# Counters
chat_counter = meter.create_counter(
    name="chat_requests_total",
    description="Total number of chat requests",
    unit="1"
)
chat_error_counter = meter.create_counter(
    name="chat_errors_total",
    description="Total number of failed chat requests",
    unit="1"
)

# Histogram for latency
chat_latency = meter.create_histogram(
    name="chat_latency_seconds",
    description="Latency of chat endpoint",
    unit="s"
)

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
""",
    responses={
        200: {"description": "Successful response from bot", "model": ChatResponse},
        400: {"description": "Bad request - invalid input", "model": ErrorResponse},
        404: {"description": "Bot not found", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    }
)
async def chat_with_bot(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    """
    Chat with a specific bot with OpenTelemetry metrics and tracing.
    """
    start_time = time.time()
    chat_counter.add(1, {"bot_id": request.bot_id})

    # Get the root span created by FastAPIInstrumentor
    root_span = trace.get_current_span()
    if root_span.is_recording():
        root_span.set_attribute("mlflow.traceName", f"Chat: {request.bot_id}")
        root_span.set_attribute("mlflow.trace.metadata.bot_id", request.bot_id)
        root_span.set_attribute("mlflow.trace.metadata.session_id", request.session_id or "none")
        root_span.set_attribute("mlflow.trace.metadata.question", request.question)
        root_span.set_attribute("mlflow.trace.inputs", json.dumps(request.model_dump(), default=json_serial))
        
        # Custom Metadata (Verbose for User Request)
        root_span.set_attribute("mlflow.trace.metadata.base_vector", "ChromaDB")
        root_span.set_attribute("mlflow.trace.metadata.vectordbname", "ChromaDB")
        root_span.set_attribute("mlflow.trace.metadata.retriever_k", 3)
        root_span.set_attribute("mlflow.trace.metadata.retreiver_k", 3)
        root_span.set_attribute("mlflow.trace.metadata.operation", "chat")
        root_span.set_attribute("mlflow.trace.metadata.operations", "chat")
        root_span.set_attribute("mlflow.trace.metadata.prompt", request.question)
        root_span.set_attribute("mlflow.trace.metadata.question", request.question)
        root_span.set_attribute("mlflow.trace.metadata.bot_id", request.bot_id)
        root_span.set_attribute("mlflow.trace.metadata.bot_name", request.bot_id) # Fallback to ID
        root_span.set_attribute("mlflow.trace.metadata.team_name", "unknown")
        root_span.set_attribute("mlflow.trace.metadata.Session", request.session_id or "none")
        root_span.set_attribute("mlflow.trace.metadata.User", request.session_id or "guest")
        root_span.set_attribute("mlflow.trace.metadata.Run_name", f"Chat-{request.bot_id}")

        # MLflow UI Standard Tags/Columns
        root_span.set_attribute("mlflow.user", request.session_id or "guest")
        root_span.set_attribute("mlflow.runName", f"Chat-{request.bot_id}")
        root_span.set_attribute("mlflow.tag.Session", request.session_id or "none")
        root_span.set_attribute("mlflow.tag.User", request.session_id or "guest")
        root_span.set_attribute("mlflow.tag.Run_name", f"Chat-{request.bot_id}")
        root_span.set_attribute("mlflow.tag.team_name", "unknown")
        root_span.set_attribute("mlflow.tag.bot_id", request.bot_id)

    try:
        if not bot_service.bot_exists(request.bot_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bot with ID '{request.bot_id}' not found"
            )

        with tracer.start_as_current_span("chat_pipeline") as chat_span:
            chat_span.set_attribute("mlflow.traceName", f"Chat with {request.bot_id}")
            chat_span.set_attribute("mlflow.trace.metadata.bot_id", request.bot_id)
            chat_span.set_attribute("mlflow.trace.metadata.session_id", request.session_id or "none")
            chat_span.set_attribute("mlflow.trace.metadata.question", request.question)
            chat_span.set_attribute("mlflow.trace.inputs", request.model_dump_json())
            
            chat_span.set_attribute("bot.id", request.bot_id)
            chat_span.set_attribute("session.id", request.session_id or "none")
            
            # Retrieval span
            with tracer.start_as_current_span("retrieval") as retrieval_span:
                docs = chat_service.retrieve_documents(request.bot_id, request.question)
                retrieval_span.set_attribute("retrieved.docs.count", len(docs))

            # Generation span
            with tracer.start_as_current_span("generation") as generation_span:
                response = chat_service.chat(
                    bot_id=request.bot_id,
                    question=request.question,
                    session_id=request.session_id,
                    context=docs
                )
                generation_span.set_attribute("llm.tokens", response.get("token_count", 0))
                generation_span.set_attribute("llm.model", response.get("model_name", "unknown"))
                
                chat_span.set_attribute("mlflow.trace.outputs", json.dumps(response, default=json_serial))
                chat_span.set_attribute("mlflow.trace.metadata.tokens", response.get("token_count", 0))

            # Background evaluation
            background_tasks.add_task(
                chat_service.evaluate_chat_response,
                question=request.question,
                answer=response['answer'],
                source_documents=response['source_documents']
            )

        duration = time.time() - start_time
        if root_span.is_recording():
            root_span.set_attribute("mlflow.trace.outputs", json.dumps(response, default=json_serial))
            root_span.set_attribute("mlflow.trace.metadata.tokens", response.get("token_count", 0))
            root_span.set_attribute("mlflow.trace.metadata.execution_time", f"{duration:.3f}s")
        chat_latency.record(duration, {"bot_id": request.bot_id})
        logger.info(
            "Chat request completed | bot_id=%s | session_id=%s | duration=%.3fs",
            request.bot_id,
            request.session_id,
            duration
        )

        return ChatResponse(**response)

    except HTTPException:
        raise
    except Exception as e:
        chat_error_counter.add(1, {"bot_id": request.bot_id})
        logger.exception("Error during chat | bot_id=%s | session_id=%s", request.bot_id, request.session_id)
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
