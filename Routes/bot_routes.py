"""
API routes for bot upload and creation.
"""

from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from Schema.schema import (
    BotCreateResponse,
    ErrorResponse
)
from Utills.file_utills import validate_and_save_files
from Utills.botservice import bot_service
from Config.logger import logger


import time
from opentelemetry import trace, metrics
import json

from datetime import datetime

# Helper for JSON serialization of datetime
def json_serial(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

router = APIRouter(prefix="/bot", tags=["Bot Management"])

# Initialize tracer and meter
tracer = trace.get_tracer("bot.routes")
meter = metrics.get_meter("bot.routes")

# Metrics
bot_creation_counter = meter.create_counter(
    name="bot_creation_total",
    description="Total number of bot creation attempts",
    unit="1"
)
bot_error_counter = meter.create_counter(
    name="bot_errors_total",
    description="Total number of failed bot operations",
    unit="1"
)
bot_latency = meter.create_histogram(
    name="bot_operation_latency_seconds",
    description="Latency of bot operations",
    unit="s"
)


@router.post(
    "/upload",
    response_model=BotCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload files and create a new bot",
    description="""
    Upload PDF, DOCX, or TXT files to create a new chatbot instance.
    
    - **team_name**: Name of the team/organization (required)
    - **bot_name**: Name of the bot (required)
    - **files**: One or more files to upload (PDF, DOCX, or TXT)
    
    Each bot is isolated in its own namespace, ensuring data separation
    between different teams and bots.
    """,
    responses={
        201: {
            "description": "Bot created successfully",
            "model": BotCreateResponse
        },
        400: {
            "description": "Bad request - invalid input or file type",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    }
)
async def upload_and_create_bot(
    team_name: str = Form(..., description="Name of the team/organization"),
    bot_name: str = Form(..., description="Name of the bot"),
    files: List[UploadFile] = File(..., description="Files to upload (PDF, DOCX, TXT)")
) -> BotCreateResponse:
    
    start_time = time.time()
    bot_creation_counter.add(1, {"team": team_name})

    root_span = trace.get_current_span()

    if root_span.is_recording():
        root_span.set_attribute("mlflow.traceName", f"Create Bot: {bot_name}")
        root_span.set_attribute("mlflow.trace.metadata.team_name", team_name)
        root_span.set_attribute("mlflow.trace.metadata.bot_name", bot_name)
        
        # MLflow UI expects JSON for inputs/outputs to render nicely
        inputs = {
            "team_name": team_name,
            "bot_name": bot_name,
            "file_count": len(files)
        }
        root_span.set_attribute("mlflow.trace.inputs", json.dumps(inputs))
        logger.info("✅ Set attributes on root span")

    logger.info(
        "Bot creation request received | team_name=%s | bot_name=%s | file_count=%d",
        team_name,
        bot_name,
        len(files),
    )

    """
    Upload files and create a new bot instance.
    
    Args:
        team_name: Name of the team/organization
        bot_name: Name of the bot
        files: List of files to upload
        
    Returns:
        BotCreateResponse with bot metadata
        
    Raises:
        HTTPException: If validation fails or bot creation fails
    """
    try:
        # Validate team_name and bot_name
        if not team_name or not team_name.strip():
            logger.warning("Empty team name provided")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Team name cannot be empty"
            )
        
        if not bot_name or not bot_name.strip():
            logger.warning("Empty bot name provided | team_name=%s", team_name)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bot name cannot be empty"
            )
        
        with tracer.start_as_current_span("bot_creation_workflow") as span:
            span.set_attribute("mlflow.traceName", f"Create Bot: {bot_name}")
            span.set_attribute("mlflow.trace.metadata.team_name", team_name)
            span.set_attribute("mlflow.trace.metadata.bot_name", bot_name)
            span.set_attribute("mlflow.trace.metadata.num_files", len(files))
            span.set_attribute("mlflow.trace.inputs", f"Team: {team_name}, Bot: {bot_name}, Files: {len(files)}")
            
            span.set_attribute("team.name", team_name)
            span.set_attribute("bot.name", bot_name)
            span.set_attribute("file.count", len(files))

            logger.info("Validating and saving files | team_name=%s | bot_name=%s",
                team_name,bot_name,)

            # Validate and save files
            file_paths = await validate_and_save_files(files)
            
            logger.info(
                "Files saved successfully | count=%d | team_name=%s | bot_name=%s",
                len(file_paths),
                team_name,
                bot_name,
            )

            # Create bot
            bot_metadata = bot_service.create_bot(
                team_name=team_name.strip(),
                bot_name=bot_name.strip(),
                file_paths=file_paths
            )
            
            span.set_attribute("bot.id", bot_metadata.get("bot_id", "unknown"))
            span.set_attribute("mlflow.trace.metadata.bot_id", bot_metadata.get("bot_id", "unknown"))
            span.set_attribute("mlflow.trace.outputs", f"Bot created with ID: {bot_metadata.get('bot_id')}")

        duration = time.time() - start_time
        if root_span.is_recording():
            root_span.set_attribute("mlflow.trace.metadata.execution_time", f"{duration:.3f}s")
            root_span.set_attribute("mlflow.trace.metadata.num_files", len(files))
            root_span.set_attribute("mlflow.trace.outputs", json.dumps(bot_metadata, default=json_serial))
            
            # Custom Metadata (Verbose for User Request)
            root_span.set_attribute("mlflow.trace.metadata.bot_id", bot_metadata.get("bot_id"))
            root_span.set_attribute("mlflow.trace.metadata.bot_name", bot_name)
            root_span.set_attribute("mlflow.trace.metadata.team_name", team_name)
            root_span.set_attribute("mlflow.trace.metadata.num_files", len(files))
            root_span.set_attribute("mlflow.trace.metadata.operation", "create_bot")
            root_span.set_attribute("mlflow.trace.metadata.operations", "create_bot")
            root_span.set_attribute("mlflow.trace.metadata.User", team_name)
            root_span.set_attribute("mlflow.trace.metadata.Run_name", f"Create-{bot_name}")
            root_span.set_attribute("mlflow.trace.metadata.base_vector", "ChromaDB")
            root_span.set_attribute("mlflow.trace.metadata.vectordbname", "ChromaDB")

            # MLflow UI Standard Tags/Columns
            root_span.set_attribute("mlflow.user", team_name)
            root_span.set_attribute("mlflow.runName", f"Create-{bot_name}")
            root_span.set_attribute("mlflow.tag.User", team_name)
            root_span.set_attribute("mlflow.tag.Run_name", f"Create-{bot_name}")
            root_span.set_attribute("mlflow.tag.team_name", team_name)
            root_span.set_attribute("mlflow.tag.bot_id", bot_metadata.get("bot_id"))
            root_span.set_attribute("mlflow.tag.bot_name", bot_name)
            root_span.set_attribute("mlflow.tag.operation", "create_bot")
        bot_latency.record(duration, {"operation": "create", "team": team_name})

        logger.info(
            "Bot created successfully | bot_id=%s | team_name=%s | bot_name=%s | duration=%.3fs",
            bot_metadata.get("bot_id"),
            team_name,
            bot_name,
            duration
        )        
        # Return response
        return BotCreateResponse(**bot_metadata, message="Bot created successfully")
       
    except HTTPException:
        bot_error_counter.add(1, {"operation": "create", "type": "http_exception"})
        raise
    except Exception as e:
        bot_error_counter.add(1, {"operation": "create", "type": "error"})
        logger.exception("Error creating bot | team_name=%s | bot_name=%s", team_name, bot_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating bot: {str(e)}"
        )


@router.delete(
    "/{bot_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a bot",
    description="Delete a bot and all its associated data from the vector database.",
    responses={
        200: {"description": "Bot deleted successfully"},
        404: {"description": "Bot not found", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def delete_bot(bot_id: str):
    logger.info("Delete bot request received | bot_id=%s", bot_id)
    """
    Delete a bot and its associated data.
    
    Args:
        bot_id: Unique bot identifier
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If bot not found or deletion fails
    """
    start_time = time.time()
    try:
        # Check if bot exists
        if not bot_service.bot_exists(bot_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bot with ID '{bot_id}' not found"
            )
        
        with tracer.start_as_current_span("bot_deletion") as span:
            span.set_attribute("bot.id", bot_id)
            # Delete bot
            bot_service.delete_bot(bot_id)
        
        duration = time.time() - start_time
        bot_latency.record(duration, {"operation": "delete"})
        logger.info("Bot deleted successfully | bot_id=%s | duration=%.3fs", bot_id, duration)

        return {
            "message": f"Bot '{bot_id}' deleted successfully",
            "bot_id": bot_id
        }
        
    except HTTPException:
        bot_error_counter.add(1, {"operation": "delete", "type": "http_exception"})
        raise
    except Exception as e:
        bot_error_counter.add(1, {"operation": "delete", "type": "error"})
        logger.exception("Unhandled error during bot deletion | bot_id=%s", bot_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting bot: {str(e)}"
        )