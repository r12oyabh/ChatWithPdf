# IMPORTING FILES AND LIBRARY
from Config.logger import logger
from Config.settings import settings
from Routes.bot_routes import router as bot_router
from Routes.chat_routes import router as chat_router
# Core LangChain
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from datetime import datetime
import uvicorn
import mlflow

logger.info("Starting MultiAgent application")

app = FastAPI(
    title="MultiAgent",
    description="This is a MultiAgent",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

logger.info("Configuring CORS middleware")
mlflow.set_tracking_uri(settings.ML_FLOW_TRACKING_URL)
mlflow.set_experiment(settings.ML_FLOW_EXPERIMENT_NAME)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()

    logger.info(
        "Incoming request | method=%s | path=%s | client=%s",
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled exception during request | method=%s | path=%s",
            request.method,
            request.url.path,
        )
        raise

    duration = (datetime.utcnow() - start_time).total_seconds()

    logger.info(
        "Request completed | method=%s | path=%s | status_code=%d | duration=%.3fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )

    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "message": "Invalid request data",
            "details": exc.errors(),
            "timestamp": datetime.now().isoformat()
        }
    )


app.include_router(bot_router)
logger.info("Bot routes registered")

app.include_router(chat_router)
logger.info("Chat routes registered")


if __name__ == "__main__":
    logger.info("Running application with Uvicorn")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )