from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
import mlflow
from mlflow.entities import SpanType
from Config.logger import logger
from Config.settings import settings


class LLMManager:
    """
    A singleton-style manager for LLMs:
    - Starts with Azure OpenAI
    - Falls back to Google Gemini when Azure OpenAI fails
    """

    def __init__(self):
        self._llm = None
        self._llm_type = None  # Tracks which LLM is currently in use
        self._initialize_llm()

    @mlflow.trace(name="Initialize_LLM", span_type=SpanType.CHAIN)
    def _initialize_llm(self):
        """Initialize the primary LLM (Azure OpenAI), fallback to Gemini."""
        try:
            # Track Azure OpenAI initialization
            with mlflow.start_span(name="Setup_Azure_OpenAI", span_type=SpanType.CHAT_MODEL) as span:
                span.set_inputs({
                    "provider": "Azure OpenAI",
                    "deployment": settings.AZURE_OPENAI_DEPLOYMENT,
                    "api_version": settings.AZURE_OPENAI_API_VERSION
                })
                
                mlflow.openai.autolog()
                self._llm = AzureChatOpenAI(
                    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,  # e.g., "gpt-4o"
                    api_version=settings.AZURE_OPENAI_API_VERSION,  # e.g., "2023-06-01-preview"
                    temperature=settings.TEMPERATURE,
                    openai_api_key=settings.OPEN_API_KEY,
                    max_retries=2,
                    streaming=True
                )
                self._llm_type = "azure"
                
                span.set_outputs({
                    "llm_type": "azure",
                    "model": settings.AZURE_OPENAI_DEPLOYMENT,
                    "success": True
                })
                span.set_attributes({
                    "provider": "Azure OpenAI",
                    "temperature": settings.TEMPERATURE,
                    "max_retries": 2
                })
                
                logger.info(f"LLM initialized with Azure OpenAI deployment: {settings.AZURE_OPENAI_DEPLOYMENT}")
        except Exception as e:
            logger.warning(f"Failed to initialize Azure OpenAI LLM: {e}")
            logger.info("Falling back to Google Gemini...")
            self._initialize_gemini_llm()

    def _initialize_gemini_llm(self):
        """Initialize the Google Gemini LLM."""
        try:
            mlflow.gemini.autolog()
            self._llm = ChatGoogleGenerativeAI(
                temperature=settings.TEMPERATURE,
                model=settings.GEMINI_MODEL,
                google_api_key=settings.GEMINI_API_KEY
            )
            self._llm_type = "gemini"
            logger.info(f"LLM initialized with Google Gemini model: {settings.GEMINI_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Google Gemini LLM: {e}")
            raise e

    @property
    def llm(self):
        """Return the LLM instance."""
        if self._llm is None:
            self._initialize_llm()
        return self._llm

    @property
    def llm_type(self):
        """Return which LLM is currently active: 'azure' or 'gemini'."""
        return self._llm_type

    def call(self, *args, **kwargs):
        """
        Call the LLM. If Azure OpenAI fails due to quota or token errors, 
        automatically switch to Google Gemini.
        """
        try:
            return self.llm(*args, **kwargs)
        except Exception as e:
            logger.warning(f"{self._llm_type} LLM failed with error: {e}")
            if self._llm_type == "azure":
                logger.info("Switching to Google Gemini due to Azure OpenAI failure...")
                self._initialize_gemini_llm()
                return self.llm(*args, **kwargs)
            else:
                raise e