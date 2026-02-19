from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

from Config.logger import logger
from Config.settings import settings
from Config.telemetry import tracer


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

    def _initialize_llm(self):
        """Initialize the primary LLM (Azure OpenAI), fallback to Gemini."""
        with tracer.start_as_current_span("_initialize_llm") as span:
            try:
                # Track Azure OpenAI initialization
                self._llm = AzureChatOpenAI(
                    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,  # e.g., "gpt-4o"
                    api_version=settings.AZURE_OPENAI_API_VERSION,  # e.g., "2023-06-01-preview"
                    temperature=settings.TEMPERATURE,
                    openai_api_key=settings.OPEN_API_KEY,
                    max_retries=2,
                )
                self._llm_type = "azure"
                span.set_attribute("llm.type", "azure")
                                    
                logger.info(f"LLM initialized with Azure OpenAI deployment: {settings.AZURE_OPENAI_DEPLOYMENT}")
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"Failed to initialize Azure OpenAI LLM: {e}")
                logger.info("Falling back to Google Gemini...")
                self._initialize_gemini_llm()


    def _initialize_gemini_llm(self):
        """Initialize the Google Gemini LLM."""
        try:
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
        with tracer.start_as_current_span("llm_call") as span:
            span.set_attribute("llm.type", self._llm_type)
            try:
                return self.llm(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                logger.warning(f"{self._llm_type} LLM failed with error: {e}")
                if self._llm_type == "azure":
                    logger.info("Switching to Google Gemini due to Azure OpenAI failure...")
                    self._initialize_gemini_llm()
                    span.set_attribute("llm.fallback", True)
                    span.set_attribute("llm.new_type", "gemini")
                    return self.llm(*args, **kwargs)
                else:
                    raise e
