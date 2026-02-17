import logging
import os
from Config.settings import settings
class Logger:
    """
    Logger class for setting up and managing a logger.
    """
    def __init__(
        self,
        logger_name: str = None,
        log_format: str = None,
        datefmt: str = None,
        level: logging = logging.INFO,
        filename: str = None,
    ):
        """
        Initialize a Logger object.
        """
        if filename is None:
            raise ValueError("Filename must be specified")

        self.log_format = log_format
        self.datefmt = datefmt
        self.level = level
        self.filename = filename
        self.name = logger_name
        self._logger = self._setup_logger()

    def _setup_logger(self):
        """
        Set up a logger with the specified parameters.
        """
        logging.basicConfig(
            format=self.log_format,
            datefmt=self.datefmt,
            level=self.level,
            filename=self.filename,
        )

        return logging.getLogger(self.name)

    @property
    def logger(self) -> logging.Logger:
        """
        Return the logger object.
        """
        return self._logger

    @logger.setter
    def logger(self, logger):
        """
        Set the logger object.
        """
        self._logger = logger

    def add_handler(self, handler):
        """
        Add a handler to the logger.
        """
        self.logger.addHandler(handler)

    def add_filter(self, filter):
        """
        Add a filter to the logger.
        """
        self.logger.addFilter(filter)


def get_logger():
    """
    Get a logger with the settings from the Settings object.
    """
    logger = setup_logger()
    return logger


def setup_logger() -> Logger:
    """
    Set up a logger with the settings from the Settings object.
    """
    log_dir = settings.LOG_DIR

    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'app.log')
    logger_obj = Logger(
        logger_name=__name__,
        log_format="%(asctime)s [%(levelname)s] '%(pathname)s' %(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file
    )

    logger = logger_obj.logger
    return logger

logger = get_logger()
