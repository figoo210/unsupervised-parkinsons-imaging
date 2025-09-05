import logging
import os
import sys
from datetime import datetime
from pathlib import Path


class Logger:
    """A configurable logging system for the project.
    
    Features:
    - Console and file logging
    - Configurable log levels
    - Customizable formatters
    - Log rotation support
    - Module-specific loggers
    """
    
    # Default log format
    DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    
    # Log levels mapping
    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    
    def __init__(self, name="main", log_level="info", log_to_file=True, log_to_console=True):
        """Initialize the logger with the given configuration.
        
        Args:
            name (str): Logger name (typically module name)
            log_level (str): Logging level (debug, info, warning, error, critical)
            log_to_file (bool): Whether to log to a file
            log_to_console (bool): Whether to log to console
        """
        self.name = name
        self.log_level = self.LEVELS.get(log_level.lower(), logging.INFO)
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        self.logger.propagate = False  # Don't propagate to parent loggers
        
        # Clear any existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Add handlers based on configuration
        if log_to_console:
            self._add_console_handler()
        
        if log_to_file:
            self._add_file_handler()
    
    def _add_console_handler(self):
        """Add a console handler to the logger."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        formatter = logging.Formatter(self.DEFAULT_FORMAT)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self, filename=None, log_dir="logs"):
        """Add a file handler to the logger.
        
        Args:
            filename (str, optional): Log filename. Defaults to None (auto-generated).
            log_dir (str, optional): Directory for log files. Defaults to "logs".
        """
        # Create logs directory if it doesn't exist
        Path(log_dir).mkdir(exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.log"
        
        file_path = os.path.join(log_dir, filename)
        file_handler = logging.FileHandler(file_path, mode="a")
        file_handler.setLevel(self.log_level)
        formatter = logging.Formatter(self.DEFAULT_FORMAT)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        return file_path
    
    def set_level(self, level):
        """Set the logging level.
        
        Args:
            level (str): Logging level (debug, info, warning, error, critical)
        """
        if level.lower() in self.LEVELS:
            self.log_level = self.LEVELS[level.lower()]
            self.logger.setLevel(self.log_level)
            for handler in self.logger.handlers:
                handler.setLevel(self.log_level)
    
    def debug(self, message):
        """Log a debug message."""
        self.logger.debug(message)
    
    def info(self, message):
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log an error message."""
        self.logger.error(message)
    
    def critical(self, message):
        """Log a critical message."""
        self.logger.critical(message)
    
    def exception(self, message):
        """Log an exception message with traceback."""
        self.logger.exception(message)


# Create a default logger instance for quick access
def get_logger(name="main", log_level="info", log_to_file=True, log_to_console=True):
    """Get a configured logger instance.
    
    Args:
        name (str): Logger name (typically module name)
        log_level (str): Logging level (debug, info, warning, error, critical)
        log_to_file (bool): Whether to log to a file
        log_to_console (bool): Whether to log to console
        
    Returns:
        Logger: Configured logger instance
    """
    return Logger(name, log_level, log_to_file, log_to_console)


# Example usage
if __name__ == "__main__":
    # Basic usage
    logger = get_logger("example")
    logger.info("This is an info message")
    logger.error("This is an error message")
    
    # Module-specific logger with custom settings
    data_logger = get_logger("data_processing", log_level="debug")
    data_logger.debug("Processing data...")
    data_logger.info("Data processed successfully")
