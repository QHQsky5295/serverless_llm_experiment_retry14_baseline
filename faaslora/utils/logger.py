"""
FaaSLoRA Logging System

Provides structured logging with multiple output formats and log level management.
"""

import logging
import logging.handlers
import sys
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import threading
from dataclasses import dataclass


@dataclass
class LogConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True
    json_format: bool = False
    structured_logging: bool = True


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs"""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if enabled
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                              'filename', 'module', 'lineno', 'funcName', 'created',
                              'msecs', 'relativeCreated', 'thread', 'threadName',
                              'processName', 'process', 'getMessage', 'exc_info',
                              'exc_text', 'stack_info']:
                    log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """Filter to add context information to log records"""
    
    def __init__(self):
        super().__init__()
        self.context_data = threading.local()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context data to log record"""
        # Add context data if available
        if hasattr(self.context_data, 'data'):
            for key, value in self.context_data.data.items():
                setattr(record, key, value)
        
        return True
    
    def set_context(self, **kwargs):
        """Set context data for current thread"""
        if not hasattr(self.context_data, 'data'):
            self.context_data.data = {}
        self.context_data.data.update(kwargs)
    
    def clear_context(self):
        """Clear context data for current thread"""
        if hasattr(self.context_data, 'data'):
            self.context_data.data.clear()


class FaaSLoRALogger:
    """
    Enhanced logger for FaaSLoRA with structured logging support
    """
    
    def __init__(self, name: str, config: Optional[LogConfig] = None):
        """
        Initialize logger
        
        Args:
            name: Logger name
            config: Logging configuration
        """
        self.name = name
        self.config = config or LogConfig()
        self.logger = logging.getLogger(name)
        self.context_filter = ContextFilter()
        
        # Configure logger
        self._configure_logger()
    
    def _configure_logger(self):
        """Configure the logger"""
        # Set log level
        level = getattr(logging, self.config.level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Add context filter
        self.logger.addFilter(self.context_filter)
        
        # Configure console handler
        if self.config.console_output:
            self._add_console_handler()
        
        # Configure file handler
        if self.config.file_path:
            self._add_file_handler()
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _add_console_handler(self):
        """Add console handler"""
        handler = logging.StreamHandler(sys.stdout)
        
        if self.config.json_format:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                fmt=self.config.format,
                datefmt=self.config.date_format
            )
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def _add_file_handler(self):
        """Add file handler with rotation"""
        # Create log directory if it doesn't exist
        log_path = Path(self.config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            filename=self.config.file_path,
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count,
            encoding='utf-8'
        )
        
        if self.config.json_format:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                fmt=self.config.format,
                datefmt=self.config.date_format
            )
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def set_context(self, **kwargs):
        """Set logging context for current thread"""
        self.context_filter.set_context(**kwargs)
    
    def clear_context(self):
        """Clear logging context for current thread"""
        self.context_filter.clear_context()
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        kwargs['exc_info'] = True
        self._log(logging.ERROR, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method"""
        # Add extra fields to log record
        extra = {}
        for key, value in kwargs.items():
            if key != 'exc_info':
                extra[key] = value
        
        # Log the message
        self.logger.log(level, message, extra=extra, exc_info=kwargs.get('exc_info', False))
    
    def log_performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metrics"""
        self.info(f"Performance: {operation}", 
                 operation=operation, 
                 duration_ms=duration_ms, 
                 **kwargs)
    
    def log_request(self, method: str, path: str, status_code: int, 
                   duration_ms: float, **kwargs):
        """Log HTTP request"""
        self.info(f"Request: {method} {path} -> {status_code}",
                 method=method,
                 path=path,
                 status_code=status_code,
                 duration_ms=duration_ms,
                 **kwargs)
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with additional context"""
        self.error(f"Error: {str(error)}", 
                  error_type=type(error).__name__,
                  error_message=str(error),
                  **context,
                  exc_info=True)


# Global logger registry
_loggers: Dict[str, FaaSLoRALogger] = {}
_logger_lock = threading.Lock()
_default_config: Optional[LogConfig] = None


def configure_logging(config: LogConfig):
    """
    Configure global logging settings
    
    Args:
        config: Logging configuration
    """
    global _default_config
    _default_config = config
    
    # Reconfigure existing loggers
    with _logger_lock:
        for logger in _loggers.values():
            logger.config = config
            logger._configure_logger()


def get_logger(name: str, config: Optional[LogConfig] = None) -> FaaSLoRALogger:
    """
    Get or create a logger instance
    
    Args:
        name: Logger name
        config: Optional logging configuration
        
    Returns:
        Logger instance
    """
    with _logger_lock:
        if name not in _loggers:
            logger_config = config or _default_config or LogConfig()
            _loggers[name] = FaaSLoRALogger(name, logger_config)
        
        return _loggers[name]


def setup_logging_from_config(config_dict: Dict[str, Any]):
    """
    Setup logging from configuration dictionary
    
    Args:
        config_dict: Configuration dictionary
    """
    logging_config = config_dict.get('logging', {})
    
    log_config = LogConfig(
        level=logging_config.get('level', 'INFO'),
        format=logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        date_format=logging_config.get('date_format', '%Y-%m-%d %H:%M:%S'),
        file_path=logging_config.get('file_path'),
        max_file_size=logging_config.get('max_file_size', 10 * 1024 * 1024),
        backup_count=logging_config.get('backup_count', 5),
        console_output=logging_config.get('console_output', True),
        json_format=logging_config.get('json_format', False),
        structured_logging=logging_config.get('structured_logging', True)
    )
    
    configure_logging(log_config)


class LoggingContext:
    """Context manager for logging context"""
    
    def __init__(self, logger: FaaSLoRALogger, **kwargs):
        """
        Initialize logging context
        
        Args:
            logger: Logger instance
            **kwargs: Context data
        """
        self.logger = logger
        self.context_data = kwargs
    
    def __enter__(self):
        """Enter context"""
        self.logger.set_context(**self.context_data)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context"""
        self.logger.clear_context()


def with_logging_context(logger: FaaSLoRALogger, **kwargs):
    """
    Create logging context manager
    
    Args:
        logger: Logger instance
        **kwargs: Context data
        
    Returns:
        Context manager
    """
    return LoggingContext(logger, **kwargs)


class PerformanceLogger:
    """Performance logging utility"""
    
    def __init__(self, logger: FaaSLoRALogger, operation: str):
        """
        Initialize performance logger
        
        Args:
            logger: Logger instance
            operation: Operation name
        """
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        """Start timing"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log performance"""
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.logger.log_performance(self.operation, duration_ms)


def log_performance(logger: FaaSLoRALogger, operation: str):
    """
    Create performance logging context manager
    
    Args:
        logger: Logger instance
        operation: Operation name
        
    Returns:
        Context manager
    """
    return PerformanceLogger(logger, operation)


# Convenience function for quick logging setup
def setup_basic_logging(level: str = "INFO", 
                       file_path: Optional[str] = None,
                       json_format: bool = False):
    """
    Setup basic logging configuration
    
    Args:
        level: Log level
        file_path: Optional file path for logging
        json_format: Whether to use JSON format
    """
    config = LogConfig(
        level=level,
        file_path=file_path,
        json_format=json_format
    )
    configure_logging(config)
