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


_diagnostic_stack_state = None


def diagnostic_control_enabled():
    """Only an actually qualified, current-process diagnostic may emit events."""
    import os
    return (_diagnostic_stack_state is not None
            and _diagnostic_stack_state[0] == os.getpid())


def diagnostic_control_event(attempt_id, boundary, *, byte_count=None, outcome=None):
    """Bounded read-only source-RPC observations, never a serving metric.

    Missing terminal events remain incomplete. Async thread CPU deltas include
    other tasks and must not be interpreted as per-request CPU consumption.
    Emission overhead is deliberately not subtracted from observed intervals.
    """
    if attempt_id is None:
        return
    import os
    import uuid
    from faaslora.clock import local_monotonic_clock_id
    if not diagnostic_control_enabled():
        raise ValueError('control observation requires a qualified diagnostic process')
    if not isinstance(attempt_id, str) or uuid.UUID(attempt_id).hex != attempt_id:
        raise ValueError('invalid control observation identity')
    if boundary not in ('parent_begin', 'parent_send', 'parent_received', 'parent_terminal',
                        'worker_received', 'frontend_begin', 'frontend_native_send',
                        'frontend_native_received', 'frontend_ready', 'native_begin', 'native_ready'):
        raise ValueError('invalid control observation boundary')
    if byte_count is not None and (type(byte_count) is not int or byte_count < 0
                                  or boundary not in ('parent_send', 'parent_received')):
        raise ValueError('invalid control observation byte count')
    if ((boundary == 'parent_terminal' and outcome not in ('success', 'error', 'cancelled'))
            or (boundary != 'parent_terminal' and outcome is not None)):
        raise ValueError('invalid control observation outcome')
    row = dict(event='request_source_control_boundary_v1', attempt_id=attempt_id,
        boundary=boundary, pid=os.getpid(), thread_ident=threading.get_ident(),
        clock_id=local_monotonic_clock_id(), monotonic_s=time.monotonic(),
        thread_cpu_s=time.thread_time(), byte_count=byte_count, outcome=outcome)
    payload = (json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n').encode()
    if _diagnostic_stack_state[4].write(payload) != len(payload):
        raise OSError('incomplete diagnostic control observation write')


def _diagnostic_thread_frames():
    """Copy frame locations while holding ordinary Python references, no locals."""
    threads = []
    frames = sys._current_frames()
    try:
        for ident, frame in frames.items():
            locations = []
            while frame is not None and len(locations) < 100:
                locations.append((frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno))
                frame = frame.f_back
            threads.append(dict(ident=ident, frames=locations, truncated=frame is not None))
        return threads
    finally:
        # Do not retain application frames/locals between observations.
        frame = None
        frames.clear()


def enable_diagnostic_stack_sampling():
    """Cooperative observations, not a CPU-time profiler or serving policy.

    Unlike an asynchronous C stack walk, this observer uses Python frame
    references under the GIL. Its own scheduling can be delayed: record actual
    times and do not treat samples as an unbiased profile. No signal/ptrace.
    """
    import os
    global _diagnostic_stack_state
    flag = os.environ.get('FAASLORA_TC_STACK_SAMPLING')
    if flag is None or flag == '0':
        return
    if flag != 'python_frames_v1' or os.environ.get('FAASLORA_FORMAL_RUN') != '0':
        raise ValueError('stack sampling requires an explicit nonformal Python-frame diagnostic')
    pid = os.getpid()
    if _diagnostic_stack_state is not None:
        if _diagnostic_stack_state[0] != pid:
            raise ValueError('fork-inherited diagnostic observer is not active in this process')
        return
    receipt_path = Path(os.environ['FAASLORA_TC_LAUNCH_RECEIPT']).resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    unified = [r[3:] for r in Path('/proc/self/cgroup').read_text().splitlines() if r.startswith('0::')]
    actual = Path('/sys/fs/cgroup') / unified[0].lstrip('/') if len(unified) == 1 else None
    expected = Path(receipt['service_identity']['path'])
    replay = receipt.get('external_replay', {})
    if (receipt.get('allow_exec') is not True or actual is None
            or not actual.is_relative_to(expected)
            or replay.get('replay_scope') != 'diagnostic_prefix_v1'
            or type(replay.get('diagnostic_prefix_count')) is not int
            or replay['diagnostic_prefix_count'] <= 0):
        raise ValueError('stack sampling is outside its gated diagnostic service')
    import atexit
    ticks = int(Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19])
    directory = receipt_path.parent/'diagnostic_stacks'
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink():
        raise ValueError('stack output must not be a symlink')
    stem = directory/f'{pid}-{ticks}'
    with stem.with_suffix('.json').open('x') as stream:
        json.dump(dict(kind='diagnostic_python_frames_v1', pid=pid, parent_pid=os.getppid(),
            start_ticks=ticks, cgroup=str(actual), executable=sys.executable,
            main_thread_ident=threading.main_thread().ident,
            capture_start_monotonic_s=time.monotonic(), period_seconds=2.0,
            max_frames_per_thread=100, includes_idle_threads=True,
            cpu_time_profile=False, formal_performance_result=False,
            gil_scheduling_bias=True, final_partial_line_possible=True,
            control_boundaries='request_source_control_boundary_v1',
            control_overhead_included=True), stream, indent=2)
    output = stem.with_suffix('.stacks.jsonl').open('xb', buffering=0)
    control_output = stem.with_suffix('.control.jsonl').open('xb', buffering=0)
    stop = threading.Event()

    def observe():
        previous = time.monotonic()
        previous_cpu = time.process_time()
        next_due = previous + 2.0
        observer_cpu = 0.0
        index = 0
        while not stop.wait(max(0.0, next_due-time.monotonic())):
            started = time.monotonic()
            cpu = time.process_time()
            own_cpu = time.thread_time()
            threads = _diagnostic_thread_frames()
            snapshot_done = time.monotonic()
            index += 1
            row = dict(event='python_frames', index=index, pid=pid,
                scheduled_monotonic_s=next_due, observed_monotonic_s=started,
                interval_seconds=started-previous, scheduler_delay_seconds=started-next_due,
                process_cpu_seconds=cpu, process_cpu_delta_seconds=cpu-previous_cpu,
                observer_thread_ident=threading.get_ident(),
                observer_cpu_seconds_before_sample=observer_cpu,
                snapshot_wall_seconds=snapshot_done-started, threads=threads)
            payload = (json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n').encode()
            if output.write(payload) != len(payload):
                raise OSError('incomplete diagnostic observation write')
            observer_cpu += time.thread_time()-own_cpu
            previous, previous_cpu = started, cpu
            # No catch-up burst after GIL/I/O delays; preserve the observed gap.
            next_due = time.monotonic()+2.0

    thread = threading.Thread(target=observe, name='tc-python-frame-observer', daemon=True)
    _diagnostic_stack_state = (pid, stop, thread, output, control_output)
    atexit.register(stop.set)
    thread.start()


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
