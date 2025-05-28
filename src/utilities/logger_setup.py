"""
Logger setup module for configuring logging in the project.

This module provides functionality for setting up and configuring the logging system
used throughout the project. It configures loggers with appropriate handlers for 
console output, main log files, and error-specific log files.

All logs are stored in the directory defined by the LOGS_DIR constant from the config.py

The structure of the log files is as follows:
    - logs/
        - backtest_logs/
            - {timestamp}_utc_{model_name}_{run_type}.log
        - errors/
            - {timestamp}_utc_{model_name}_{run_type}_error.log
            
Importing the module does **not** perform the configuration; 
Call ``configure_logging `` from your entry-point script or notebook. 

Usage example:

from src.utilities import logger_setup as log_setup

logger = log_setup.configure_logging (
    name=__name__, 
    run_type='backtest_logs',
    console_log_level=logging.INFO,     
    model_name="TiDE",
    timestamp=pd.Timestamp.now(tz='UTC'),
)

"""

#%% #>-------------------- Imports and module exports --------------------

#* standard library
import logging
import sys
import time
from pathlib import Path

#* third-party libraries
import pandas as pd

#* Local imports
from config import LOGS_DIR

#* Module exports
__all__ = [
    'configure_logging ',
]

#%%#>------------ Function: Setup logger --------------


def configure_logging (
    name: str,
    run_type: str = 'backtest_logs',
    console_log_level: int = logging.INFO,
    file_log_level: int = logging.DEBUG,
    model_name: str = None,
    timestamp: str = None,
) -> logging.Logger:
    """Creates and configures a logger instance with specialized handlers.
    
    This function creates a logger with three handlers:
    1. Main log file - captures all logs at file_log_level
    2. Error log file - captures only ERROR level logs
    3. Console output - displays logs at console_log_level

    The logger inherits from the root logger and uses UTC timestamps for all logs.
    Log files are organized by run_type and named with timestamps and model information.

    Args:
        name (str): Logger name (typically __name__ from the calling module)
        run_type (str): Type of logs to categorize the log directory (default: 'backtest_logs')
        console_log_level (int): Logging level for console output (default: logging.INFO)
        file_log_level (int): Logging level for main file output (default: logging.DEBUG)
        model_name (str): Name of the model for log file naming (default: None)
        timestamp (str): Custom timestamp (YYYYMMDD_HHMMSS) for log files (default: current UTC time)
        
    Returns:
        logging.Logger: A configured logger instance with file and console handlers
        
    Example:
        logger = configure_logging (
            name=__name__, 
            run_type='backtest_logs',
            console_log_level=logging.INFO,     
            model_name="TiDE",
        )
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.error("This is an error message")
    """
    
    ##* Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers from root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    ##* Setup directories
    log_root = Path(LOGS_DIR)
    
    main_log_dir = log_root / run_type.lower()
    error_log_dir = log_root / 'errors'
    
    main_log_dir.mkdir(parents=True, exist_ok=True)
    error_log_dir.mkdir(parents=True, exist_ok=True)
    
    ##* Setup log files names
    timestamp = timestamp or pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')
    main_log_file = main_log_dir / f'{timestamp}_utc_{model_name}_{run_type}.log'
    error_log_file = error_log_dir / f'{timestamp}_utc_{model_name}_{run_type}_error.log'
    
    ##* Create and configure handlers
    file_formatter = logging.Formatter(
        '%(asctime)s UTC - %(name)s - %(funcName)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'
    )
    file_formatter.converter = time.gmtime

    console_formatter = logging.Formatter('%(asctime)s UTC - %(levelname)s - %(message)s', '%H:%M:%S')
    console_formatter.converter = time.gmtime

    handlers = [
        (logging.FileHandler(main_log_file), file_log_level, file_formatter),
        (logging.FileHandler(error_log_file), logging.ERROR, file_formatter),
        (logging.StreamHandler(sys.stdout), console_log_level, console_formatter)
    ]
    
    # Add handlers to root logger
    for handler, level, formatter in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    
    # Get the named logger (will inherit from root)
    logger = logging.getLogger(name)
    logger.propagate = True  # Ensure propagation is enabled
        
    return logger
