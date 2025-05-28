
"""
This module provides general utility functions used throughout the project. 

Functions provided:
- set_all_seeds: Sets random seeds for reproducibility across different libraries (numpy, pytorch, etc.)
- save_backtest_results: Saves backtest results to CSV files with timestamps.
- start_dashboard / close_dashboard: Launches and closes TensorBoard and Optuna dashboards.
- startup / cleanup: convenience wrappers that combine seeding and dashboard management.

Note:
- All paths are relative to the project root and configurable via the config.py file.
- The module is import-only; nothing executes until the functions are called.

Usage example:
    from src.utilities import general_utils
    
    # Set random seeds for reproducibility
    general_utils.set_all_seeds(seed=42)
    
    # Start TensorBoard dashboard
    general_utils.start_dashboard(dashboard_type="tensorboard")
    
"""

#%% #>-------------------- Imports, logger and module exports --------------------

#* standard library
import os
import logging
import random
import time
import subprocess
import atexit
from pathlib import Path
from typing import Any

#* third-party libraries
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import optuna
import psutil
import webbrowser

#* Local imports
from config import (
    PROJECT_ROOT,
    BACKTEST_RESULTS_PATH,
    OPTUNA_STORAGE_PATH,
    TENSORBOARD_PATH,
)

#* Logger setup
logger = logging.getLogger(__name__)

#* Module exports
__all__ = [
    'set_all_seeds',
    'save_backtest_results',
    'start_dashboard',
    'close_dashboard',
    'startup',
    'cleanup',
]

#%% #>------------------- Function: Set Random Seeds -----------------------------
def set_all_seeds(seed: int = 123) -> None:
    """Sets random seeds for reproducibility across all libraries.
    
    Sets consistent random seeds for Python's random module, NumPy, PyTorch,
    and PyTorch Lightning and configures CUDNN for deterministic behavior to 
    ensure reproducible results across runs.
    
    Args:
        seed (int): Seed value for random number generators (default: 123)
        
    Returns:
        None
    """
    logger.info("Setting random seed to %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True, verbose=False)
    
    # Enable deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#%% #>------------------- Function: Save backtest results -----------------------------

def save_backtest_results(
    results_dict: dict[str, pd.DataFrame],
    config_dict: dict[str, Any],
) -> None:
    """Saves backtest results to CSV files with timestamps.
    
    Creates two CSV files:
        1. Window-level results
        2. Series-level results
    
    The files are saved with a timestamp and model name in the filename.
       
    Args:
        results_dict (dict[str, pd.DataFrame]): Dictionary containing 'window_df' and 'series_df' DataFrames
        config_dict (dict[str, Any]): Configuration dictionary containing model_name and optional keys 
            backtest_timestamp and backtest_results_save_path.
    
    Returns:
        None
    """
    #* unpack variables from config_dict
    model_name = config_dict['model_name']
    backtest_timestamp = config_dict.get(
        'backtest_timestamp', 
        pd.Timestamp.now(tz='UTC').strftime("%Y%m%d_%H%M")
    )
    
    #* Get path to save results and create directory if it does not exist
    backtest_results_path = config_dict.get('backtest_results_save_path') or BACKTEST_RESULTS_PATH
    os.makedirs(backtest_results_path, exist_ok=True)
    
    #* Define file names with timestamp
    file_names = {
        'window_df': f"{backtest_timestamp}_utc_{model_name}_backtest_window_results.csv",
        'series_df': f"{backtest_timestamp}_utc_{model_name}_backtest_series_results.csv"
    }
    
    # Save results to csv
    for df_type, file_name in file_names.items():
        file_path = os.path.join(backtest_results_path, file_name)
        results_dict[df_type].to_csv(file_path, index=False)
        logger.info("Results saved to: %s", file_path)
    


#%%#>--------- Functions: launch/stop dash (optuna and tensorboard)------------

def get_processes_on_port(port: int) -> list[psutil.Process]:
    """Returns processes listening on a specified TCP port.
    
    Identifies all system processes that are currently bound to the specified
    TCP port using psutil.
    
    Args:
        port (int): The TCP port number to check
        
    Returns:
        list[psutil.Process]: List of psutil.Process objects bound to the specified port
    """
    results = []
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            # Use net_connections(kind="inet")
            for con in proc.net_connections(kind="inet"):
                if con.laddr.port == port:
                    results.append(proc)
                    break  # No need to keep checking more connections for this proc
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return results


def start_dashboard(
    dashboard_type: str,
    dashboard_path: str | Path | None = None,
    port: int | None = None,
) -> None:
    """Starts a dashboard (TensorBoard or Optuna) and opens it in the browser.
    
    If the dashboard is already running, it will skip starting a new instance.
    Creates necessary directories or databases if they don't exist.
    
    Args:
        dashboard_type (str): Type of dashboard, either "tensorboard" or "optuna"
        dashboard_path (str): Path to logs or database. If None, uses standard paths from config
        port (int): Port number for the dashboard. If None, uses default (6006 for TensorBoard, 
            8080 for Optuna)
    
    Returns:
        None    
    """
    #* 1. Decide default paths & ports 
    if dashboard_type == "tensorboard":
        default_path = TENSORBOARD_PATH
        default_port = 6006
        expected_cmd_substring = "tensorboard"
    elif dashboard_type == "optuna":
        default_path = OPTUNA_STORAGE_PATH / "optuna_study.db"
        default_port = 8080
        expected_cmd_substring = "optuna-dashboard"
    else:
        raise ValueError("dashboard_type must be 'tensorboard' or 'optuna'")
    
    dashboard_path = dashboard_path or default_path
    port = port or default_port

    #* 2. Check if any processes are using this port
    # if is using and is the correct process, skip
    # if is using and is not the correct process, kill it
    
    procs_on_port = get_processes_on_port(port)
    if procs_on_port:
        # See if one of them is the correct dashboard
        found_correct = False
        for p in procs_on_port:
            cmdline = p.info["cmdline"] or []
            full_cmd = " ".join(cmdline)
            if expected_cmd_substring in full_cmd:
                logger.info("%s is already running on port %d (PID: %d). Skipping new launch.",
                            dashboard_type, port, p.pid)
                found_correct = True
                break
        if found_correct:
            return # Skip launching new dashboard
        else:
        # Port is in use by something else => kill the process
            logger.warning(
                "Port %d is in use by %d process(es), but not %s. Killing them...",
                port, len(procs_on_port), expected_cmd_substring
            )
            for p in procs_on_port:
                try:
                    logger.warning("Killing PID %d (%s)", p.pid, " ".join(p.info["cmdline"]))
                    p.kill()
                    p.wait()
                except Exception as e:
                    logger.error("Error killing process %d: %s", p.pid, e)
            
    
    # If we reach here, no correct dashboard is on port => start a new one
    #* 3. Define absolute path and check if it exists. If not create directory and/or optuna database
    path = Path(PROJECT_ROOT / dashboard_path)
    if not path.exists():
        if dashboard_type == "tensorboard":
            logger.info("Log directory for Tensorboard not found! Creating directory.")
            path.mkdir(parents=True, exist_ok=True)
        else:
            logger.info("Database file for Optuna not found! Creating new database.")
            path.parent.mkdir(parents=True, exist_ok=True)
            optuna.storages.RDBStorage(url=f"sqlite:///{path}",
                                       engine_kwargs={"connect_args": {"timeout": 300}}
            )

    #* 4. Start the process
    if dashboard_type == "tensorboard":
        command = ["tensorboard", "--logdir", str(path), "--port", str(port)]
    else:
        command = ["optuna-dashboard", f"sqlite:///{path}", "--port", str(port)]

    logger.info("Starting %s: %s", dashboard_type, " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)
    
    #* 5. Check for errors
    if process.poll() is not None:
        out, err = process.communicate()
        logger.warning("%s failed to start:\nOutput: %s\nError: %s", 
            dashboard_type, 
            out.decode() if out else "None", 
            err.decode() if err else "None")
        return
    
    #* 6. Open the dashboard in the browser
    url = f"http://localhost:{port}"
    try:
        webbrowser.open(url)
    except Exception as e:
        logger.warning("Failed to open browser automatically: %s", e)
    
    logger.info("%s started at http://localhost:%d", dashboard_type, port)
    

def close_dashboard(
    dashboard_type: str,
    port: int | None = None,
) -> None:
    """Terminate the dashboard running on port.
    
    Attempts to kill any process bound to the specified port for the given dashboard type.
    
    Args:
        dashboard_type (str): Type of dashboard, either "tensorboard" or "optuna"
        port (int): Port number for the dashboard. If None, uses default (6006 for TensorBoard,
            8080 for Optuna)
    
    Returns:
        None
    """
    
    if dashboard_type == "tensorboard":
        default_port = 6006
    elif dashboard_type == "optuna":
        default_port = 8080
    else:
        raise ValueError("dashboard_type must be 'tensorboard' or 'optuna'")

    port = port or default_port
    
    #* get process on port
    procs_on_port = get_processes_on_port(port)
    if not procs_on_port:
        logger.info("No processes found on port %d to kill.", port)
        return
    
    for p in procs_on_port:
        try:
            p.kill()
            p.wait()
        except Exception as e:
            logger.error("Error killing process %d: %s", p.pid, e)


#%% #>-------------- Startup and cleanup functions for models pipelines  ----------------------

def startup(config_dict: dict[str, Any]) -> None:
    """Common pipeline initialization
    
    Sets random seeds and starts requested dashboards based on configuration

    Args:
        config_dict (dict[str, Any]): Configuration dictionary containing initialization parameters
            including random_state, open_tensorboard, and open_optuna_dashboard, 
    
    Returns:
        None
    """
    logger.debug("Initializing random seed and dashboards")
    # Set random seeds
    set_all_seeds(seed=config_dict['random_state'])

    # Start dashboards if needed
    if config_dict['open_tensorboard']:
        start_dashboard(dashboard_type="tensorboard")

    if config_dict['open_optuna_dashboard']:
        start_dashboard(dashboard_type="optuna")


def cleanup(config_dict: dict[str, Any]) -> None:
    """Performs cleanup operations when the model pipeline ends.
    
    Registers dashboard closing functions with atexit based on configuration.
    
    Args:
        config_dict (dict[str, Any]): Configuration dictionary containing cleanup parameters
            including close_tensorboard and close_optuna_dashboard
    
    Returns:
        None
    """
    logger.debug("Performing cleanup operations")
    if config_dict['close_tensorboard']:
        atexit.register(lambda: close_dashboard("tensorboard"))
    
    if config_dict['close_optuna_dashboard']:
        atexit.register(lambda: close_dashboard("optuna"))
    
