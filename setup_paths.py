"""
Script to setup project paths and create or update a .env file.
Script is automatically run when setting up the .devcontainer
"""
from pathlib import Path
import logging

# Basic logging configuration for the setup script.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def setup_project_paths():
    """
    Create project directories and update the .env file with relative paths.
    """
    # Use script location as project root
    PROJECT_ROOT = Path(__file__).resolve().parent
    logger.info("Setting up project at: %s", PROJECT_ROOT)
   
    # Define directories relative to project root
    directories = {
        "PROJECT_ROOT": ".",  
        "DATA_DIR": "data",
        "LOGS_DIR": "logs",
        "REPORTS_DIR": "reports",
        "SRC_DIR": "src",
        "TENSORBOARD_PATH": "logs/tensorboard_logs/",
        "TENSORBOARD_PATH_BACKTEST": "logs/tensorboard_logs/backtest",
        "TENSORBOARD_PATH_OPTUNA": "logs/tensorboard_logs/optuna",
        "OPTUNA_STORAGE_PATH": "data/optuna/",
        "BACKTEST_RESULTS_PATH": "data/backtest_results",
    }
   
    # Create directories if they don't exist
    for dir_path in directories.values():
        if dir_path != ".":  # Don't try to create project root
            path = PROJECT_ROOT / dir_path
            path.mkdir(parents=True, exist_ok=True)
            logger.debug("Created/verified directory: %s", path)
   
    # Path to .env
    env_path = PROJECT_ROOT / '.env'
    existing_lines = set()

    # If .env exists, read its content
    if env_path.exists():
        with open(env_path, 'r') as f:
            existing_lines = set(f.readlines())
            logger.debug("Read existing .env file")
    
    # Add paths to .env, skipping duplicates
    with open(env_path, 'a') as f:
        f.write("\n# Project paths - added/updated by setup_paths.py\n")
        for key, dir_path in directories.items():
            line = f"{key}=${{PROJECT_ROOT}}/{dir_path}\n" if key != "PROJECT_ROOT" else f"{key}={PROJECT_ROOT}\n"
            if line not in existing_lines:  # Avoid duplicates
                f.write(line)
                logger.debug("Added to .env: %s", line.strip())
      
        
    logger.info("Directories structured:")
    for dir_name, dir_path in directories.items():
        logger.info(" %s: %s", dir_name, dir_path)
        
    logger.info("Environment file updated at: %s", env_path)

if __name__ == "__main__":
    setup_project_paths()
