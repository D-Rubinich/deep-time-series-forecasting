"""
Configuration module for project paths.

This module dynamically loads environment variables from the `.env` file
and provides easy access to project paths.

Before using this module, run `setup_paths.py` first to create the `.env` file . 
(if running on .devcontainer, 'setup_paths.py' is run automatically during container build)
"""

import os
from pathlib import Path
from dotenv import dotenv_values

# Load environment variables from .env
PROJECT_ROOT  = Path(__file__).resolve().parent
env_path = PROJECT_ROOT / '.env'
config_vars = dotenv_values(env_path)  # Load all environment variables from .env into a dictionary

# Dynamically create module-level variables converting values to Path objects
for key, value in config_vars.items():
    if value:  # Optionally, you can add extra validation here
        globals()[key] = Path(value)

# define __all__ for easier imports
__all__ = list(config_vars.keys())

