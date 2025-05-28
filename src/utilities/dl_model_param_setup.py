"""
Deep learning model parameter setup module.

Darts expects a specific nested dictionary structure for model parameters containing information
such as base model parameters, model architecture, PyTorch Lightning trainer parameters, 
callbacks, and learning rate scheduler settings. 

This module provides utility functions to seamlessly manage and update these parameters
for any deep learning model in the Darts library. 


Main functionalities include:
- Wrapper function to set up model parameters
- Retrieve best model's parameters from Optuna studies
- Configuring PyTorch Lightning trainer
- Setting up callbacks (early stopping, tensorboard, checkpointing)
- Calculating proper steps for OneCycle learning rate scheduler

Usage example:
    from src.utilities import dl_model_param_setup as dl_param_setup

    model_params_dict, total_steps_dict = dl_param_setup.setup_model_parameters( 
        datasets=datasets,
        config_dict=config_dict,
        model_config=model_config,
        mode='backtest'
    )
"""



#%% #>-------------------- Imports, logger and module exports --------------------

#* standard library
import logging
import math
from pathlib import Path
from typing import Any
from copy import deepcopy  
import pprint

#* third-party libraries
import pandas as pd
import optuna
from darts import TimeSeries
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import AdamW

#* Local imports
from config import (
    TENSORBOARD_PATH,
    TENSORBOARD_PATH_BACKTEST,
    TENSORBOARD_PATH_OPTUNA,
    OPTUNA_STORAGE_PATH,
)
#* Logger setup
logger = logging.getLogger(__name__)

#* Module exports
__all__ = [
    'setup_model_parameters',
    'retrieve_optuna_best_parameters',
    'update_params_dict',
    'setup_pytorch_trainer_params',
    'initialize_early_stopping',
    'initialize_tensorboard_logger',
    'initialize_model_checkpoint',
    'get_lr_scheduler_total_steps_per_window',
]

#%%#>------------- Function: setup_model_parameters  -----------------------

def setup_model_parameters(
    datasets: dict[str, Any],
    config_dict: dict[str, Any],
    model_config: dict[str, Any],
    params_for_update: dict[str, Any] | None = None,
    mode: str = "backtest",
) -> tuple[dict[str, Any], dict[str, int]]:
    """Wrapper to set up full model & scheduler parameters for backtest or Optuna.
    
    This function performs several steps:
    1a. For backtest runs:
        - retrieves best parameters from Optuna trials (if params_for_update is not provided)
        - If no study is found for the model, the function will fallback to original model_config parameters
    1b. For optuna runs:
        - params_for_update is expected to be provided with the structured parameters search space 
        created by the function `dl_optuna.create_trial_params()`
    2. Updates the model configuration with either the optimized parameters or search space
    3. Structures the training parameters for PyTorch Lightning
    4. Computes total steps for the OneCycle lr scheduler based
    
    Args:
        datasets (dict[str, Any]): Datasets created with `data_pipe.execute_data_pipeline()`
        config_dict (dict[str, Any]): General configuration dictionary defined in the
            entry-point scripts for each model at src/backtest
        model_config (dict[str, Any]): Base model configuration dictionary defined in the
            entry-point scripts for each model at src/backtest
        params_for_update (dict[str, Any], optional): parameters set that overrides the default
            model_config parameters. To be used either for trying a specific set of model parameters
            in backtest runs, or for providing the configured search space for optuna runs.
            If None, the function will try to retrieve the best parameters from the Optuna study. 
            If no study is found, the function will fallback to original model_config parameters.
        mode (str, optional): "backtest" or "optuna" to select trainer/logging paths.

    Returns:
        tuple[dict[str, Any], dict[str, int]]: 
            - params_dict: complete model parameters dictionary to be passed for model initialization.
            - total_steps_dict: Mapping of window identifiers to total scheduler steps.
    """
    
    logger.debug("Preparing model parameters...")
        
    ###* 0. Retrieve Optuna trial parameters if params_for_update is not provided
    if params_for_update is None:
        try:
            params_for_update = retrieve_optuna_best_parameters(config_dict)
        except RuntimeError as e:
            logger.warning(f"No Optuna trials found: {str(e)}. Falling back to original model_config.")
            params_for_update = {}
        except Exception as e:
            raise e
        
    ###* 1. Update parameters dict with best trial parameters
    params_dict = update_params_dict(params_for_update, model_config)

    ###* 2. Finalize trainer parameters
    params_dict['trainer_params'] = setup_pytorch_trainer_params(
        params_dict=params_dict,
        config_dict=config_dict,
        mode=mode,
    )

    ###* 3. Compute total steps for OneCycle learning rate scheduler
    total_steps_dict  = get_lr_scheduler_total_steps_per_window(
        datasets=datasets,
        model_params=params_dict,
    )
    
    logger.info("Model parameters setup completed.")
    
    return params_dict, total_steps_dict


#%%#>---------------------- Function: retrieve_optuna_best_parameters ---------------------------------   

def retrieve_optuna_best_parameters(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Selects the best set of parameters from either a specific Optuna study or all studies from a model.
    
    - If study_name is provided, gets trials from that specific study.
    - If only model_name is provided, finds the best trial across all studies for that model.
    
    Args:
        config_dict (dict[str, Any]): Dictionary containing:
            Required (one of the two):
                - study_name (str):  Name of the specific Optuna study to use
                - model_name: (str): Name of the model to filter studies for (e.g., 'TiDE')
            Optional keys:
                - trial_rank (int, optional): Rank of the trial to select (0 for best, 1 for second-best, etc.).
                - storage_url (str, optional): URL for the Optuna storage. If None uses default path.
                - optuna_direction (str, optional): Optimization direction to sort trials, either "minimize" or "maximize". 
    
    Returns:
        dict[str, Any]: Parameters set with the best score from the selected trial/model
    
    Raises:
        ValueError: If direction is invalid or both study_name and model_name are None.
        RuntimeError: If no studies or trials are found
    """
    ##* 0. Extract parameters from config_dict with defaults
    study_name = config_dict.get('study_name', None)
    model_name = config_dict.get('model_name', None)
    trial_rank = config_dict.get('trial_rank', 0)
    storage_url = config_dict.get('storage_url')
    direction = config_dict.get('optuna_direction', 'minimize')

    ##* 1. Validate inputs
    if direction not in ["minimize", "maximize"]:
        raise ValueError("direction must be either 'minimize' or 'maximize'")

    if study_name is None and model_name is None:
        raise ValueError("Either study_name or model_name must be provided.")

    ##* 2. Set default storage path if not provided
    if storage_url is None:
        storage_url = f"sqlite:///{OPTUNA_STORAGE_PATH}/optuna_study.db"
        
    ##* 3. Load studies
    study_summaries = optuna.get_all_study_summaries(storage=storage_url)
    if not study_summaries:
        raise RuntimeError("No Optuna studies found in the specified storage.")

    ##* 4. Load trials
    trials = []

    # Use specific study
    if study_name:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        trials = study.trials

    # Get all trials from all studies for the model
    else:
        model_studies = [s for s in study_summaries if s.study_name.startswith(f"{model_name}_utc")]
        if not model_studies:
            raise RuntimeError(f"No studies found for model: {model_name}")

        for summary in model_studies:
            study = optuna.load_study(study_name=summary.study_name, storage=storage_url)
            trials.extend([(t, summary.study_name) for t in study.trials
                           if t.value is not None]) # get only completed trials

    # Check if trials were found
    if not trials:
        error_msg = (f"No trials found for study: {study_name}" if study_name
                     else f"No trials found for model: {model_name}")
        raise RuntimeError(error_msg)

    ##* 5. Select trial
    # sort trials by value
    reverse_order = (direction == "maximize")
    sorted_trials = sorted(trials, key=lambda t: t[0].value, reverse=reverse_order)

    # check if requested rank is valid
    if trial_rank >= len(sorted_trials):
        logger.warning(f"Requested trial rank {trial_rank} exceeds available trials ({len(sorted_trials)}). Fallback trial = 0 ")
        trial_rank = 0

    # Get selected trial
    selected_trial, study_name  = sorted_trials[trial_rank]
    logger.debug(
        "\nSelected trial from study '%s' with direction %s and value: %.5f\n"
        "Parameters from best trial: \n%s",
        study_name, direction, selected_trial.value, pprint.pformat(selected_trial.params)
    )

    return selected_trial.params


#%%#>------------- Functions: update parameters dictionaries -----------------------


def update_params_dict(
    params_for_update: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Recursively updates a single dictionary (flat or nested) with values from params_for_update.

    This function makes a deep copy of the original dictionary and traverses the dictionaries in `model_config`
    replacing any keys found in `params_for_update` with their corresponding values.
    
    Args:
        params_for_update (dict[str, Any]): Overrides to apply.
        model_config (dict[str, Any]): Original parameter dictionary (can be flat or nested).

    Returns:
        dict[str, Any]: A new dictionary with updates applied.
    """
    updated_dict = deepcopy(model_config)

    for key, value in updated_dict.items():
        if isinstance(value, dict):
            updated_dict[key] = update_params_dict(params_for_update, value)
        elif key in params_for_update:
            old_value = updated_dict[key]
            updated_dict[key] = params_for_update[key]

    return updated_dict


#%%#>------------- Functions: setup Pytorch Lightning trainer -----------------------

def setup_pytorch_trainer_params(
    params_dict: dict[str, Any],
    config_dict: dict[str, Any],
    mode: str = 'backtest',
) -> dict[str, Any]:
    """Build PyTorch Lightning trainer settings including callbacks.
    
    Configures all parameters needed for PyTorch Lightning training with Darts, including optimizer settings, 
    learning rate scheduler, early stopping, tensorboard logging, and model checkpointing.
    
    Args:
        params_dict (dict[str, Any]): Must contain:
            - `trainer_params` (dict[str, Any]): Dictionary containing trainer specific parameters such as
                n_epochs, batch_size, and optimizer settings, etc. Defined in the entry-point scripts for 
                each model at src/backtest.
            - optional `early_stopping_kwargs`, `model_checkpoint_kwargs`.
        config_dict (dict[str, Any]): Should include:
            - `tensorboard_logger` (bool): Whether to enable TensorBoard logging (default: True).
            - `tensorboard_id` (str): ID to identify TensorBoard logging.
            - `use_modelcheckpoint_callback` (bool): Whether to enable model checkpointing (default: False).
            - path overrides: `tensorboard_path_backtest`, `tensorboard_path_optuna`.
        mode (str, optional):  Either 'optuna' or 'backtest' to determine tensorboard path (default: 'backtest')

    Returns:
        dict[str, Any]: Fully configured trainer parameters for model initialization in Darts.
    """
    ##* Unpack common configuration values
    trainer_params=params_dict['trainer_params']
    early_stopping_kwargs = params_dict.get('early_stopping_kwargs', None)
    model_checkpoint_kwargs = params_dict.get('model_checkpoint_kwargs', None)
    tensorboard_logger = config_dict.get('tensorboard_logger', True)
    tensorboard_id = config_dict.get('tensorboard_id', None)
    use_modelcheckpoint_callback = config_dict.get('use_modelcheckpoint_callback', False)
    

    # Choose the appropriate tensorboard path based on the mode.
    if mode == "backtest":
        tensorboard_path = config_dict.get('tensorboard_path_backtest') or TENSORBOARD_PATH_BACKTEST
    elif mode == "optuna":
        tensorboard_path = config_dict.get('tensorboard_path_optuna') or TENSORBOARD_PATH_OPTUNA
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be either 'optuna' or 'backtest'")
    
    ##* Setup early stopping 
    early_stopping = initialize_early_stopping(early_stopping_kwargs)

    ##* Setup tensorboard if requested
    tb_logger = None
    if tensorboard_logger:
        tb_logger = initialize_tensorboard_logger(
            tensorboard_id=tensorboard_id,
            tensorboard_path=tensorboard_path,
            add_timestamp=False,
            version=None
        )

    ##* PyTorch Lightning Trainer Parameters
    complete_trainer_params = {
        "optimizer_cls": trainer_params.get("optimizer_cls", AdamW),
        "n_epochs": trainer_params.get("n_epochs", 300),
        "batch_size": trainer_params.get("batch_size", 128),
        "pl_trainer_kwargs": {
            "accelerator": trainer_params.get("accelerator", "gpu"),
            "devices": trainer_params.get("devices", 1),
            "enable_progress_bar": trainer_params.get("enable_progress_bar", True),
            "enable_model_summary": trainer_params.get("enable_model_summary", True),
            "enable_checkpointing": trainer_params.get("enable_checkpointing", False),
            "check_val_every_n_epoch": trainer_params.get("check_val_every_n_epoch", 1),
            "log_every_n_steps": trainer_params.get("log_every_n_steps", 10),
            "gradient_clip_algorithm": trainer_params.get("gradient_clip_algorithm", "norm"),
            "gradient_clip_val": trainer_params.get("gradient_clip_val", 1.0),
            "logger": tb_logger,
            "callbacks": [early_stopping],
        },
        'optimizer_kwargs': {
            "weight_decay": trainer_params.get("weight_decay", 5e-3),
        },
    }
    
    ##* Setup model checkpointing if requested
    if use_modelcheckpoint_callback:
        model_checkpoint = initialize_model_checkpoint(
            model_checkpoint_kwargs=model_checkpoint_kwargs
        )
        complete_trainer_params['pl_trainer_kwargs']['callbacks'].append(model_checkpoint)
        

    return complete_trainer_params


#%%#>--------------- Functions setup callbacks ---------------------------

def initialize_early_stopping(
    early_stopping_kwargs: dict[str, Any] | None = None,
) -> EarlyStopping:
    """Create an EarlyStopping callback with defaults and overrides.

    Merges default patience, delta, monitor, and mode settings with any overrides in `early_stopping_kwargs`.
    
    Args:
        early_stopping_kwargs (dict[str, Any], optional): Overrides for
            `patience`, `min_delta`, `monitor`, `mode`, etc.

    Returns:
        EarlyStopping: Configured EarlyStopping callback instance
    """
    early_stopping_default = {
        "patience": 15,
        "min_delta": 1e-3,
        "monitor": "val_loss",
        "mode": "min",
        "verbose": False,
        "check_on_train_epoch_end": True
    }
    early_stopping_config = {
        **early_stopping_default,
        **(early_stopping_kwargs or {})
    }
    
    return EarlyStopping(**early_stopping_config)

def initialize_tensorboard_logger(
    tensorboard_id: str,
    add_timestamp: bool = False,
    tensorboard_path: str | Path | None = None,
    version: str | None = None,
) -> TensorBoardLogger:
    """Creates a TensorBoardLogger with specified parameters for experiment tracking.

    Args:
        tensorboard_id (str): ID to identify the experiment
        add_timestamp (bool, optional): Whether to add a UTC timestamp to logger name (default: False)
        tensorboard_path (str | Path, optional): Directory to save TensorBoard logs. 
            Defaults to `TENSORBOARD_PATH` from config.
        version (str, optional):  Explicit logger version. Defaults to None.
    
    Returns:
        TensorBoardLogger: Configured TensorBoard logger instance
    """

    logger_path = (
        tensorboard_path if tensorboard_path is not None
        else TENSORBOARD_PATH
    )
    logger_name = (
        f"{tensorboard_id}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
        if add_timestamp
        else tensorboard_id
    )

    logger = TensorBoardLogger(
        save_dir=str(logger_path),
        name=logger_name,
        version=version,
    )

    return logger

def initialize_model_checkpoint(
    model_checkpoint_kwargs: dict[str, Any] | None = None,
) -> ModelCheckpoint:
    """Create a ModelCheckpoint callback with defaults and overrides.
    
    Args:
        model_checkpoint_kwargs (dict[str, Any], optional): Custom settings for the 
            ModelCheckpoint callback, such as `save_top_k`, `monitor`, `mode`, etc.
    
    Returns:
        ModelCheckpoint: Configured ModelCheckpoint callback instance
    """
    model_checkpoint_default_kwargs = {
        'save_top_k':1,                 # Only save the best model
        'monitor':"val_loss",           # Monitor validation loss
        'mode':"min",
        'every_n_epochs':15,            # Only checkpoint every x epochs instead of every epoch
        'save_last':False,              # Don't save the last model
        'auto_insert_metric_name':True,
        'enable_version_counter':False,
    }
    
    model_checkpoint_config = {
        **model_checkpoint_default_kwargs,
        **(model_checkpoint_kwargs or {})
    }
    
    return ModelCheckpoint(**model_checkpoint_config) 


#%%#>--------------- Function: rate scheduler total steps calculation ---------------------------

def get_lr_scheduler_total_steps_per_window(
    datasets: dict[str, Any],
    model_params: dict[str, Any],
    set_key: str = "ts_train",
) -> dict[str, int]:
    """Precompute OneCycle scheduler steps for each backtest window based on dataset and model parameters.
    
    Iterates over `datasets`, calling `_get_lr_scheduler_total_steps` for each window key.

    Args:
        datasets (dict[str, Any]): Dictionary mapping window keys to dataset dictionaries.
        model_params (dict[str, Any]): Must include:
            `base_params['input_chunk_length']`; 
            `trainer_params['batch_size']`;
            `trainer_params['n_epochs']`;
        set_key (str, optional): Key for the training dataset to be used in each window for the calculations
            (default: 'ts_train')

    Returns:
        dict[str, int]: Dictionary mapping window keys to their calculated total steps
    """
    total_steps_dict = {
        window: _get_lr_scheduler_total_steps(
            datasets=datasets[window],
            input_chunk_length=model_params['base_params']['input_chunk_length'],
            batch_size=model_params['trainer_params']['batch_size'],
            n_epochs=model_params['trainer_params']['n_epochs'],
            set_key=set_key,
            #window=window,
        )
        for window in datasets.keys()
    }

    return total_steps_dict


def _get_lr_scheduler_total_steps(
    datasets: dict[str, Any],
    input_chunk_length: int,
    batch_size: int,
    n_epochs: int,
    set_key: str = 'ts_train',
) -> int:
    """Calculate the total optimizer steps for OneCycle scheduler for a specific dataset window.
    
    In PyTorch Lightning, a "step" refers to a single optimizer update, which typically 
    occurs after processing one batch of data. The total steps parameter defines how many 
    optimizer updates will occur throughout the entire training process across all epochs.
    
    OneCycleLR requires the total number of steps to be known in advance so it can change the 
    learning rate throughout the training process in its 3 phases (warm-up, annealing, and ramp down).     
    
    The calculation follows the logic:
    1. Calculate total_chunks based on time series length:
       - For each series: max(0, series_length - input_chunk_length + 1)
       - This represents all possible sliding windows in the time series (input-output pairs)
       
    2. Calculate steps_per_epoch = ceil(total_chunks / batch_size)
       - How many optimizer updates occur in one complete pass (epoch) through the data
       - This accounts for the last batch that might be smaller than batch_size
    
    3. total_steps = steps_per_epoch * n_epochs + 1
        - This represents the total number of optimizer updates across all training epochs
        - The +1 ensures the scheduler completes the full cycle
    
    Args:
        datasets (dict[str, Any]): Dictionary containing training dataset for one window. 
            key `set_key` selects the training series.
        input_chunk_length (int): Length of input sequences for the model.
        batch_size (int): Training batch size.
        n_epochs (int): Number of training epochs.
        set_key (str, optional): Key for the training dataset (default: 'ts_train').

    Returns:
        int: Total number of optimizer update steps that will occur during training
    
    Raises:
        TypeError: If dataset is not a TimeSeries object or a list of TimeSeries objects.
    """

    ##* 0. Get dataset
    ts_data = datasets[set_key]

    ##* 1. Calculate total chunks
    total_chunks = 0
    ##* a. Multivariate case - single TimeSeries object
    if isinstance(ts_data, TimeSeries) and (not ts_data.is_univariate):
        #if is a timeseries object in multivariate format:
        total_chunks = max(0, ts_data.n_timesteps - input_chunk_length + 1)

    ##* b. list of univariate TimeSeries objects
    elif isinstance(ts_data, list):
        for series in ts_data:
            total_chunks += max(0, series.n_timesteps - input_chunk_length + 1)
    else:
        raise TypeError("Unexpected type for selected dataset. Must be a TimeSeries object or a list of TimeSeries.")

    ##* 2. Calculate steps per epoch and total steps
    steps_per_epoch =  math.ceil(total_chunks / batch_size)
    total_steps = steps_per_epoch * n_epochs + 1

    return total_steps



