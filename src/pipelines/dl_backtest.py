"""
Deep-learning backtest module for Darts forecasting models.

This module provides functionality for backtesting deep learning time series forecasting models
using expanding windows methodology. It handles parameters setup via the dl_model_param_setup module,
model initialization, training, prediction, and evaluation across multiple forecast windows.

Works with both univariate and multivariate time series data formats and supports various
covariates configurations. All functions are designed to be used with data created by the 
data_pipeline module and expects a specific data configuration dictionary structure with lists 
of TimeSeries objects.

Main functionalities:
- Executing complete backtests of deep learning models
- Running models across multiple time windows with configurable retraining strategy
- Training and predicting for individual windows
- Initializing models with correct parameters for each window
- Computing appropriate backtest start dates

Usage example:
    from src.pipelines import dl_backtest as dl_bt
                
    backtest_results = dl_bt.execute_backtest(
        datasets=datasets['backtest'],
        config_dict=config_dict,
        model_config=model_config
    )

"""

#%% #>-------------------- Imports, logger and module exports --------------------

#* standard library
import logging
import time
from pathlib import Path
from copy import deepcopy  
import pprint
from typing import Any

#* third-party libraries
import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

#* Local imports
from src.utilities import dl_model_param_setup as dl_param_setup
from src.pipelines import data_pipeline as data_pipe
from src.utilities import evaluation_utils as eval_utils
from src.utilities import general_utils as utils

#* Logger setup
logger = logging.getLogger(__name__)

#* Module exports
__all__ = [
    'execute_backtest',
    'run_dl_models_across_windows',
    'run_dl_model_single_window',
    'initialize_model_for_training',
    'compute_backtest_initial_date',
]

#%%#>---------------- Function: execute backtest pipeline ------------------

def execute_backtest(
    datasets: dict[int, dict[str, Any]],
    config_dict: dict[str, Any],
    model_config: dict[str, Any],
    save_results: bool = True,
) -> dict[str, pd.DataFrame]:
    """Runs the backtest pipeline for deep learning models.
    
    This function orchestrates the core of the backtest process:
        0. Receives datasets created by the data_pipeline module.
        1. Sets up model parameters using the parameter setup module
        2. Runs the model across all backtest windows
        3. Optionally saves the results to CSV files
    
    Args:
        datasets (dict[int, dict[str, Any]]): Dictionary of datasets for each window created 
            by the data_pipeline module.
        config_dict (dict[str, Any]): General configuration settings. Check model entry 
                point scripts for the full list of keys expected.
        model_config (dict[str, Any]): Model-specific configuration. Check model entry 
                point scripts for the full list of keys expected.
        save_results (bool): Whether to save results to CSV files (default: True)
    
    Returns:
        dict[str, pd.DataFrame]: Dictionary containing:
            - window_df: DataFrame with window-level metrics
            - series_df: DataFrame with series-level metrics
    """
    logger.info(">>>>>>> Starting backtest execution")
    start_time = time.time()
    
    ###* 1. Create models parameters dict
    model_params_dict, total_steps_dict = dl_param_setup.setup_model_parameters( 
        datasets=datasets,
        config_dict=config_dict,
        model_config=model_config,
        mode='backtest'
    )
            
    ###* 2. Run backtest for each window
    results_dict=run_dl_models_across_windows(
        datasets=datasets,
        model_params_dict=model_params_dict,
        total_steps_dict=total_steps_dict,
        tensorboard_tag=f"backtest",
        mode='backtest',
        config_dict=config_dict
    )
        
    ###* 3. Saves results to csv
    if save_results:
        utils.save_backtest_results(
            results_dict=results_dict,
            config_dict=config_dict,
        )
    
    logger.info("Backtest completed successfully in %.1f hours (%.1f minutes)", 
            (time.time() - start_time)/3600,(time.time() - start_time)/60
    )
                    
    return results_dict

#%%#>---------------- Function: Run across time windows  ------------------

def run_dl_models_across_windows(
    datasets: dict[int, dict[str, Any]],
    model_params_dict: dict[str, Any],
    total_steps_dict: dict[str, int],
    tensorboard_tag: str | None,
    mode: str,
    config_dict: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Train and evaluate deep learning models across multiple backtest windows.
    
    This function:
    0. Expects the datasets created by the data_pipeline module and parameters created by the 
        dl_model_param_setup module as inputs
    1. Retrain the model with fresh parameters (controlled by 'retrain' parameter)
        Note: always trains in the first window
    2. Make predictions using the trained model
    3. Process and evaluate the predictions
    4. Calculate window and series level metrics
    5. Returns dictionary containing dataframes with window and series level results
    
    Args:
        datasets (dict[int, dict[str, Any]]): Dictionary of datasets for each window created 
            by the data_pipeline module
        model_params_dict (dict[str, Any]): Dictionary containing model parameters created by the
            dl_model_param_setup module
        total_steps_dict (dict[str, int]): Pre-calculated training steps for each window created by the
            dl_model_param_setup module
        tensorboard_tag (str | None): Tag for tensorboard logging versions
        mode (str): 'backtest' or 'optuna', chooses retraining frequency and log paths.
        config_dict (dict[str, Any]): General configuration settings. Check model entry 
                point scripts for the full list of keys expected.
    
    Returns:
        dict[str, pd.DataFrame]: Dictionary containing:
            - window_df: DataFrame with window-level results 
            - series_df: DataFrame with series-level results
        
    Raises:
        ValueError: If mode is not 'optuna' or 'backtest'
        
    Notes:
        'multivariate' data format, predictions and test sets are split into univariate format 
            for metrics calculation to maintain consistency in the API.
    """
        
    ##* 0. unpack variables
    model_class = config_dict['model_class']
    data_format = config_dict.get('data_format', 'univariate')
    backtest_timestamp = config_dict.get(
        'backtest_timestamp', 
        pd.Timestamp.now(tz='UTC').strftime("%Y%m%d_%H%M")
    )
    if mode == "backtest":
        retrain = config_dict.get('backtest_retraining_frequency', False)
        logger.debug("Mode: backtest selected. Retraining frequency: %s", retrain)
    elif mode == "optuna":
        retrain = config_dict.get('optuna_retraining_frequency', False)
        logger.debug("Mode: optuna selected. Retraining frequency: %s", retrain)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be either 'optuna' or 'backtest'")
    
    ##* 1. Initialize results containers
    window_level_results = []
    series_level_results = []
    predictions_dict={}
    model = None
    
    ##* 2. Run backtest for each window
    for window in sorted(datasets.keys()):
        ##* 3. Checks if model should be retrained
        should_retrain = (
            retrain is True or 
            (retrain is False and window == 0) or 
            (isinstance(retrain, int) and retrain > 0 and window % retrain == 0) # Retrain every n windows (retrain is integer)
        )
        
        ##* 4. If retrain: Update parameters and initialize model
        if should_retrain:
            model, updated_params_dict = initialize_model_for_training(
                model_params_dict=model_params_dict,
                total_steps_dict=total_steps_dict,
                window=window,
                model_class=model_class,
                tensorboard_tag=tensorboard_tag,
                config_dict=config_dict,
            )
        
        ##* 5. Train/predict for current window
        logger.info(f"Running model pipeline for window {window}")
        predictions_dict[window] = run_dl_model_single_window(
            model=model,
            datasets=datasets[window],
            retrain=should_retrain,
            config_dict=config_dict,
        )
        
        ##* 6. If multivariate, splits back into univariate format for metrics calculation
        predictions_univariate = None
        test_set_univariate = None
        if data_format=='multivariate' and not predictions_dict[window]['predictions'][0].is_univariate:
            
            # Split multivariate predictions into list of univariate series
            predictions_univariate = data_pipe.split_multivariate_ts(
                predictions_dict[window]['predictions']
            )
            # Split multivariate predictions into list of univariate series
            test_set_univariate = data_pipe.split_multivariate_ts(
                datasets[window]['ts_test']
            )
        
        predictions = predictions_univariate or predictions_dict[window]['predictions']
        test_set = test_set_univariate or datasets[window]['ts_test']
        
        ##* 6. Calculate and store results - window level
        window_level_results.append(
            eval_utils.compute_window_level_metrics(
                datasets=datasets[window],
                predictions_processed=predictions,
                test_set=test_set,
                config_dict=config_dict,
                backtest_timestamp=backtest_timestamp,
                window=window,
                should_retrain=should_retrain,
                updated_params_dict=updated_params_dict,
            )
        )
            
        ##* 7. Calculate and store results - series level
        for idx, (actual_ts, pred_ts) in enumerate(zip(test_set, predictions)):
            series_level_results.append(
                eval_utils.compute_series_level_metrics(
                    actual_ts=actual_ts,
                    predicted_ts=pred_ts,
                    series_idx=idx,
                    config_dict=config_dict,
                    backtest_timestamp=backtest_timestamp,
                    window=window,
                    should_retrain=should_retrain,
                    updated_params_dict=updated_params_dict,
                    base_date=datasets[window]['base_date'],
                )
            )
    

        ##* 8. Log window results
        avg_wape_across_periods = window_level_results[-1].get('avg_wape_across_periods', None)
        logger.info("Completed window %s with mean WAPE: %s", window, avg_wape_across_periods)
    
    
    ##* Returns dataframes from results
    return {
        'window_df': pd.DataFrame(window_level_results),
        'series_df': pd.DataFrame(series_level_results),
    }



#%%#>---------------- Functions: run model single window (fit, predict, process) --------------

def run_dl_model_single_window(
    model: TorchForecastingModel,
    datasets: dict[str, Any],
    retrain: bool,
    config_dict: dict[str, Any],
) -> dict[str, Any]:
    """Train (optional), predict, and process forecasts for a single window.
    
    This function handles model training (if requested), prediction generation,
    and prediction processing for a single forecast window.
    
    Args:
        model (TorchForecastingModel): DARTS model object initialized with desired parameters. 
            If retrain is False, the model should be pre-trained. Should be initialized with the
            initialize_model_for_training function.
        datasets(dict[str, Any]): Dictionary containing scaled train/val/test datasets for time series/covariates 
            and fitted scaler for inverse transformation. Should be created with the data_pipeline module.
        retrain (bool): Whether to retrain the model or reuse a previously trained one passed as input in 'model' arg
        config_dict (dict[str, Any]): Dictionary containing configuration parameters. Keys:
            - prediction_length: Number of steps to predict
            - output_chunk_length: Prediction horizon length
            - output_chunk_shift: Prediction horizon shift
            - dataloader_kwargs: Parameters for the PyTorch dataloader (optional)
            - use_past_covariates: Whether to use past covariates
            - use_future_covariates: Whether to use future covariates
            - use_validation_set: Whether to use validation set
            - support_output_chunk_shift: Whether model supports output chunk shift
            - use_modelcheckpoint_callback: Whether to use model checkpointing
        
    Returns:
        dict[str, Any]: Dictionary containing:
            - predictions (list[TimeSeries]): List of processed predictions scaled back to original values
            - model (TorchForecastingModel): Fitted model object
    """
    ##* 0. Unpack variables
    prediction_length = config_dict.get('prediction_length', config_dict['output_chunk_length'])
    output_chunk_shift = config_dict.get('output_chunk_shift', 1)
    dataloader_kwargs = config_dict.get('dataloader_kwargs', None)
    use_past_covariates = config_dict.get('use_past_covariates', True)
    use_future_covariates = config_dict.get('use_future_covariates', True)
    use_validation_set = config_dict.get('use_validation_set', True)
    support_output_chunk_shift = config_dict.get('support_output_chunk_shift', True)
    use_modelcheckpoint_callback = config_dict.get('use_modelcheckpoint_callback', False)
    
    ##* 1. Default dataloader kwargs if not provided
    dataloader_defaults = {
        'num_workers': 8,  # Adjust based on your CPU cores
        'persistent_workers': True,
        'pin_memory': True,
        'prefetch_factor': 4,
    }
    dataloader_kwargs = {**dataloader_defaults, **(dataloader_kwargs or {})}
        
    ###* 2. Train model
    if retrain:
        model.fit(
            series=datasets['ts_train_scaled'],
            past_covariates=datasets['past_cov_train_scaled'] if use_past_covariates else None,
            future_covariates= datasets['future_cov_train_scaled'] if use_future_covariates else None,
            val_series=datasets['ts_validation_scaled'] if use_validation_set else None,
            val_past_covariates=datasets['past_cov_validation_scaled'] if use_past_covariates else None,
            val_future_covariates=datasets['future_cov_validation_scaled'] if use_future_covariates else None,
            dataloader_kwargs=dataloader_kwargs,
        )
    
    ###* 3. Retrieve Best Model from Training if modelcheckpoint callback is used
    if use_modelcheckpoint_callback:
        best_checkpoint_path = model.trainer.checkpoint_callback.best_model_path
        if best_checkpoint_path:
            # retrieve Lightning module from the darts wrapper
            lightning_module = model.model
            # Create a new instance of the same Lightning module class
            new_lightning_module = lightning_module.__class__.load_from_checkpoint(best_checkpoint_path)
            # Update the model with the loaded weights
            model.model = new_lightning_module
            logger.debug("Best model loaded using ModelCheckpoint callback: %s", best_checkpoint_path)
        
    
    ###* 4. Make predictions
    prediction_steps = prediction_length if support_output_chunk_shift else (prediction_length+output_chunk_shift)
    
    predictions_scaled = model.predict(
        n=prediction_steps,
        series=datasets['ts_validation_scaled'] if use_validation_set else datasets['ts_train_scaled'], # use the latest available data for the prediction
        past_covariates=datasets['past_cov_validation_scaled'] if use_past_covariates else None,
        future_covariates= datasets['future_cov_test_scaled'] if use_future_covariates else None,
    )
    
    ###* 5. Process Predictions
    # inverse transform and guarantee non negative values
    predictions_processed = eval_utils.process_predictions(
        actual_series=datasets['ts_test'],
        predictions_scaled=predictions_scaled, 
        scaler = datasets['ts_scaler'],
    )
    
    
    ###* 6. Return predictions and model
    return {
        'predictions': predictions_processed,
        'model': model
    }


#%%#>---------------- Function: initialize_model_for_training --------------

def initialize_model_for_training(
    model_params_dict: dict[str, Any],
    model_class: type[TorchForecastingModel],
    total_steps_dict: dict[str, int],
    window: int,
    config_dict: dict[str, Any],
    tensorboard_tag: str | None,
) -> tuple[TorchForecastingModel, dict[str, Any]]:
    """Prepare a new model instance with callbacks and schedulers for training in a backtest window.
            
    This function prepares a model instance for training by:
    1. Creating a deep copy of the parameters to avoid modifying originals
    2. Setting up a window-specific TensorBoard logger
    3. Initializing necessary callbacks (early stopping, checkpointing)
    4. Configuring learning rate scheduler with window-specific steps
    5. Creating and returning the model instance
    
    Args:       
        model_params_dict (dict[str, Any]): Dictionary containing model parameters created by the
            dl_model_param_setup module.
        model_class (type[TorchForecastingModel]): The Darts model class to instantiate
        total_steps_dict (dict[str, int]): Pre-calculated training steps for each window created by the
            dl_model_param_setup module.
        window (int): Current backtest window index to retrieve the correct training steps and logger information.
        config_dict (dict[str, Any]): Global settings. Key: 'use_modelcheckpoint_callback'. 
            If not provided, defaults to False.
        tensorboard_tag (str | None): Tag for tensorboard logging or None.
        
    Returns:
        tuple[TorchForecastingModel, dict[str, Any]]:
            - Initialized model instance
            - Updated parameters dictionary with window-specific settings and callbacks
    """
    ##* Make a copy to avoid modifying the original
    updated_params  = deepcopy(model_params_dict)
    
    
    ##* Initialize early stopping
    early_stopping_kwargs = updated_params.get('early_stopping_kwargs', None)
    early_stopping = dl_param_setup.initialize_early_stopping(early_stopping_kwargs)
    
    ##* Initialize Logger
    current_tb_logger = updated_params['trainer_params']['pl_trainer_kwargs']['logger']
    new_tb_logger = dl_param_setup.initialize_tensorboard_logger(
            tensorboard_id=current_tb_logger.name,
            tensorboard_path=Path(current_tb_logger.root_dir).parent,
            version=f"{tensorboard_tag}_window_{window}" if tensorboard_tag else None
    )

    ##* Update work_dir for model
    work_dir = Path(f"{Path(current_tb_logger.root_dir).parent}/{current_tb_logger.name}/{tensorboard_tag}_window_{window}")
    work_dir.mkdir(parents=True, exist_ok=True)
    updated_params['base_params']['work_dir'] = work_dir
    
    ##* update model dictionary    
    updated_params['trainer_params']['pl_trainer_kwargs'].update({
        'logger': new_tb_logger,
        'callbacks': [early_stopping],
    })
    
    ##* Initialize Model Checkpoint
    use_modelcheckpoint_callback = config_dict.get('use_modelcheckpoint_callback', False)
    if use_modelcheckpoint_callback:
        model_checkpoint_kwargs = updated_params['model_checkpoint_kwargs']
        model_checkpoint = dl_param_setup.initialize_model_checkpoint(
            model_checkpoint_kwargs=model_checkpoint_kwargs
        )
        updated_params['trainer_params']['pl_trainer_kwargs']['callbacks'].append(model_checkpoint)
        
    
    ##* Update learning rate scheduler steps
    updated_params['lr_scheduler_params']['lr_scheduler_kwargs']['total_steps'] = (
        total_steps_dict[window]
    )
    
    ##* Initialize new model
    model = model_class(
        **updated_params['base_params'],
        **updated_params['model_architecture_params'],
        **updated_params['trainer_params'],
        **updated_params['lr_scheduler_params'],
    )
    logger.debug("Model initialized for training in window %s.", window)
    logger.debug("Model parameters: \n%s", pprint.pformat(model))
    
    return model, updated_params 




#%%#>---------------- Function: compute backtest initial date --------------

def compute_backtest_initial_date(
    ts_list: list[TimeSeries] | TimeSeries,
    config_dict: dict[str, Any],
) -> pd.Timestamp:
    """Computes the appropriate start date for backtesting.
    
    Calculates the start date based on:
    - The end date of available data
    - Number of windows desired for backtesting
    - Output chunk length and shift parameters
    
    Args:
        ts_list (list[TimeSeries] | TimeSeries): List of TimeSeries objects from DARTS library
            or a single TimeSeries object.
        config_dict (dict[str, Any]): Configuration containing:
            - backtest_windows: Number of backtest windows desired
            - output_chunk_length: Length of the forecast period
            - output_chunk_shift: Shift of the forecast period (default=1)
    
    Returns:
        pd.Timestamp: The calculated start date for the backtest
    """
    #* unpacks variables from dictionary
    backtest_windows = config_dict['backtest_windows']
    output_chunk_length = config_dict.get('output_chunk_length', 3)
    output_chunk_shift = config_dict.get('output_chunk_shift', 1)
    
    #* Get the earliest end date to ensure all series have data
    end_date = (min([series.end_time() for series in ts_list]) 
                if isinstance(ts_list, list) 
                else ts_list.end_time())
    
    #* Calculate the base date 
    base_date = end_date - pd.DateOffset(
        months= output_chunk_length + output_chunk_shift - 1) 
    
    #* Calculate start date based on number of windows
    start_base_date_backtest = base_date - pd.DateOffset(months=backtest_windows-1)
    
    return start_base_date_backtest





            
            


            