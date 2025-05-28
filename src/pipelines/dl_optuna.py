"""
Optuna hyperparameter optimization for Darts forecasting models.

This module provides functionality for optimizing deep learning time series forecasting model's hyperparameters
using Optuna. It handles study creation, parameter sampling, study execution, and results logging.

Main functionalities:
- Setting up and executing complete Optuna studies
- Creating parameter distributions from search space definitions
- Calculating appropriate optimization dates to prevent data leakage
- Logging detailed optimization results

The module is designed to work with the other pipeline components (data_pipeline, dl_backtest)
to find optimal hyperparameters for time series forecasting models.

Usage example:
    from src.pipelines import dl_optuna as dl_opt
    
    optuna_results = dl_opt.execute_optuna_study(
        datasets=datasets['optuna'],
        model_config=model_config,
        param_search_space=param_search_space,
        config_dict=config_dict
    )
"""

#%% #>-------------------- Imports, logger and module exports --------------------

#* standard library
import logging
import time
from typing import Any, Callable

#* third-party libraries
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

#* Local imports
from src.pipelines import dl_backtest as dl_backtest
from src.utilities import dl_model_param_setup as dl_param_setup
from config import OPTUNA_STORAGE_PATH

#* Logger setup
logger = logging.getLogger(__name__)


#* Module exports
__all__ = [
    'execute_optuna_study',
    'create_n_optimize_study',
    'log_optuna_results',
    'create_trial_params',
    'compute_optuna_initial_date',
]

#%%#>----------------------------- Function: run optuna study  -----------------------------------           

def execute_optuna_study(
    datasets: dict[int, dict[str, Any]],
    model_config: dict[str, Any],
    param_search_space: dict[str, dict[str, Any]],
    config_dict: dict[str, Any],
) -> dict[str, Any]:
    """Execute a complete Optuna hyperparameter optimization study.

    This function orchestrates the entire optimization process:
    0. Creates an objective function for Optuna to optimize
    1. Sets up the correct format for the parameter search space using the create_trial_params function
    2. Sets up the final model parameters and total steps for the model with the updated search space
        using the setup_model_parameters function
    3. Runs the model across the defined windows using the run_dl_models_across_windows function
    4. Sets up and run the Optuna study with appropriate sampler and pruner via the create_n_optimize_study function

    Args:
        datasets (dict[int, dict[str, Any]],): Optuna datasets created by data_pipeline module
        model_config (dict[str, Any]): Model-specific configuration. Check model entry
                point scripts for the full list of keys expected for each model.
        param_search_space (dict[str, dict[str, Any]]): Parameter search space definition
                with parameter names and distributions.
        config_dict (dict[str, Any]): General configuration settings. Check model entry
            point scripts for the full list of keys expected and lower level functions

    Returns:
        dict[str, Any]: Optuna results:
            study (optuna.study.Study): completed optuna study object  
            best_params (dict[str, Any]): best hyperparameters and its values found during optimization
            best_value (float): Best objective value found during optimization
            trials (list[optuna.trial.FrozenTrial]): List containing the trials
    """
    
    logger.info(">>>>>>> Starting Optuna hyperparameter optimization")
    start_time = time.time()
    
    ###* 1. Create objective function for optuna study
    def objective(trial):
        
        ##* 2. Update parameters with trial values
        params_trial_tuning = create_trial_params(trial, param_search_space)
        
        trial_params_dict, trial_total_steps_dict = dl_param_setup.setup_model_parameters(
            datasets=datasets,
            config_dict=config_dict,
            model_config=model_config,
            params_for_update=params_trial_tuning,
            mode='optuna'
        )

        ##* 3. Run optuna study for each window 
        optunas_results_dict = dl_backtest.run_dl_models_across_windows(
            datasets = datasets,
            model_params_dict = trial_params_dict,
            total_steps_dict = trial_total_steps_dict,
            tensorboard_tag=f"trial_{trial.number}",
            mode='optuna',
            config_dict=config_dict,
        )
        
        ##* 4. Retrieve WAPE results and log it
        average_windows_wape = optunas_results_dict['window_df']['avg_wape_across_periods'].mean()
        wape_by_window = optunas_results_dict['window_df']['avg_wape_across_periods']
        wape_by_window = [np.round(value, 6) for value in wape_by_window.tolist()]
        trial.set_user_attr("wape_by_window", wape_by_window)  
        
        return average_windows_wape
        
    ###* 5. Run optuna study to optimize hyperparameters 
    optuna_study_dict = create_n_optimize_study(
        objective_function = objective,
        config_dict=config_dict,
    )

    ###* 3. Log study results (for more detailed analysis check optuna dashboard)
    log_optuna_results(optuna_study_dict['study'], top_n=3)

    logger.info("Optuna optimization completed successfully in %.1f hours (%.1f minutes)", 
                (time.time() - start_time)/3600,(time.time() - start_time)/60
    )
    
    return optuna_study_dict


def create_n_optimize_study(
    objective_function: Callable,
    config_dict: dict[str, Any],
) -> dict[str, Any]:
    """Create, configure, and run an Optuna study with the received objective function.
    
    This function:
    0. Configures the storage database for the study if not already created
    1. Configures the Optuna study based on provided settings
    2. Sets up appropriate sampler and pruner with customizable parameters
    3. Creates and optimize the study with the given objective function
    
    Args:
        objective_function (Callable): The objective function to optimize.
        config_dict (dict[str, Any]): Dictionary containing the following keys:
            - optuna_study_name (str): Name of the study for identification.
            - optuna_timeout (int | None): Maximum time (in seconds) for the study to run 
                (mutually exclusive with n_trials).
            - optuna_n_trials (int | None): Maximum number of trials for the study 
                (mutually exclusive with timeout).
            - optuna_direction (str): Direction of optimization ("minimize" or "maximize"). Defaults to 'minimize'.
            - optuna_storage_path (str | None): Path to the SQLite database for storing study results. 
                None for default path defined in .env.
            - sampler_class Type[BaseSampler]: Class of the Optuna sampler. Defaults to TPESampler.
            - sampler_kwargs (dict | None): Additional arguments to override default sampler parameters.
            - pruner_class Type[BasePruner]: Class of the Optuna pruner. Defaults to MedianPruner.
            - pruner_kwargs (dict | None): Additional arguments to override default pruner parameters.
            - random_state (int): Random seed for reproducibility. 

    Returns:
        dict[str, Any]: Optuna results:
            study (optuna.study.Study): completed optuna study object  
            best_params (dict[str, Any]): best hyperparameters and its values found during optimization
            best_value (float): Best objective value found during optimization
            trials (list[optuna.trial.FrozenTrial]): List containing the trials
    
    Raises:
        ValueError: If neither timeout nor n_trials is specified, or if both are specified
    """
    ###* 0. Unpack configuration dictionary
    study_name = config_dict.get('optuna_study_name', "study")
    timeout = config_dict.get('optuna_timeout', None)
    n_trials = config_dict.get('optuna_n_trials', None)
    direction = config_dict.get('optuna_direction', 'minimize')
    optuna_storage_path = config_dict.get('optuna_storage_path') or OPTUNA_STORAGE_PATH
    sampler_class = config_dict.get('sampler_class', TPESampler)
    sampler_kwargs = config_dict.get('sampler_kwargs', None)
    pruner_class = config_dict.get('pruner_class', MedianPruner)
    pruner_kwargs = config_dict.get('pruner_kwargs', None)
    random_state = config_dict.get('random_state', 123)
    
    ###* 1. Input validation
    if (n_trials is None and timeout is None) or (n_trials is not None and timeout is not None):
        raise ValueError("Exactly one of n_trials or timeout must be specified")
    
    ###* 1. Optuna studies setup (storage and study name)
    optuna_storage_full_path = f"sqlite:///{optuna_storage_path}/optuna_study.db"
        
    storage = optuna.storages.RDBStorage(
        url=optuna_storage_full_path,
        engine_kwargs={"connect_args": {"timeout": 300}}
    )

    ###* 2. Create sampler and pruner
    # default configurations
    sampler_defaults = {
        'seed': random_state,
        'constant_liar': True,
        'n_startup_trials': 10,
        'prior_weight': 1.0,
    }
    pruner_defaults = {
        'n_startup_trials': 10,
        'n_warmup_steps': 75,
        'interval_steps': 5,
    }
    # Merge defaults with any provided kwargs
    sampler_config = {**sampler_defaults, **(sampler_kwargs or {})}
    pruner_config = {**pruner_defaults, **(pruner_kwargs or {})}

    # Create sampler and pruner
    study_sampler = sampler_class(**sampler_config)
    study_pruner = pruner_class(**pruner_config)

    ###* 3. Create study
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        sampler=study_sampler,
        pruner=study_pruner,
        storage=storage,
        load_if_exists=True,
    )

    ###* 4. Run study
    study.optimize(
        objective_function, 
        timeout=timeout,            # timeout in seconds
        n_trials=n_trials,          
        show_progress_bar=True,
    )
    
    return {
        "study": study,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "trials": study.trials,
    }

     
def log_optuna_results(
    study: optuna.study.Study,
    top_n: int = 5,
) -> None:
    """Log detailed summary of Optuna study results.
    
    Generates and logs a comprehensive summary of the optimization study:
    - Overall study statistics (name, trials, best value)
    - Top N trials with their performance metrics and duration
    - Best trial's parameters with importance scores (if available)

    Args:
        study (optuna.study.Study): Completed study object.
        top_n (int, optional): Number of top trials to display (default: 5).
    
    Returns:
        None
    """
    #* Log general information from the study
    logger.info(">>>> Optuna study results")
    logger.info("Study name: %s", study.study_name)
    logger.info("Number of completed trials: %d", len(study.trials))
    wape_by_window = study.best_trial.user_attrs.get('wape_by_window', None)
    logger.info("Best value Mean WAPE: %.6f, WAPE by window: %s ", study.best_value, wape_by_window)
    
    #* 1. Sort trials by average WAPE
    sorted_trials = sorted(study.trials, key=lambda t: t.value)[:top_n]

    #* 2. Log top trials Results
    logger.info("Top %d Trials:", top_n)
    for i, trial in enumerate(sorted_trials):
        duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
        wape_by_window = study.best_trial.user_attrs.get('wape_by_window', None)
        
        logger.info("Rank %s: - trial#%s - Duration: %.2f hours (%.2f minutes)", 
                    i, trial.number, duration/3600, duration/60 )
        logger.info("Mean WAPE across windows: %.6f | WAPE by window: %s", 
                    trial.value, wape_by_window)
        
    
    #* 3. Log best trial's parameters with the importance for the study
    if len(sorted_trials) > 1:
        logger.debug("Best Trial Parameters values and importance of parameter for the study:")
        param_importances = optuna.importance.get_param_importances(
            study, 
            evaluator=optuna.importance.FanovaImportanceEvaluator(seed=123)
        )
        best_trial = study.best_trial
        sorted_params = sorted(best_trial.params.items(), key=lambda x: param_importances.get(x[0], 0), reverse=True)
    
        for param, value in sorted_params:
            importance = param_importances.get(param, 0.0)  
            logger.debug("%s: %s   (Importance: %.5f)", param, value, importance)
    else:
        logger.debug("Best Trial Parameters values:")
        best_trial = study.best_trial
        for param, value in best_trial.params.items():
            logger.debug("%s: %s", param, value)
            

#%%#>------------------ Function: create_trial_params  -----------------------------------           

def create_trial_params(
    trial: optuna.Trial,
    param_search_space: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Convert search space dictionary from entry point scripts to expected format for Optuna.
        
    Maps a search space dictionary to Optuna's suggest_* method calls based on parameter type.
    Supported parameter types to be passed by param_search_space:
        - Categorical:  {'type': 'categorical', 'values': [val1, val2, ...]}
        - Integer:      {'type': 'int', 'low': min_val, 'high': max_val}
        - Float:        {'type': 'float', 'low': min_val, 'high': max_val, 'log': bool}
            The 'log' flag enables logarithmic sampling (useful for variables with large ranges)

    Args:
        trial (optuna.Trial): Current Optuna trial
        param_search_space (dict[str, dict[str, Any]]): Dictionary defining parameter search spaces
            with structure detailed above.
            
    Returns:
        dict[str, Any]: Dictionary with parameter names and values for this trial in the expected
            optuna suggest_* format.
    """

    ##* Map parameter types to their corresponding trial suggestion functions
    suggestion_func = {
        'categorical': lambda name, config: trial.suggest_categorical(name, config['values']),
        'int': lambda name, config: trial.suggest_int(name, config['low'], config['high']),
        'float': lambda name, config: trial.suggest_float(
            name, config['low'], config['high'], log=config.get('log', False))
    }
    
    ##* Process all parameters in search space
    params_trial_tuning = {}
    for param_name, param_config in param_search_space.items():
        suggest_func = suggestion_func[param_config['type']]
        params_trial_tuning[param_name] = suggest_func(param_name, param_config)
        

    return params_trial_tuning

#%%#>------------------ Functions: Compute and validate optuna initial date  -----------------------------------           

def compute_optuna_initial_date(
    config_dict: dict[str, Any]
) -> pd.Timestamp:
    """Calculate the appropriate start date for Optuna optimization to avoid data leakage.
    
    Determines the starting date for optimization that ensures no data leakage into the backtest period. 
    Works backwards from the backtest start date, accounting for number of windows and forecast parameters.
    
    No period used in the optuna optimization should overlap with any of the test periods of the backtest.
            
    Args:
        config_dict (dict[str, Any]): Configuration containing:
            - backtest_start_date (str | pd.Timestamp): Start date for backtest
            - optuna_windows (int): Number of optimization windows (default: 1)
            - output_chunk_length (int): Length of forecast period
            - output_chunk_shift (int): Offset from base date to forecast start (default: 1)
    
    Returns:
        pd.Timestamp: Calculated start date for Optuna optimization
        
    Raises:
        ValueError: If backtest_start_date is missing or dates would create data leakage
    """
    #* unpacks variables from dictionary
    backtest_start_date = config_dict['backtest_start_date']
    optuna_windows = config_dict.get('optuna_windows', 1)
    output_chunk_length = config_dict.get('output_chunk_length', 3)
    output_chunk_shift = config_dict.get('output_chunk_shift', 1)
    
    #* validate input
    if backtest_start_date is None:
        raise ValueError("backtest_start_date is required to compute Optuna initial date")
    
    #* Ensures backtest_start_date is timestamp type
    backtest_start = pd.Timestamp(backtest_start_date)
    
    #* Calculate start base date for optuna
    base_date = backtest_start - pd.DateOffset(
        months = output_chunk_length + output_chunk_shift)
    
    # Calculate start date based on number of windows
    start_base_date_optuna = base_date - pd.DateOffset(months=optuna_windows-1)
    
    #* Check if dates wont create data leakage
    _validate_optuna_dates(
        start_base_date_optuna, backtest_start, optuna_windows, output_chunk_shift, output_chunk_length
    )
    
    return start_base_date_optuna
    
    
def _validate_optuna_dates(
    start_base_date_optuna: pd.Timestamp,
    backtest_start: pd.Timestamp,
    optuna_windows: int,
    output_chunk_shift: int,
    output_chunk_length: int,
) -> None:
    """Helper function to validate that `compute_optuna_initial_date` doesn't create data leakage.
    
    Ensures the final forecast period of Optuna doesn't overlap with backtest period,
    which would create data leakage and invalidate results.

    Args:
        start_base_date_optuna (pd.Timestamp): Starting date for Optuna optimization
        backtest_start (pd.Timestamp): Starting date for backtest
        optuna_windows (int): Number of optimization windows
        output_chunk_shift (int): Periods to shift between base date and forecast
        output_chunk_length (int): Number of periods in each forecast
    
    Returns:
        None
        
    Raises:
        ValueError: If Optuna forecast period overlaps with backtest period
    """
    end_base_date_optuna = start_base_date_optuna + pd.DateOffset(months=(optuna_windows-1))
    optuna_forecast_start_date = end_base_date_optuna + pd.DateOffset(months=output_chunk_shift)
    optuna_forecast_end_date = optuna_forecast_start_date + pd.DateOffset(months=output_chunk_length-1)

    if optuna_forecast_start_date >= backtest_start:
        raise ValueError(
            f"Optuna final forecast date overlaps with backtest dates, introducing data leakage. "
            f"Optuna final forecast period: {optuna_forecast_start_date} to {optuna_forecast_end_date} "
            f"Backtest start date: {backtest_start}"
        )
    
    


        