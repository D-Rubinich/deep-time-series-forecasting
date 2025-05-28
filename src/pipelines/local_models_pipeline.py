"""
Local (non-global) time series models pipeline.

This module provides functionality for running multiple univariate local forecasting models
in parallel across multiple backtest windows. Unlike deep learning models that train globally
across all time series, these models (like ARIMA, ETS, Prophet) train independently on each
individual time series.

The pipeline handles:
- Data preparation via the data pipeline
- Parallel model training and forecasting
- Prediction processing and evaluation
- Results collection and aggregation

Usage example:
    from src.pipelines import local_models_pipeline
    
    results = local_models_pipeline.execute_local_models_pipeline(
        models_config=models_config,
        config_dict=config_dict,
        n_jobs=-1  # Use all available cores
    )
"""

#%% #>-------------------- Imports, logger and module exports --------------------

#* standard library
import logging
import time
from typing import Any, List
import pprint

#* third-party libraries
import pandas as pd
from joblib import Parallel, delayed

#* Local imports
from src.utilities import general_utils as utils
from src.pipelines import data_pipeline as data_pipe
from src.utilities import evaluation_utils as eval_utils

#* Logger setup
logger = logging.getLogger(__name__)

#* Module exports
__all__ = [
    'main',
    'execute_local_models_pipeline',
    'run_local_models_across_windows',
    'run_local_model_single_window',
]

#%% #>-------------------- Def main() definition  -----------------------------


def main(
    config_dict: dict[str, Any],
    models_config: dict[str, dict[str, Any]],
    n_jobs: int = -1,
) -> dict[str, pd.DataFrame]:
    """Execute the backtest for each local model in parallel.

    Logs the configuration, executes the pipeline, saves results, and returns them.

    Args:
        config_dict (dict[str, Any]): Global backtest settings (timestamps, data paths, windows)
        models_config (dict[str, dict[str, Any]]): Dictionary mapping model names to 
        configurations containing:
            - model_class: The Darts model class to instantiate
            - params: Keyword arguments passed to the model constructor
        n_jobs (int, optional): Number of parallel jobs. -1 for all cores (default: -1)
    
    Returns:
        dict[str, pd.DataFrame]: 
            - `window_df`: aggregated window-level metrics dataframe
            - `series_df`: aggregated series-level metrics dataframe
            
    Raises:
        RuntimeError: On pipeline failure
    """
    logger.info("Starting local models pipeline execution for %s models", models_config.keys())
    logger.debug("General Configuration dictionary:\n%s", pprint.pformat(config_dict))
    logger.debug("Model configuration dictionary:\n%s", pprint.pformat(models_config))

    try:
        results_dict=execute_local_models_pipeline(
            models_config=models_config,
            config_dict=config_dict,
            n_jobs=n_jobs
        )
        logger.info("Local models pipeline execution completed successfully")
        return results_dict
    
    except Exception as e:
        logger.exception("Pipeline execution failed")
        raise


#%% #>-------------- Local Models Pipeline (univariate non-global models)  ------------------------

def execute_local_models_pipeline(
    models_config: dict[str, dict[str, Any]],
    config_dict: dict[str, Any],
    n_jobs: int = -1,
) -> dict[str, pd.DataFrame]:
    """Executes the full backtest pipeline for local univariate models
    
    This function:
    1. Builds datasets via the data pipeline  
    2. Runs in parallel backtests across windows and models 
    3. Saves CSV outputs via the general utils module
    
    Args:
        models_config (dict[str, dict[str, Any]]): Dictionary mapping model names to configurations
        config_dict (dict[str, Any]): General configuration settings
        n_jobs (int): Number of parallel jobs. -1 for all cores (default: -1)
    
    Returns:
        dict[str, pd.DataFrame]: 
            - `window_df`: aggregated window-level metrics dataframe
            - `series_df`: aggregated series-level metrics dataframe

    Raises:
        RuntimeError: If errors occur in data processing or backtesting
    """
    pipeline_start = time.time()
    #* Create datasets for backtest
    try:
        full_datasets = data_pipe.execute_data_pipeline(config_dict)
    except Exception as e:
        raise RuntimeError(f"Error in data processing: {str(e)}") from e

    #* Run Backtest
    try:
        backtest_results = run_local_models_across_windows(
            datasets=full_datasets['backtest'],
            config_dict=config_dict,
            models_config=models_config,
            n_jobs=n_jobs,
        )
    except Exception as e:
        raise RuntimeError(f"Error in backtesting: {str(e)}") from e
    
    #* Save results
    utils.save_backtest_results(
        results_dict=backtest_results,
        config_dict=config_dict,
    )

    #* Log completion
    pipe_duration = (time.time() - pipeline_start)
    logger.info("Local Models pipeline completed in %.1f hours (%.1f minutes)", 
                    pipe_duration/3600, pipe_duration/60)
    
    return backtest_results




#%% #>-------------------- Function Definition -------------------------------

def run_local_models_across_windows(
    datasets: dict[str, dict[str, Any]],
    config_dict: dict[str, Any],
    models_config: dict[str, dict[str, Any]],
    n_jobs: int = -1,
) -> dict[str, pd.DataFrame]:
    """Execute all local models backtests across all windows in parallel.
    
    Flattens (model, window) pairs into tasks and dispatches via joblib for
    efficient parallelization and then combines the results into window-level and
    series-level DataFrames.
    
    Args:
        datasets (dict[str, dict[str, Any]]): Dictionary of backtesting datasets by window
        config_dict (dict[str, Any]): General configuration settings
        models_config (dict[str, dict[str, Any]]): Configurations for each model to be tested
        n_jobs (int, optional): Number of parallel jobs. -1 for all cores (default: -1)

    Returns:
        dict[str, pd.DataFrame]:
            - `window_df`: window-level results sorted by model & window  
            - `series_df`: series-level results sorted by model, series_index, window
    """
    ##*  unpack variables
    backtest_timestamp = config_dict.get(
        'backtest_timestamp', 
        pd.Timestamp.now(tz='UTC').strftime("%Y%m%d_%H%M")
    )
    all_window_results = []
    all_series_results = []
    
    ##* Build a list of tasks for each (model, window) pair
    tasks = []
    for model_name, model_config in models_config.items():
        for window in sorted(datasets.keys()):
            tasks.append((model_name, model_config, window))
    
    ##* Process models in parallel
    logger.info(
        "Starting parallel processing of %s models and %s windows", len(models_config), len(datasets)
    )
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(run_local_model_single_window)(
            datasets=datasets[window],
            window=window,
            config_dict=config_dict,
            backtest_timestamp=backtest_timestamp,
            model_config=model_config,
            model_name=model_name,
        )
        for (model_name, model_config, window) in tasks
    )
    
    ##* Merge the individual dictionaries into final DataFrames
    for res in results_list:
        all_window_results.extend(res['window_results'])
        all_series_results.extend(res['series_results'])
    
    window_df = pd.DataFrame(all_window_results).sort_values(['model','window'])
    series_df = pd.DataFrame(all_series_results).sort_values(['model','series_index','window'])

    return {
        'window_df': window_df,
        'series_df': series_df,
    }


def run_local_model_single_window(
    datasets: dict[str, Any],
    window: int,
    config_dict: dict[str, Any],
    backtest_timestamp: pd.Timestamp,
    model_config: dict[str, Any],
    model_name: str,
) -> dict[str, list[dict[str, Any]]]:
    """Runs a single model on a single window and evaluates performance.

    This function is designed to be called in parallel for each (model, window) pair
    in the `run_local_models_across_windows` function. 
    
    This function:
    1. Trains the model on each series in the training set
    2. Generates and process predictions for each series in the test set
    3. Calculates series-level and window-level metrics
    
    Args:
        datasets (dict[str, Any]): Datasets for a specific window
        window (int):  Window identifier/index
        config_dict (dict[str, Any]): General configuration settings
        backtest_timestamp (pd.Timestamp): Timestamp for the backtest run
        model_config (dict[str, Any]): Configuration for the specific model
        model_name (str): Name of the model being evaluated

    Returns:
        dict[str, list[dict[str, Any]]]:
            - `window_results`: single-item list of window-level metrics dict  
            - `series_results`: list of per-series metrics dicts
        
    Raises:
        RuntimeError: If model training or prediction fails
    """
    
    logger.info("Running window %s for model %s", window, model_name)

    ###* Unpack variables
    model_class=model_config['model_class']
    model_params=model_config['params']
    prediction_length = config_dict['prediction_length']
    output_chunk_shift=config_dict['output_chunk_shift']
    scaler = config_dict.get('scaler', None)
    window_level_results = []
    series_level_results = []
    predictions_proc_window = []
    
    ###* Define datasets to use
    if scaler is None:
        train_series = datasets['ts_train']
        test_series = datasets['ts_test']
    else:
        train_series = datasets['ts_train_scaled']
        test_series = datasets['ts_test_scaled']
    
    ###* Loop through series
    for idx, (train, test) in enumerate(zip(train_series, test_series)):
        
        #* Initialize model
        model = model_class(**model_params)
    
        try:
            #* Fit model and predict
            model.fit(train)
            predictions_scaled = model.predict(n=output_chunk_shift+prediction_length)
            
            #* Process Predictions
            predictions_processed = eval_utils.process_predictions(
                actual_series = test,
                predictions_scaled=predictions_scaled, 
                scaler = scaler,
            )
            predictions_proc_window.append(predictions_processed)
            
            #* Calculate and store results - series level
            series_level_results.append(
                eval_utils.compute_series_level_metrics(
                    actual_ts=test,
                    predicted_ts=predictions_processed,
                    series_idx=idx,
                    config_dict=config_dict,
                    backtest_timestamp=backtest_timestamp,
                    window=window,
                    should_retrain=True,
                    updated_params_dict=model_config['params'],
                    base_date=datasets['base_date'],
                    model_name=model_name,
                )
            )
        except Exception as e:
            error_msg = (
                f"Error in model {model_name} for series idx {idx}: {str(e)}")
            raise RuntimeError(error_msg) from e
        
    ##* Calculate and store results - window level
    window_level_results.append(
        eval_utils.compute_window_level_metrics(
            datasets=datasets,
            predictions_processed=predictions_proc_window,
            config_dict=config_dict,
            backtest_timestamp=backtest_timestamp,
            window=window,
            should_retrain=True,
            updated_params_dict=model_config['params'],
            model_name=model_name,
        )
    )
    ##* 8. Log window results
    avg_wape_across_periods = window_level_results[-1].get('avg_wape_across_periods', None)
    logger.info("Model %s - Completed window %s with mean WAPE: %s", 
                model_name, window, avg_wape_across_periods)
    
    results = {
        'window_results': window_level_results, 
        'series_results': series_level_results
    }
    
    return results


