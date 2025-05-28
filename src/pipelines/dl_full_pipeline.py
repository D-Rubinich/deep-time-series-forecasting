"""
High-level pipeline orchestrator for Darts deep-learning models based on Pytorch Lightning.

The pipeline orchestrates the entire workflow in a structured sequence, using the config_dict,
model_config, and param_search_space dictionaries from the model entry point script to guide the process. 

It includes the following steps:
1. Environment setup (random seeds, dashboards, input validation)
2. Data preparation via the data pipeline 
3. Optuna hyperparameter optimization (optional) via optuna pipeline
4. Backtesting across multiple windows (optional) via backtest pipeline
5. Resource cleanup and results reporting

Usage Example:
    from src.pipelines import dl_full_pipeline as dl_pipe
    
    if __name__ == "__main__":
        dl_pipe.main(
            config_dict=config_dict, 
            model_config=model_config, 
            param_search_space=param_search_space,
        )

"""

#%% #>-------------------- Imports, logger and module exports --------------------

#* standard library
import logging
import time
import pprint
from typing import Any

#* Local imports
from src.pipelines import data_pipeline as data_pipe
from src.pipelines import dl_backtest as dl_backtest   
from src.pipelines import dl_optuna as dl_optuna  
from src.utilities import general_utils as utils

#* Logger setup
logger = logging.getLogger(__name__)

#* Module exports
__all__ = [
    'main',
    'execute_deep_learning_pipeline',
]

#%% #>-------------------- Def main() definition  -----------------------------

def main(
    config_dict: dict[str, Any],
    model_config: dict[str, Any],
    param_search_space: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Entry point for the deep learning pipeline execution.

    Logs configurations, executes the pipeline, handles exceptions, and returns results.
    This function is the main entry point called by the model scripts.
    
    Args:
        config_dict (dict[str, Any]): General configuration settings for data loading,
            training, optimization, and evaluation. 
        model_config (dict[str, Any]): Model-specific configuration including architecture,
            training parameters and callbacks.
        param_search_space (dict[str, Any]): Search space definition for Optuna hyperparameter tuning
   
    Returns:
        dict[str, Any]: Pipeline results containing optuna study and/or backtest results. Keys:
            optuna_study (dict[str, Any]): Optuna study results (if run). Keys:
                - study (optuna.study.Study): completed optuna study object  
                - best_params (dict[str, Any]): best hyperparameters and its values found during optimization
                - best_value (float): Best objective value found during optimization
                - trials (list[optuna.trial.FrozenTrial]): List containing the trials
            backtest_results (dict[str, pd.DataFrame]): Dictionary with backtest metrics. Keys:
                - window_df: window-level results containing metrics such as WAPE
                - series_df: series-level results containing metrics such as MAPE and predictions
    
    Raises:
        Exception: Propagates any error from the pipeline execution.
    
    Note:
        The full specifications for configuration dictionaries can be found in the
        model entry point scripts (e.g., run_tide_backtest.py, run_rnn_backtest.py).
    """
    logger.info(">>>>> Starting deep learning pipeline execution for %s model", config_dict['model_name'])
    logger.debug("General Configuration dictionary:\n%s", pprint.pformat(config_dict))
    logger.debug("Model configuration dictionary:\n%s", pprint.pformat(model_config))
    logger.debug("Parameter search space:\n%s", pprint.pformat(param_search_space))
    
    try:
        results_dict = execute_deep_learning_pipeline(
            config_dict=config_dict, 
            model_config=model_config, 
            param_search_space=param_search_space,
        )
        logger.info("Deep learning Pipeline execution completed successfully")
        return results_dict
    
    except Exception:
        logger.exception("Deep learning Pipeline execution failed")
        raise


#%% #>-------------------- Deep Learning Darts Models Pipeline  -----------------------------

def execute_deep_learning_pipeline(
    config_dict: dict[str, Any], 
    model_config: dict[str, Any], 
    param_search_space: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Orchestrates the complete deep learning forecasting pipeline.
    
    This function executes the entire pipeline workflow and is called by `main`. It includes:
        0. Validates inputs and initializes environment (seeds, dashboards)
        1. Processes data through the data pipeline
        2. Optionally runs Optuna hyperparameter optimization
        3. Optionally performs backtesting with optimal parameters
        4. Handles cleanup and logs execution time
    
    Args:
        config_dict (dict[str, Any]): General configuration settings for data loading,
            training, optimization, and evaluation. 
        model_config (dict[str, Any]): Model-specific configuration including architecture,
            training parameters and callbacks.
        param_search_space (dict[str, Any]): Search space definition for Optuna hyperparameter tuning

    Returns:
        dict[str, Any]: Pipeline results containing optuna study and/or backtest results. Keys:
            optuna_study (dict[str, Any]): Optuna study results (if run). Keys:
                - study (optuna.study.Study): completed optuna study object  
                - best_params (dict[str, Any]): best hyperparameters and its values found during optimization
                - best_value (float): Best objective value found during optimization
                - trials (list[optuna.trial.FrozenTrial]): List containing the trials
            backtest_results (dict[str, pd.DataFrame]): Dictionary with backtest metrics. Keys:
                - window_df: window-level results containing metrics such as WAPE
                - series_df: series-level results containing metrics such as MAPE and predictions
    
    Raises:
        ValueError: If data_format is invalid
        RuntimeError: If any pipeline component fails (data processing, Optuna, backtest) 
        
    Note:
        The full specifications for configuration dictionaries can be found in the
        model entry point scripts (e.g., run_tide_backtest.py, run_rnn_backtest.py).
    """
    ###* Start time and initialize results dictionary
    pipeline_start = time.time()
    results_dict = {}
    
    ###* unpack variables
    run_optuna = config_dict.get('run_optuna', True)
    run_backtest = config_dict.get('run_backtest', True)
    data_format = config_dict.get('data_format', 'univariate')
    
    try:
        ###* Validate inputs
        if data_format not in ['univariate', 'multivariate']:
            raise ValueError(f"Invalid data format: {data_format}. Choose 'univariate' or 'multivariate'")
        
        ###* Set random seed and start dashboards (if applicable)
        utils.startup(config_dict) 
        
        ###* Create datasets for backtest and optuna
        try:
            datasets = data_pipe.execute_data_pipeline(config_dict)
        except Exception as e:
            raise RuntimeError(f"Error in data processing: {str(e)}") from e

        ###* Run Optuna Study
        if run_optuna:
            try:
                results_dict['optuna_study'] = dl_optuna.execute_optuna_study(
                    datasets['optuna'],
                    model_config, 
                    param_search_space, 
                    config_dict
                )
            except Exception as e:
                raise RuntimeError(f"Error in Optuna optimization: {str(e)}") from e
        
        ###* Run Backtest
        if run_backtest:
            try:
                results_dict['backtest_results'] = dl_backtest.execute_backtest(
                    datasets['backtest'], 
                    config_dict, 
                    model_config
                )
            except Exception as e:
                raise RuntimeError(f"Error in backtesting: {str(e)}") from e
            
        ###* Cleanup
        utils.cleanup(config_dict)
        
        ###* Log completion
        dl_pipe_duration = (time.time() - pipeline_start)
        logger.info("Deep learning pipeline completed in %.1f hours (%.1f minutes)", 
                    dl_pipe_duration/3600, 
                    dl_pipe_duration/60
        )
        
        return results_dict
        
    except Exception as e:
        logger.error("Deep learning pipeline failed")
        if config_dict is not None:
            try:
                utils.cleanup(config_dict)
            except Exception as cleanup_error:
                logger.error("Cleanup failed after pipeline error: %s", cleanup_error)
        raise 
    
    
    