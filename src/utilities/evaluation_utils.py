"""
Evaluation utilities module for the project.

This module provides functions for evaluating time series forecasting models,
processing predictions, calculating error metrics (WAPE, MAPE, APE) and format 
backtest results.

Main functionality includes:
- Processing predictions and aligning time indexes
- Computing error metrics (WAPE, MAPE, APE) for window and series levels

All functions work with Darts TimeSeries objects (single TimeSeries or lists of TimeSeries) 
and follow consistent error handling for mismatched time indices or components. 


Usage example:
    from src.utilities import evaluation_utils
    
    processed_predictions = evaluation_utils.process_predictions(
        actual_series=test_series,
        predictions_scaled=model_predictions,
        scaler=scaler
    )
    
    # Calculate WAPE metrics
    metrics = evaluation_utils.calculate_wape(
        actual_series=test_series,
        predicted_series=processed_predictions
    )
"""


#%% #>-------------------- Imports, logger and module exports --------------------

#* standard library
import logging
from typing import Any

#* third-party libraries
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mape, ape

#* Logger setup
logger = logging.getLogger(__name__)

#* Module exports
__all__ = [
    'process_predictions', 
    'align_time_indexes',
    'compute_window_level_metrics',
    'compute_series_level_metrics',
    'calculate_wape', 
]

#%%#>---------------- Functions to process predictions --------------

def process_predictions(
    actual_series: list[TimeSeries] | TimeSeries, 
    predictions_scaled: list[TimeSeries] | TimeSeries, 
    scaler: Any | None = None,
) -> list[TimeSeries] | TimeSeries:
    """Processes model predictions for evaluation.
    
    This function performs three main steps:
    1. Aligns (clip) time indexes from predictions to match actual series
    2. Inverse transforms to original scale the scaled predictions if a scaler is provided
    3. Ensures non-negative values in the predictions (business requirement)
    
    Accepts either a single TimeSeries object or a list of TimeSeries objects for both actuals and predictions.
    
    Args:
        actual_series (list[TimeSeries] | TimeSeries): Actual values as TimeSeries or list of TimeSeries objects
        predictions_scaled (list[TimeSeries] | TimeSeries): Scaled predictions as TimeSeries or list of TimeSeries objects
        scaler: Optional scaler object fitted to the training data with inverse_transform method for reversing scaling. Pass
            ``None`` if data are already in the original scale.
        
    Returns:
        list[TimeSeries] | TimeSeries: Processed predictions as TimeSeries or list of TimeSeries objects. 
            If a single series is passed in, a single series is returned; otherwise a list is returned.
    """
    #* Check if input type is a single TimeSeries object and convert to list to be processed
    single_input = False
    if isinstance(actual_series, TimeSeries) and isinstance(predictions_scaled, TimeSeries):
        actual_series = [actual_series]
        predictions_scaled = [predictions_scaled]
        single_input = True
        
    #* align time indexes of predictions with actuals (necessary for models that does not have output_chunk_shift)
    preds_scaled_aligned = align_time_indexes(
        actual_series=actual_series,
        predicted_series=predictions_scaled
    )
    
    #* Inverse transform to original scale
    preds_aligned = (scaler.inverse_transform(preds_scaled_aligned) 
                     if scaler else preds_scaled_aligned)
    
    #* Ensure non-negative values
    preds_processed = [ts.map(lambda x: np.maximum(0, x)) for ts in preds_aligned]
    
    if single_input:
        preds_processed = preds_processed[0]
        
    return preds_processed


def align_time_indexes(
    actual_series: list[TimeSeries] | TimeSeries, 
    predicted_series: list[TimeSeries] | TimeSeries, 
) -> list[TimeSeries]:
    """Aligns predicted series time indexes with actual series by clipping out-of-bounds values.
    
    Creates a new list of TimeSeries objects where each predicted series is sliced to intersect with
    the exactly same time index as its corresponding actual series.
    
    Args:
        actual_series (list[TimeSeries] | TimeSeries): list of Darts TimeSeries or TimeSeries containing actual values
        predicted_series (list[TimeSeries] | TimeSeries): list of Darts TimeSeries or TimeSeries containing predictions
        
    Returns:
        list[TimeSeries]: The aligned predicted series. Always returns a list, even when called with single series
        
    Raises:
        ValueError: If count of series in list differs or time indexes don't match after alignment
    """
    #* Initialize variables
    predictions_aligned = []
    
    #* Check if input type is a single TimeSeries object and convert to list to be processed 
    if isinstance(actual_series, TimeSeries) and isinstance(predicted_series, TimeSeries):
        actual_series = [actual_series]
        predicted_series = [predicted_series]
    
    ###* list[Timeseries] as inputs
    if isinstance(actual_series, list) and isinstance(predicted_series, list):
        # Checks if list of TimeSeries Objects have same length
        if len(actual_series) != len(predicted_series):
            raise ValueError(
                f"Number of actual series ({len(actual_series)}) does not match "
                f"number of predicted series ({len(predicted_series)})."
            )
        #* loop through each pair of actual and predicted series and align them
        for i, (actual, pred) in enumerate(zip(actual_series, predicted_series)):
            # Align time indexes
            aligned_pred = pred.slice_intersect(actual)
            
            # Check if time indexes match
            if not aligned_pred.time_index.equals(actual.time_index):
                error_msg = (
                    f"Predicted series #{i} does not perfectly match the actual series time index after clipping.\n"
                    f"Actual time_index: {actual.time_index}\n"
                    f"Clipped aligned_pred time_index: {aligned_pred.time_index}.\n"
                    f"Original pred time_index: {pred.time_index}.\n"
                )
                raise ValueError(error_msg)
            
            # Append aligned predictions
            predictions_aligned.append(aligned_pred)
            
    return predictions_aligned


#%%#>------------------------ Functions to calculate metrics ----------------------------


def compute_window_level_metrics(
    datasets: dict[str, Any],
    predictions_processed: list[TimeSeries],
    config_dict: dict[str, Any],
    backtest_timestamp: pd.Timestamp,
    window: int,
    should_retrain: bool,
    updated_params_dict: dict[str, Any],
    test_set: list[TimeSeries] | None = None,
    test_set_key: str = "ts_test",
    model_name: str | None = None,
) -> dict[str, Any]:
    """Computes window-level forecast metrics.
    
    Calculates aggregate WAPE for one forecast window across all time series and 
    store results and additional information in a dictionary.

    Args:
        datasets (dict[str, Any]): Dictionary containing dataset splits
        predictions_processed (list[TimeSeries]): Processed model predictions
        config_dict (dict[str, Any]): Configuration dictionary with model settings
        backtest_timestamp (pd.Timestamp): Timestamp when backtest was run
        window (int): Window index in the backtest
        should_retrain (bool): Whether model was retrained for this window
        updated_params_dict (dict[str, Any]): Model hyperparameters used for this window
        test_set (list[TimeSeries], optional): Test set if not using datasets[test_set_key]
        test_set_key (str, optional): Key for test set in datasets dict (default: 'ts_test')
        model_name (str, optional): Name of model (defaults to class name in config_dict)

    Returns:
        dict[str, Any]: Window-level metrics including WAPE and metadata
    """
    ##* Unpack variables
    window_results = {}
    test_set = test_set or datasets[test_set_key]
    model_name = model_name or config_dict.get('model_name', config_dict['model_class'].__name__)
    should_retrain = should_retrain or config_dict.get('retrain', None)
    round_decimals = config_dict.get('round_decimals', 6)
    
    ##* Check inputs (list have same length, match components and match time indexes)
    _validate_series_inputs(test_set, predictions_processed)
    
    #* Calculate WAPE metrics
    wape_dict=calculate_wape(
        actual_series=test_set,
        predicted_series=predictions_processed,
        round_decimals=round_decimals
    )
    ##* 6. Store results - window level
    window_results={
        'backtest_date': backtest_timestamp,
        'model': model_name,
        'window': window,
        'base_date': datasets['base_date'],
        'retrained': should_retrain,
        'forecast_start': test_set[0].time_index[0],
        'forecast_end': test_set[0].time_index[-1],
        'avg_wape_across_periods': wape_dict['avg_wape_across_periods'],
        'wape_per_period': str(wape_dict['wape_per_period']),
        'model_hyperparameters': str(updated_params_dict),
    }
    
    return window_results
    
def compute_series_level_metrics(
    actual_ts: TimeSeries,
    predicted_ts: TimeSeries,
    series_idx: int,
    config_dict: dict[str, Any],
    backtest_timestamp: pd.Timestamp,
    window: int,
    base_date: pd.Timestamp,
    should_retrain: bool,
    updated_params_dict: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Computes series-level forecast metrics.
    
    Calculates metrics (MAPE, APE) for all single time series in one forecast window.
    
    Args:
        actual_ts (TimeSeries): Actual values as TimeSeries
        predicted_ts (TimeSeries): Predicted values as TimeSeries
        series_idx (int): Index of the series
        config_dict (dict[str, Any]): Configuration dictionary with model settings
        backtest_timestamp (pd.Timestamp): Timestamp when backtest was run
        window (int): Window index in the backtest
        base_date (pd.Timestamp): Base date for the forecast window
        should_retrain (bool): Whether model was retrained for this window
        updated_params_dict (dict[str, Any], optional): Model hyperparameters used for this window
        model_name (str, optional): Name of model (defaults to class name in config_dict)
        
    Returns:
        dict[str, Any]: Series-level metrics including MAPE, APE and metadata
    """
    ##* Check inputs (list have same length, match components and match time indexes)
    _validate_series_inputs(actual_ts, predicted_ts)
    
    #* unpack variables
    series_results = {}
    model_name = model_name or config_dict.get('model_name', config_dict['model_class'].__name__)
    round_decimals = config_dict.get('round_decimals', 6)
    
    #* calculate mape and ape for each series
    series_mape = np.array(mape(actual_ts, predicted_ts)) / 100
    series_ape = np.array(ape(actual_ts, predicted_ts)) / 100
    
    #* Round results to specified decimals
    series_mape = np.round(series_mape, round_decimals)
    series_ape = np.round(series_ape, round_decimals)
    
    #* Store results - series level
    series_results={
        'backtest_date': backtest_timestamp,
        'model': model_name,
        'series_index': series_idx,
        'series_name': actual_ts.components[0], 
        'window': window,
        'base_date': base_date,
        'retrained': should_retrain,
        'forecast_start': actual_ts.time_index[0],
        'forecast_end': actual_ts.time_index[-1],
        'mape': series_mape,
        'ape': str(series_ape),
        'actual_values': actual_ts.values().ravel().tolist(),
        'forecast_values': predicted_ts.values().ravel().tolist(),
        'model_hyperparameters': str(updated_params_dict),
    }
    
    return series_results


def calculate_wape(
    actual_series: list[TimeSeries] | TimeSeries, 
    predicted_series: list[TimeSeries] | TimeSeries, 
    round_decimals: int = 6,    
) -> dict[str, Any]:
    """Calculates WAPE (Weighted Absolute Percentage Error) for time series forecasts.
    
    WAPE is calculated as: sum_across_series(|actual - predicted|) / sum_across_series(actual) for each period.
       
    Args:
        actual_series (list[TimeSeries] | TimeSeries): list of TimeSeries or TimeSeries containing actual values
        predicted_series (list[TimeSeries] | TimeSeries): list of TimeSeries or TimeSeries containing predictions
        round_decimals (int, optional): Number of decimals to round results to (default: 6)
        
    Returns:
        dict[str, Any]: Dictionary containing:
            - 'avg_wape_across_periods': Average WAPE across all periods
            - 'wape_per_period': list of WAPE values for each forecast period
            
    Raises:
        ValueError: If inputs don't match in number of series, components, or time indices           
    """
    ##* 0. Check inputs (list have same length, match components and match time indexes)
    _validate_series_inputs(actual_series, predicted_series)
            
    ##* 1. Convert to numpy arrays for vectorized operations
    actuals = np.array([series.values().ravel() for series in actual_series])
    predictions = np.array([series.values().ravel() for series in predicted_series])
    if actuals.shape != predictions.shape:
        error_msg = (
            f"Shape mismatch: actuals {actuals.shape} != predictions {predictions.shape}\n"
            f"Ensure all series have the same length and components.")
        raise ValueError(error_msg)
    
    ##* 2. Calculate WAPE 
    abs_errors = np.abs(actuals - predictions) # Calculate absolute errors (element-wise operations)
    sum_abs_errors = np.sum(abs_errors, axis=0) # Sum absolute errors across series for each period
    sum_actuals = np.sum(actuals, axis=0)
    wape_per_period = np.where(
        sum_actuals != 0, 
        sum_abs_errors / sum_actuals, 
        np.nan
    )
    
    ##* 3. Round WAPE Results
    avg_wape_across_periods = np.round(np.mean(wape_per_period), round_decimals)
    wape_per_period = [np.round(value, round_decimals) for value in wape_per_period.tolist()]
    
    
    ##* 4. Store WAPE by period and average WAPE across all periods
    results_dict = {
        'avg_wape_across_periods': avg_wape_across_periods,
        'wape_per_period': wape_per_period,
    }
    
    return results_dict




def _validate_series_inputs(
    actual_series: list[TimeSeries],
    predicted_series: list[TimeSeries], 
) -> None:
    """Validate that two lists of TimeSeries match in length, components, and time indexes.
   
    Args:
        actual_series (list[TimeSeries]): Actual values
        predicted_series (list[TimeSeries]): Predicted values
        
    Returns:
        None
        
    Raises:
        ValueError: If any validation checks fail
    """
    if len(actual_series) != len(predicted_series):
        raise ValueError("Number of actual and predicted TimeSeries objects must match")

    for i, (actual, pred) in enumerate(zip(actual_series, predicted_series)):
        if actual.components[0] != pred.components[0]:
            raise ValueError(f"Series components mismatch for series index {i}: "
                             f"{actual.components} != {pred.components}")

        if not actual.time_index.equals(pred.time_index):
            raise ValueError(f"Time index mismatch for series index {i}: "
                             f"{actual.time_index} != {pred.time_index}")
    

    
    
    