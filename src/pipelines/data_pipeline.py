"""
Data processing pipeline for time series forecasting.

The data pipeline orchestrates the entire data preparation process from raw data (and covariates)
to model-ready datasets for both Optuna optimization and backtesting scenarios.

Main functionalities:
- Loading and transforming data and covariates into Darts TimeSeries objects (univariate or multivariate)
- Creating train/validation/test splits and scaling data across rolling windows for backtesting and Optuna

Usage example:
    from src.pipelines import data_pipeline as data_pipe
    
    # Execute the full data pipeline
    datasets = data_pipe.execute_data_pipeline(config_dict)
        
    ts_main, ts_past_cov, ts_future_cov = data_pipe.create_univariate_timeseries(
        df=data_df, 
        config_dict=config_dict,              
    )
"""

#%% #>-------------------- Imports, logger and module exports --------------------

#* standard library
import logging
import time
from typing import Any

#* third-party libraries
import numpy as np
import pandas as pd
from darts import TimeSeries, concatenate
from sklearn.preprocessing import MinMaxScaler
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer

#* Local imports
from src.pipelines import dl_optuna as dl_optuna
from src.pipelines import dl_backtest as dl_backtest

#* Logger setup
logger = logging.getLogger(__name__)

#* Module exports
__all__ = [
    'execute_data_pipeline',
    'create_univariate_timeseries',
    'create_multivariate_timeseries',
    'split_scale_datasets',
    'split_multivariate_ts',
]

#%% #>-------------------- Data Transformation Pipeline  -----------------------------

def execute_data_pipeline(
    config_dict: dict[str, Any]
) -> dict[str, Any]:
    """Runs the complete data processing pipeline for time series forecasting.
    
    This function orchestrates the entire data pipeline:
    1. Loads data from specified paths in 'config_dict'
    2. Transforms data into Darts TimeSeries objects (univariate or multivariate)
    3. Creates appropriate train/validation/test splits for backtest and/or optuna
    4. Handles scaling if requested
    5. Returns dictionary containing TimeSeries objects and datasets for backtest and optuna

    Args:
        config_dict (dict[str, Any]): defined in model entry point containing the keys:
            - data_format (str): 'univariate' or 'multivariate'
            - data_path (str): Path to main data file
            - covariates_path (str): Path to covariates file (for multivariate)
            - run_backtest (bool): Whether to create backtest datasets
            - run_optuna (bool): Whether to create optuna datasets
            - backtest_start_date (Optional[str]): Start date for backtest
            - optuna_start_date (Optional[str]): Start date for optuna (if false will be calculated to prevent data leakage)
            - series_id (str): Column name for series identifier
            - time_index (str): Column name for datetime index
            - main_series (str): Column name for target variable
            - Other keys passed to functions called inside the pipeline:
                - use_static_covariates (bool): Whether to use static covariates
                - use_past_covariates (bool): Whether to use past covariates
                - use_future_covariates (bool): Whether to use future covariates 
                - use_validation_set (bool): Whether to create validation sets
                - backtest_windows (int): Number of backtest windows
                - optuna_windows (int): Number of optuna windows
                - output_chunk_shift (int): Offset in time steps to shift start of forecast
                - output_chunk_length (int): Number of time steps to forecast
                - prediction_length (int): Number of time steps to forecast
                - scaler: A scikit-learn-like scaler object with `fit_transform`, `transform` 
                        and `inverse_transform` methods. If None, no scaling is applied.
 
    Returns:
        dict[str, Any]: Dictionary containing:
            timeseries (dict[str, list[TimeSeries])
                - main_series (list[TimeSeries]): List of main series TimeSeries objects
                - past_covariates (list[TimeSeries]): List of past covariates TimeSeries objects (optional)
                - future_covariates (list[TimeSeries]): List of future covariates TimeSeries objects (optional)
            backtest (dict[int, dict[str, Any]]): Backtest datasets (if run_backtest=True)
            optuna (dict[int, dict[str, Any]]): Optuna datasets (if run_optuna=True)

    Raises:
        RuntimeError: If errors occur during data loading or processing
        ValueError: If an invalid data format is specified
    """
    start_time = time.time()
    logger.info(">>>>>>> Starting data pipeline execution")
    
    #*unpack config_dict
    data_format = config_dict.get('data_format', 'univariate')
    data_path = config_dict.get('data_path', None)
    covariates_path = config_dict.get('covariates_path', None)
    run_backtest = config_dict.get('run_backtest', True)
    run_optuna = config_dict.get('run_optuna', False)
    backtest_start_date = config_dict.get('backtest_start_date', None)
    optuna_start_date = config_dict.get('optuna_start_date', None)
    series_id = config_dict.get('series_id', 'series_id')
    time_index = config_dict.get('time_index', 'time_index')
    main_series = config_dict.get('main_series', 'value')
    
    #* Load dataset
    try:
        # Main data set
        df = pd.read_parquet(data_path)  
        logger.info("Loaded dataset from %s", data_path)
        
        # Covariates
        if data_format != 'univariate':
            covariates_df = pd.read_parquet(covariates_path)
            logger.info("Loaded covariates from %s", covariates_path)
            
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {str(e)}") from e
    
    #* Create DARTS TimeSeries objects (list of Univariate Timeseries) 
    try:
        logger.info("Transforming data to Darts TimeSeries Objects")
        if data_format == 'univariate':
            ts_main, ts_past_cov, ts_future_cov = create_univariate_timeseries(
                df, 
                series_id=series_id, 
                time_index_col=time_index, 
                main_series_col=main_series, 
                config_dict=config_dict,      
            )
        elif data_format == 'multivariate':
            ts_main, ts_past_cov, ts_future_cov = create_multivariate_timeseries(
                df, 
                covariates_df,
                series_id=series_id, 
                time_index_col=time_index, 
                main_series_col=main_series, 
                config_dict=config_dict,      
            )
        else:
            raise ValueError(f"Invalid data format: {data_format}. Must be 'univariate' or 'multivariate'")
    except Exception as e:
        raise RuntimeError(f"Error transforming data to Darts TimeSeries: {str(e)}") from e
    
    #* Create backtest datasets
    # compute backtest start date
    config_dict['backtest_start_date'] = (backtest_start_date
        or dl_backtest.compute_backtest_initial_date(ts_list=ts_main,config_dict=config_dict)
    )
    if run_backtest:
        try:
            logger.info("Creating Backtest datasets. Backtest start base date: %s", config_dict['backtest_start_date'])
            # crete backtest datasets
            datasets = split_scale_datasets(
                ts_main_series=ts_main,  
                ts_past_covariates=ts_past_cov,
                ts_future_covariates=ts_future_cov, 
                mode='backtest',
                config_dict=config_dict,
            )
            logger.info("Backtest datasets created. Number of windows: %s", len(datasets))
        except Exception as e:
            raise RuntimeError(f"Error creating Backtest datasets: {str(e)}") from e
    
    #* Create optuna datasets
    if run_optuna:
        try:
            # compute optuna start date
            config_dict['optuna_start_date'] = (optuna_start_date
                or dl_optuna.compute_optuna_initial_date(config_dict=config_dict)
            )
            logger.info("Creating Optuna datasets. Optuna start base date: %s", config_dict['optuna_start_date'])
            
            # create optuna datasets
            optuna_datasets = split_scale_datasets(
                ts_main_series=ts_main,  
                ts_past_covariates=ts_past_cov,
                ts_future_covariates=ts_future_cov, 
                mode='optuna',
                config_dict=config_dict,
            )
            logger.info("Optuna datasets created. Number of windows: %s", len(optuna_datasets))
        except Exception as e:
            raise RuntimeError(f"Error creating Optuna datasets: {str(e)}") from e
        
    
    #* Return datasets
    datasets = {
        'timeseries': {'main_series': ts_main,},   
        'optuna': optuna_datasets if run_optuna else None,
        'backtest': datasets if run_backtest else None,
    }
    if ts_past_cov is not None:
        datasets['timeseries']['past_covariates'] = ts_past_cov
    
    if ts_future_cov is not None:
        datasets['timeseries']['future_covariates'] = ts_future_cov
    
    logger.info("Data pipeline completed successfully in %.1f minutes", (time.time() - start_time)/60)
    return datasets


#%% #>---------- Functions: Create Time Series (univariate and multivariate) -----------------------

def create_univariate_timeseries(
    df: pd.DataFrame,
    series_id: str = 'series_id',
    time_index_col: str = 'time_index',
    main_series_col: str = 'value',
    config_dict: dict[str, Any] | None = None
) -> tuple[list[TimeSeries], list[TimeSeries] | None, list[TimeSeries] | None]:
    """Convert a long-format DataFrame into a list of univariate Darts TimeSeries objects.
    
    Creates three sets of time series:
    1. Main series (with optional static covariates)
    2. Past covariates (optional)
    3. Future covariates (optional)
    
    Args:
        df (pd.DataFrame): Input dataframe in long format - one row per time step per series
        series_id (str): Column identifying each time series (default: 'series_id')
        time_index_col (str): Name of datetime column (default: 'time_index')
        main_series_col (str): Name of target variable column (default: 'value')
        config_dict (Optional[dict[str, Any]]): Configuration with keys:
            - use_static_covariates (bool): Whether to include static covariates in main series
            - use_past_covariates (bool): Whether to create past covariates object
            - use_future_covariates (bool): Whether to create future covariates object   
    
    Returns:
        tuple[list[TimeSeries], list[TimeSeries] | None, list[TimeSeries] | None]:
            main_series (list[TimeSeries]): List of TimeSeries for target variables
            past_covariates (list[TimeSeries] | None): List of TimeSeries for past covariates (or None)
            future_covariates (list[TimeSeries] | None): List of TimeSeries for future covariates
    """
    logger.info("Timeseries data format: list of univariate TimeSeries")
    ##* 0. unpack variables
    use_static_covariates = config_dict.get('use_static_covariates', True)
    use_past_covariates = config_dict.get('use_past_covariates', True)
    use_future_covariates = config_dict.get('use_future_covariates', True)
    
    ##* 1. Create a list of univariate Darts TimeSeries Objects for the main series 'value'
    # Define columns for static covariates if requested
    static_covariates_cols = (
        [col for col in df.columns if col.startswith('static_')] 
        if use_static_covariates else None
    )

    # ensure time_index is datetime
    df['time_index'] = pd.to_datetime(df['time_index']) 

    # Create TimeSeries objects
    ts = TimeSeries.from_group_dataframe(
        df,
        group_cols=series_id,                  # Column identifying each series
        time_col=time_index_col,                # Datetime column
        value_cols=main_series_col,             # Target variable
        static_cols=static_covariates_cols,     # Static covariates
        fill_missing_dates=True,            
        fillna_value=0,                     
        freq='MS',
    )
    # change the components names to the series_id for better identification if needed it
    ts = [series.with_columns_renamed(series.columns[0], series.static_covariates[series_id].values[0]) 
          for series in ts]
    
    ##* 2. Create a list of univariate Darts TimeSeries Objects for past covariates
    past_covariates = None
    if use_past_covariates is True:
        past_covariates_cols = [col for col in df.columns if col.startswith('past_')]
        past_covariates = TimeSeries.from_group_dataframe(
            df,
            group_cols=series_id,              
            time_col=time_index_col,            
            value_cols=past_covariates_cols,    
            static_cols=None,
            fill_missing_dates=True,            
            fillna_value=0,                     
            freq='MS',
        )
    
    ##* 3. Create a list of univariate Darts TimeSeries Objects for future covariates
    future_covariates = None
    if use_future_covariates is True:
        future_covariates_cols = [col for col in df.columns if col.startswith('future_')]
        future_covariates = TimeSeries.from_group_dataframe(
            df,
            group_cols=series_id,              
            time_col=time_index_col,                  
            value_cols=future_covariates_cols, 
            static_cols=None, 
            fill_missing_dates=True,            
            fillna_value=0,                     
            freq='MS',
        )
    
    ##* 4. Transform data to float32
    ts = [series.astype('float32') for series in ts]
    past_covariates = ([series.astype('float32') for series in past_covariates] 
                       if past_covariates else None)
    future_covariates = ([series.astype('float32') for series in future_covariates] 
                         if future_covariates else None)
    
    ##* 5. Log debug information
    logger.debug("Darts TimeSeries Objects information from function create_univariate_timeseries:")
    _log_timeseries_information(ts, "Main Series")
    if past_covariates:
        _log_timeseries_information(past_covariates, "Past Covariates")
    if future_covariates:
        _log_timeseries_information(future_covariates, "Future Covariates")
    
    return ts, past_covariates, future_covariates


def create_multivariate_timeseries(
    df: pd.DataFrame,
    covariates_df: pd.DataFrame,
    series_id: str = 'series_id',
    time_index_col: str = 'time_index',
    main_series_col: str = 'value',
    config_dict: dict[str, Any] | None = None
) -> tuple[list[TimeSeries], list[TimeSeries] | None, list[TimeSeries] | None]:
    """Convert a long-format DataFrame into a multivariate Darts TimeSeries objects
    
    Creates three sets of time series (main series, past covariates, future covariates).
    
    For API consistency with the univariate format, each result is wrapped in a list containing 
    a single TimeSeries object.
    
    For the covariates we need to pass a separate preprocessed dataframe containing the covariates,
    if we create the covariates from the original main dataframe (as in create_univariate_timeseries),
    we have the risk of duplicating many times the same information and have performance issues.
        - E.g.: a covariate containing an aggregate data, like `market growth`, has a single value for 
            each time step, however in a long format dataframe, this value appear many times for each 
            series-time step, creating a duplication in the multivariate format. 
       
    Args:
        df (pd.DataFrame): Input dataframe in long format - one row per time step per series
        covariates_df (pd.DataFrame): Dataframe containing past and future covariates.
        series_id (str): Column identifying each time series (default: 'series_id')
        time_index_col (str): Name of datetime column (default: 'time_index')
        main_series_col (str): Name of target variable column (default: 'value')
        config_dict (Optional[dict[str, Any]]): Configuration with keys:
            - use_static_covariates (bool): Whether to include static covariates in main series
            - use_past_covariates (bool): Whether to create past covariates object
            - use_future_covariates (bool): Whether to create future covariates object  

    Returns:
        tuple[list[TimeSeries], list[TimeSeries] | None, list[TimeSeries] | None]:
            _series (list[TimeSeries]): List of TimeSeries for target variables
            st_covariates (list[TimeSeries] | None): List of TimeSeries for past covariates (or None)
            ture_covariates (list[TimeSeries] | None): List of TimeSeries for future covariates

    """
    logger.info("Timeseries data format: single Multivariate TimeSeries")
    #* unpack variables
    use_past_covariates = config_dict.get('use_past_covariates', True)
    use_future_covariates = config_dict.get('use_future_covariates', True)
    use_static_covariates = config_dict.get('use_static_covariates', True)
    df[time_index_col] = pd.to_datetime(df[time_index_col]) 
    covariates_df[time_index_col] = pd.to_datetime(covariates_df[time_index_col])
    
    ##* Define covariates columns
    static_covariates_cols = ([col for col in df.columns if col.startswith('static_')] 
                              if use_static_covariates else None)
    
    past_covariates_cols = ([col for col in covariates_df.columns if col.startswith('past_')] 
                            if use_past_covariates else None)
    
    future_covariates_cols = ([col for col in covariates_df.columns if col.startswith('future_')] 
                              if use_future_covariates else None)
    
     ##* Create main series multivariate TimeSeries objects
    # start by creating a list of univariate TimeSeries objects for each group
    ts_univ_list = TimeSeries.from_group_dataframe(
        df,
        group_cols=series_id,                   
        time_col=time_index_col,                
        value_cols=main_series_col,             
        static_cols=static_covariates_cols,     
        fill_missing_dates=True,            
        fillna_value=0,                     
        freq='MS',
    )
    # change the components names to the series_id for better identification
    ts_univ_list = [
        ts.with_columns_renamed(ts.columns[0], ts.static_covariates[series_id].values[0]) 
        for ts in ts_univ_list
    ]
    # concatenate into multivariate format
    ts_multivariate = (concatenate(ts_univ_list, axis=1)) #axis=1 to concatenate components

    ##* Create Past Covariates
    past_covariates_mv = None
    if use_past_covariates:
        past_covariates_mv = TimeSeries.from_dataframe(
            covariates_df,
            time_col=time_index_col,            
            value_cols=past_covariates_cols,    
            fill_missing_dates=True,            
            fillna_value=0,                     
            freq='MS',
        )
    
    ##* Create Future Covariates
    future_covariates_mv = None
    if use_future_covariates:
        future_covariates_mv = TimeSeries.from_dataframe(
            covariates_df,
            time_col=time_index_col,            
            value_cols=future_covariates_cols,    
            fill_missing_dates=True,            
            fillna_value=0,                     
            freq='MS',
        )
        
    ##* Transform data to float32 and list
    ts_multivariate = [ts_multivariate.astype('float32')]
    past_covariates_mv = [past_covariates_mv.astype('float32') if past_covariates_mv else None]
    future_covariates_mv = [future_covariates_mv.astype('float32') if future_covariates_mv else None]
    
    ##* Log debug information
    logger.debug("Darts TimeSeries Objects information from function create_multivariate_timeseries:")
    _log_timeseries_information(ts_multivariate, "Main Series")
    if use_past_covariates:
        _log_timeseries_information(past_covariates_mv, "Past Covariates")
    if use_future_covariates:
        _log_timeseries_information(future_covariates_mv, "Future Covariates")
        
    return ts_multivariate, past_covariates_mv, future_covariates_mv


def _log_timeseries_information(
    series_list: list[TimeSeries] | TimeSeries | None,
    series_name: str
) -> None:
    """Helper function to log information about DARTS TimeSeries objects.
    
    Logs type information, length, component details, and time ranges
    for TimeSeries objects, helping with debugging and verification.
    
    Args:
        series_list (list[TimeSeries] | TimeSeries | None): Series to inspect.
        series_name (str): Label to use in log messages.
        
    Returns:
        None
    """
    # Check if series_list is None
    if series_list is None:
        logger.debug('%s: is None', series_name)
        return
    
    #* List of TimeSeries objects
    if isinstance(series_list, list):
        unique_starts = {s.start_time() for s in series_list}
        unique_ends = {s.end_time() for s in series_list}
        data_format = 'univariate' if series_list[0].is_univariate else 'multivariate'
        total_components = sum([s.n_components for s in series_list])
        
        
        message = (
            f"\n{series_name} info:\n"
            f"  Data Type: {type(series_list)}  -  Length of list: {len(series_list)}"  
            f"    -  Total components across all series: {total_components}\n"
            f"  Element type: {type(series_list[0])}  -  Format: {data_format}\n"
            f"  Start period {unique_starts} - End periods in series: {unique_ends}"
        )
        logger.info(message)
       
    #* Single TimeSeries object 
    if isinstance(series_list, TimeSeries):
        data_format = 'univariate' if series_list[0].is_univariate else 'multivariate'
        message = (
            f"\n{series_name} info:\n"
            f"  Data Type: {type(series_list)}  -  Format: {data_format}"
            f"    -  Total components: {series_list.n_components}\n "
            f"  Start period: {series_list.start_time()} - End period: {series_list.end_time()}\n"
        )
        logger.info(message)


#%% #>------------------- Function: Split and scale datasets -----------------------------

def split_scale_datasets(
    ts_main_series: list[TimeSeries],
    ts_past_covariates: list[TimeSeries] | None,
    ts_future_covariates: list[TimeSeries] | None,
    mode: str,
    config_dict: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    """Split and scale lists of TimeSeries objects into train/validation/test for each window.
    
    Creates rolling forecast windows with proper train, validation (optional),
    and test splits for each window. Optionally applies scaling to each split.
    
    Args:
        ts_main_series (list[TimeSeries]): Main target series.
        ts_past_covariates (list[TimeSeries] | None): Past covariates.
        ts_future_covariates (list[TimeSeries] | None): Future covariates.
        mode (str): Either 'optuna' or 'backtest' to determine settings
        config_dict (dict[str, Any]): Configuration containing:
            - backtest_start_date/backtest_windows (for 'backtest')
            - optuna_start_date/optuna_windows (for 'optuna')
            - prediction_length (int): Number of time steps to forecast
            - output_chunk_shift (int): Offset in time steps to shift start of forecast
            - scaler: A scikit-learn-like scaler object with `fit_transform`, `transform` 
                    and `inverse_transform` methods. If None, no scaling is applied.
            - use_validation_set (bool): Whether to create validation sets
            - use_past_covariates (bool): Whether to use past covariates
            - use_future_covariates (bool): Whether to use future covariates 
     
    Returns:
        dict[int, dict[str, Any]]: Maps window index to a dict with keys: 
            base_date: Forecast base date for the window
            train_end_date: End date for training data
            forecast_period: Start and end dates for forecast period
            ts_train(_scaled): Training data scaled and unscaled for main series, 
                past and future covariates (covariates only if requested)
            ts_validation(_scaled): Validation data scaled and unscaled, if requested,
                for main series, past and future covariates (covariates only if requested)
            ts_test(_scaled): Test data scaled and unscaled for main series, past and 
                future covariates (covariates only if requested)
            ts_scaler: Fitted scaler for future inverse transformation of predictions
    
    Raises:
        ValueError: If mode is not 'optuna' or 'backtest'
    """
    ###* 0. Unpack variables
    if mode == 'backtest':
        forecast_base_date = config_dict['backtest_start_date']
        windows = config_dict['backtest_windows']
    elif mode == 'optuna':
        forecast_base_date = config_dict['optuna_start_date']
        windows = config_dict['optuna_windows']
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be either 'optuna' or 'backtest'")
    
    prediction_length = config_dict.get('prediction_length', 3)
    output_chunk_shift = config_dict.get('output_chunk_shift', 1)
    scaler = config_dict.get('scaler', MinMaxScaler())
    validation_set = config_dict.get('use_validation_set', True)
    use_past_covariates = config_dict.get('use_past_covariates', True)
    use_future_covariates = config_dict.get('use_future_covariates', True)
    

    logger.debug(
        "Function split_scale_datasets preparing datasets for %s window(s), forecast base date=%s, "
             "prediction_length=%s, output_chunk_shift=%s",
             windows, forecast_base_date, prediction_length, output_chunk_shift
    )
    
    ###* 1. Apply Static covariates transformer
    ts = ts_main_series
    if np.mean([s.has_static_covariates for s in ts_main_series]) == 1:
        ts_static_transformer = StaticCovariatesTransformer()
        ts = ts_static_transformer.fit_transform(ts_main_series)
    
    # Check if past/future covariates exists and have static covariates to be transformed
    past_covariates = ts_past_covariates
    if use_past_covariates and np.mean([s.has_static_covariates for s in ts_past_covariates]) == 1:
        past_cov_static_transformer = StaticCovariatesTransformer()
        past_covariates = past_cov_static_transformer.fit_transform(ts_past_covariates)
    
    future_covariates = ts_future_covariates
    if use_future_covariates and np.mean([s.has_static_covariates for s in ts_future_covariates]) == 1:
        future_cov_static_transformer = StaticCovariatesTransformer()
        future_covariates = future_cov_static_transformer.fit_transform(ts_future_covariates)

    ###* 2. Loop over periods to create different windows with train, validation and test sets
    datasets = {}
    for period in range(windows):
        ##* 2.1 Define dates
        current_base_date = pd.Timestamp(forecast_base_date) + pd.DateOffset(months=period)
        validation_end_date =  current_base_date - pd.DateOffset(months=1)
        train_set_end_date = (current_base_date - pd.DateOffset(months=prediction_length+1) 
                              if validation_set else validation_end_date)
        forecast_start_date = current_base_date + pd.DateOffset(months=output_chunk_shift)
        
        min_end_time = min(series.end_time() for series in ts)
        forecast_end_date = min(
            forecast_start_date + pd.DateOffset(months=prediction_length-1),
            min_end_time
        )
                             
        ##* save dates
        datasets[period] = {
            'base_date': current_base_date,
            'train_end_date': train_set_end_date,
            'forecast_period': (forecast_start_date, forecast_end_date),
        } 
        if validation_set:
            datasets[period].update({'validation_end_date': validation_end_date})
        
        logger.info(
            "Processing window %s - Base date: %s - Train end date: %s - Validation end date: %s - Forecast period: %s - %s",
            period, 
            datasets[period]['base_date'].strftime('%Y-%m-%d'),
            datasets[period]['train_end_date'].strftime('%Y-%m-%d'),
            datasets[period]['validation_end_date'].strftime('%Y-%m-%d') if validation_set else None,
            datasets[period]['forecast_period'][0].strftime('%Y-%m-%d'),
            datasets[period]['forecast_period'][1].strftime('%Y-%m-%d'))

        
        ##* 2.2 Split and scale datasets
        main_sets = _split_and_scale_single_set(
            ts, 
            train_set_end_date, 
            validation_end_date, 
            forecast_start_date, 
            forecast_end_date,
            scaler, 
            validation_set,
            series_name='ts',
        )
        datasets[period].update(main_sets)
        
        if use_past_covariates:
            past_cov_sets  = _split_and_scale_single_set(
                past_covariates, 
                train_set_end_date, 
                validation_end_date, 
                forecast_start_date, 
                forecast_end_date,
                scaler, 
                validation_set,
                series_name='past_cov',
            )
            datasets[period].update(past_cov_sets)
            
        if use_future_covariates:
            future_cov_sets = _split_and_scale_single_set(
                future_covariates, 
                train_set_end_date, 
                validation_end_date, 
                forecast_start_date, 
                forecast_end_date,
                scaler, 
                validation_set,
                series_name='future_cov',
            )
            datasets[period].update(future_cov_sets)
    
            
    return datasets


def _split_and_scale_single_set(
    ts_list: list[TimeSeries],
    train_set_end_date: pd.Timestamp,
    validation_end_date: pd.Timestamp,
    forecast_start_date: pd.Timestamp,
    forecast_end_date: pd.Timestamp,
    scaler: Scaler | None,
    validation_set: bool,
    series_name: str
) -> dict[str, Any]:
    """Helper function called by split_scale_datasets to split and scale a single set of TimeSeries objects
    
    Splits a list of TimeSeries into train, validation, and test sets based
    on provided dates, and optionally applies scaling to each split.
    
    Args:
        ts_list (list[TimeSeries]): List of TimeSeries objects to split
        train_set_end_date (pd.Timestamp): End date for training data (inclusive)
        validation_end_date (pd.Timestamp): End date for validation data (inclusive)
        forecast_start_date (pd.Timestamp): Start date for test/forecast period (inclusive)
        forecast_end_date (pd.Timestamp): End date for test/forecast period (inclusive)
        scaler: A scikit-learn-like scaler object with `fit_transform`, `transform` 
                and `inverse_transform` methods. If None, no scaling is applied.
        validation_set (bool): Whether to create validation splits
        series_name (str): Prefix for keys in output dictionary
  
    Returns:
        dict[str, Any]: Dictionary containing:
            {series_name}_train: Training data
            {series_name}_test: Test data 
            {series_name}_validation: Validation data if validation_set is True
            {series_name}_train_scaled: Training data scaled if scaler is provided
            {series_name}_validation_scaled: Validation data scaled if validation_set is True and scaler is provided
            {series_name}_test_scaled: Test data scaled if scaler is provided
            {series_name}_scaler: Fitted scaler object if scaling is applied
        
    """
    ##* 1. Split the datasets
    ts_train = [series.split_after(train_set_end_date)[0] for series in ts_list]
    
    ts_validation = ([series.split_after(validation_end_date)[0] for series in ts_list] 
                        if validation_set else None)
    if series_name != 'future_cov':
        ts_test = [series.slice(forecast_start_date, forecast_end_date) for series in ts_list]
    else:
        ts_test = [series.split_after(forecast_end_date)[0] for series in ts_list]
    
        
    ##* 2. Store raw splits
    sets = {
        f'{series_name}_train': ts_train,
        f'{series_name}_test': ts_test,
    }
    if validation_set:
        sets.update({f'{series_name}_validation': ts_validation})
    
    ##* 3. Conditionally apply scaling
    if scaler is not None:
        ts_scaler = Scaler(scaler=scaler, global_fit=False)
        ts_train_scaled = ts_scaler.fit_transform(ts_train)
        ts_validation_scaled = (ts_scaler.transform(ts_validation) if validation_set else None)
        ts_test_scaled = ts_scaler.transform(ts_test)
        
        sets.update({
            f'{series_name}_train_scaled': ts_train_scaled,
            f'{series_name}_test_scaled': ts_test_scaled,
            f'{series_name}_scaler': ts_scaler,
        })
        if validation_set:
            sets.update({f'{series_name}_validation_scaled': ts_validation_scaled})
        
    
    return sets




#%% #>------------- Function: Split multivariate series into univariate --------------------------


def split_multivariate_ts(
    multivariate_list: list[TimeSeries]
) -> list[TimeSeries]:
    """Splits a list containing a multivariate TimeSeries into a list of individual univariate components.
    
    Extracts each component from multivariate TimeSeries objects and returns
    them as separate univariate TimeSeries objects in a flat list.
    
    If is passed multiple multivariate TimeSeries objects inside the list, it will extract
    each component an return them as a single flat list of univariate TimeSeries objects.
    
    Args:
        multivariate_list (list[TimeSeries]): List of multivariate TimeSeries objects
        
    Returns:
        list[TimeSeries]: Flattened list of univariate TimeSeries components
    """
    
    #* Initialize list to store univariate series
    univariate_series = []
    
    #* Loop through each multivariate and each series component and extract as univariate series
    for mv_ts in multivariate_list:
        components = mv_ts.n_components
        for i in range(components):
            univariate_series.append(mv_ts.univariate_component(i))
    
    return univariate_series







