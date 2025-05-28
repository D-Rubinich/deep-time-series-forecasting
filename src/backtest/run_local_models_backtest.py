'''
Univariate Local Time Series Models backtest runner script.

Orchestrates rolling backtesting (across multiple windows) for multiple univariate local 
models using Darts library and parallelization with joblib. This script primarily handles 
non-deep learning  models that train on each time series independently, including:
- AutoARIMA: Automated ARIMA model with seasonal components
- AutoETS: Exponential smoothing with automated parameter selection
- Prophet: Facebook's Prophet forecasting model

Script Structure:
1. Imports and logger setup
2. Build `config_dict` (general configuration setups for backtest)
3. Build `models_config` (dictionary of models with their parameters)
4. Call local_m_pipe.main(...) for execution via local_models_pipeline.py

Configuration Overview:
- config_dict: controls backtest configurations, forecast parameters, data configurations, etc.
- models_config: dictionary of models to test, where each model has:
    • model_class: The Darts model class to instantiate
    • params: Dictionary of model-specific parameters

Usage:
    python src/backtest/run_local_models_backtest.py

References:
- AutoARIMA: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_arima.html
- AutoETS: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_ets.html
- Prophet: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.prophet_model.html

Notes: These local models have specific forecast restrictions:
    • They handle univariate data and train a separate model for each time series
    • The models don't directly support output_chunk_shift (forecast offset)
    • To handle multi-step forecasts with offset, the pipeline will generate predictions 
        for output_chunk_shift + prediction_length steps, then discard the initial 
        output_chunk_shift predictions to align with the desired forecast horizon
'''



#%% #>-------------------- Imports, logger setup and settings --------------------

#* standard library
import logging
import warnings

#* third-party libraries
import pandas as pd
from darts.models import StatsForecastAutoARIMA, StatsForecastAutoETS, Prophet
from optuna._experimental import ExperimentalWarning

#* Local imports
from src.utilities import logger_setup as log_setup
from src.pipelines import local_models_pipeline as local_m_pipe
from config import DATA_DIR

#* Logger setup
backtest_timestamp = pd.Timestamp.now(tz='UTC').strftime("%Y%m%d_%H%M")
log_setup.configure_logging (
    name= __name__, 
    run_type='backtest_logs',
    console_log_level=logging.INFO,     
    model_name = "local_models",
    timestamp=backtest_timestamp,
)

#* Warnings settings
warnings.filterwarnings('ignore', category=ExperimentalWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


#%%#>-------------------- Configs  ---------------------------

#####> 0. Backtest and general configurations
#* Configuration dictionary
config_dict = {
    #* General parameters
    'model_name': 'local_models',
    'output_chunk_length': 3,       
    'output_chunk_shift': 1,  
    'prediction_length': 3,        
    'backtest_timestamp': backtest_timestamp,   
    
    #* Data parameters
    'data_path':f"{DATA_DIR}/processed/synthetic_data_sample.parquet",
    'scaler': None,  
    'use_static_covariates': False,
    'use_past_covariates': False,
    'use_future_covariates': False,
    'use_validation_set': False,
    
    #* Backtest parameters
    'backtest_start_date': None,            # None: automatic, else: specify date
    'backtest_windows': 6,
    'backtest_results_save_path': None,     # None: uses default path (data/backtest_results)
}
    
#####> 1. Models Configurations 
models_config = {
    'AutoARIMA': {
        'model_class': StatsForecastAutoARIMA,   
        'params': {
            'season_length':12,
            'start_p': 1,                   # Start search with a minimum autoregressive component
            'start_q': 1,                   # Start search with a minimum moving average component
            'stepwise':True,
            'max_order': 20,                # Maximum total of AR + MA terms
            'seasonal': True,               # Enable seasonal components
        }  
    },
    'ExponentialSmoothing': {
        'model_class': StatsForecastAutoETS,
        'params': {
            'season_length': 12,                 # Matches monthly seasonality
        }                  
    },
    'Prophet': {
        'model_class': Prophet,
        'params': {
            'seasonality_mode': 'additive',     # Additive seasonality; can switch to 'multiplicative' for proportional seasonality
            'yearly_seasonality': True,         # Enables yearly seasonality (matches season_length)
        }
    }
}


#%%#>-------------------- Execution  ---------------------------

if __name__ == "__main__":
    local_m_pipe.main(
        config_dict=config_dict,
        models_config=models_config,
        n_jobs=-1
    )
    


