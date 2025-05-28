'''
TiDE (Time-series Dense Encoder) backtest and hyperparameter optimization runner.

Orchestrates Optuna hyperparameter search and rolling backtests (multiple windows)
for the model using the Darts library. Supports univariate and multivariate time-series formats.

Script Structure:
1. Imports and logger setup
2. Build `config_dict` (general, data, backtest, Optuna, and logging settings)
3. Build `model_config` (model hyperparameters, architecture, trainer, callbacks)
4. Build `param_search_space` (Optuna search space definitions)
5. Call `dl_pipe.main(...)` for execution via `src/pipelines/dl_full_pipeline.py`

Configuration Overview:
- config_dict: controls backtest and optuna configurations, forecast parameters, data configurations, dashboards, etc.
- model_config: defines model hyperparameters, training settings, callbacks, lr scheduler.
- param_search_space: Defines the hyperparameter search space for Optuna optimization:
    • Any parameter omitted uses its `model_config` default.    
    • Supported spec types:
        - Categorical:  {'type': 'categorical', 'values': [val1, val2, ...]}
        - Integer:      {'type': 'int', 'low': min_val, 'high': max_val}
        - Float:        {'type': 'float', 'low': min_val, 'high': max_val, 'log': bool}
            The 'log' flag enables logarithmic sampling (useful for variables with large ranges)
    • First defines univariate search space; if `data_format=='multivariate'`,
    updates with multivariate-specific specs (to account for different data characteristics and
    model behavior in each format)   
    • To customize: add/remove entries or adjust ranges as needed.

Usage:
    python src/backtest/run_tide_backtest.py

References:
    - Model Paper: https://arxiv.org/pdf/2304.08424
    - Documentation: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tide_model.html
'''

#%% #>-------------------- Imports, logger setup and settings --------------------

#* standard library
import logging
import warnings

#* third-party libraries
import pandas as pd
import torch
from darts.models import TiDEModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.preprocessing import MinMaxScaler
from optuna._experimental import ExperimentalWarning

#* Local imports
from src.utilities import logger_setup as log_setup
from src.pipelines import dl_full_pipeline as dl_pipe
from config import DATA_DIR

#* Logger setup
backtest_timestamp = pd.Timestamp.now(tz='UTC').strftime("%Y%m%d_%H%M")
log_setup.configure_logging (
    name=__name__, 
    run_type='backtest_logs',
    console_log_level=logging.INFO,     
    model_name="TiDE",
    timestamp=backtest_timestamp,   
)

#* Warnings settings
warnings.filterwarnings('ignore', category=ExperimentalWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#* Torch settings
torch.set_float32_matmul_precision('medium')

#%%#>-------------------- Configs  ---------------------------

#####> 0. Backtest, optuna and general configurations
#* Configuration dictionary
config_dict = {
    #* General parameters
    'model_class': TiDEModel,
    'model_name': 'TiDE',
    'output_chunk_length': 3,           # how many steps do train the model to predict at once
    'prediction_length': 3,             # if bigger than output_chunk_length, uses recursive forecasting
    'output_chunk_shift': 1,            # shift period between end training date and first prediction date
    'backtest_timestamp': backtest_timestamp,     
    
    #* Data parameters
    'data_format': 'univariate',       # Defines the data structure: 'univariate' or 'multivariate'
    'data_path':f"{DATA_DIR}/processed/synthetic_data_sample.parquet",
    'covariates_path': f"{DATA_DIR}/processed/synthetic_covariates_sample.parquet", # only used if data_format is 'multivariate'
    'scaler': MinMaxScaler(),
    'use_static_covariates': True,
    'use_past_covariates': True,
    'use_future_covariates': True,
    'use_validation_set': True,
    
    #* Backtest parameters
    'run_backtest': True,
    'backtest_start_date': None,            # None: automatic, else: specify date
    'backtest_windows': 6,
    'backtest_retraining_frequency': 3,     # Int:retraining frequency, True: all windows, False:only first window
    'study_name': None,                     # Name of study to get optimized parameters for backtest. If None, gets the best trial of all model's studies
    'trial_rank': 0,                        # Decides which trial to get from study (0: best trial, 1: second best trial, etc.)
    'backtest_results_save_path': None,     # None: uses default path (data/backtest_results)
    'use_modelcheckpoint_callback':False,   # weather to use the model checkpoint callback
    
    #* Optuna parameters 
    'run_optuna': True,    
    'optuna_storage_path': None,            # Path do optuna database. None uses data/optuna/optuna_study.db"
    'optuna_study_name': None,              # Name of the study to be created in this run
    'optuna_start_date': None,              # None: automatic. Avoid setting manually to prevent data leakage (optuna windows overlapping backtest windows)
    'optuna_windows': 3,
    'optuna_retraining_frequency': False,   # Int:retraining frequency, True: all windows, False:only first window
    'optuna_n_trials': 5,                   # Use either N_TRIALS OR TIMEOUT 
    #'optuna_timeout':60*60*3,                  # Use either N_TRIALS OR TIMEOUT. TIMEOUT in Seconds
    'optuna_direction': "minimize",
    'sampler_class': TPESampler,
    'pruner_class':MedianPruner,
    'pruner_kwargs': None,                  # None: uses default values
    'sampler_kwargs': None,                 # None: uses default values
    'open_optuna_dashboard': True,
    'close_optuna_dashboard': False,
    
    #* Tensorboard settings
    'tensorboard_path_optuna':None,         
    'tensorboard_path_backtest':None,       # None uses default path (logs/tensorboard_logs/backtest)
    'tensorboard_id': None,
    'open_tensorboard': True,
    'close_tensorboard': False,
    
    #* Other parameters
    'random_state': 1000,
}
#* Update model_name, tensorboard_id and optuna_study_name
config_dict['model_name'] = f"{config_dict['model_name']}_{config_dict['data_format']}"
session_name = f"{config_dict['model_name']}_utc_{config_dict['backtest_timestamp']}"
config_dict['tensorboard_id'] = session_name
config_dict['optuna_study_name'] = session_name

    
#####> 1. Model Configuration and Parameter Search Space
##* Default model configuration - all parameters have fixed values
model_config = {
    #* Base parameters
    'base_params' : {
        'output_chunk_length': config_dict['output_chunk_length'],
        'output_chunk_shift': config_dict['output_chunk_shift'],
        'input_chunk_length': 12,  
        'use_static_covariates': config_dict['use_static_covariates'],
        'random_state': config_dict['random_state'],
        'force_reset': True,       
    },

    #* Model Architecture parameters
    'model_architecture_params' : {
        "hidden_size": 128,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "decoder_output_dim": 16,
        "temporal_decoder_hidden": 16,
        "dropout": 0.1,
        "use_layer_norm": True,                     
        "use_reversible_instance_norm": True,       # Reduces distribution shift impact in forecasting accuracy
        "temporal_width_future": 4,                 # Fixed as per the paper
        "temporal_width_past": 4,                   # Fixed as per the paper
    },

    #* PyTorch Lightning Trainer parameters
    'trainer_params' : {
        "optimizer_cls": AdamW,
        "n_epochs": 300,
        "batch_size": 256,
        "gradient_clip_algorithm": "norm",          # norm-based gradient clipping
        "gradient_clip_val": 10,
        "weight_decay": 5e-3,
        "accelerator": "gpu",
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "enable_checkpointing": False,              
        "log_every_n_steps": 20,
    },
    'early_stopping_kwargs' : {
            'patience':15,  
            'min_delta':5e-4,
    },
    'model_checkpoint_kwargs' : {
            'save_top_k': 1,
            'every_n_epochs': 15,
            'save_last':False, 
    },

    ##* Learning Rate Scheduler parameters
    'lr_scheduler_params' : {
        "lr_scheduler_cls": OneCycleLR,
        "lr_scheduler_kwargs": {
            "interval": "step",
            "frequency": 1,
            'total_steps': 12601,   # Updated automatically by calculate_rate_scheduler_total_steps() inside pipeline
            "max_lr": 1e-4,
            "pct_start": 0.20,
            "div_factor": 40,
            "final_div_factor": 75,
            "three_phase": True, 
        },
    }
}


##* 1. Univariate format - Optuna parameters Search space
# If a parameter is here, it will override the fixed value in model_config
param_search_space = {
    #* Base parameters
    'input_chunk_length': {'type': 'categorical', 'values': [3, 6, 12, 18, 24]},
    
    #* Model Architecture parameters
    'hidden_size': {'type': 'categorical', 'values': [128, 256, 512]},
    'num_encoder_layers': {'type': 'int', 'low': 1, 'high': 3},
    'num_decoder_layers': {'type': 'int', 'low': 1, 'high': 3},
    'decoder_output_dim': {'type': 'categorical', 'values': [16, 32, 64]},
    'temporal_decoder_hidden': {'type': 'categorical', 'values': [8, 16, 32, 64]},
    'dropout': {'type': 'float', 'low': 0.05, 'high': 0.6},
    
    #* Trainer parameters
    'batch_size': {'type': 'categorical', 'values': [64, 128, 256, 512]},
    'gradient_clip_val': {'type': 'float', 'low': 0.3, 'high': 100.0, 'log': True},
    'weight_decay': {'type': 'float', 'low': 5e-5, 'high': 1e-1, 'log': True},
    
    #* Learning Rate Scheduler parameters
    'max_lr': { 'type': 'float', 'low': 1e-6, 'high': 1e-2, 'log': True},
    'pct_start': {'type': 'categorical', 'values': [0.20, 0.25, 0.30, 0.35]},
    'div_factor': {'type': 'categorical', 'values': [25, 30, 35, 40, 45, 50]},
    'final_div_factor': {'type': 'categorical', 'values': [75, 100, 250, 500]},
    'three_phase': {'type': 'categorical', 'values': [True, False]}
}

##* 2. Updates in Models Config and Parameters Search space for Multivariate
if config_dict['data_format'] == 'multivariate':
    
    ### Search Space
    param_search_space.update({
        # Base parameters
        'input_chunk_length': {'type': 'categorical', 'values': [3, 6, 9, 12]},
        # Model Architecture parameters
        'dropout': {'type': 'float', 'low': 0.05, 'high': 0.7},
        
        # Trainer parameters
        'batch_size': {'type': 'categorical', 'values': [4, 8, 16]},  #reduced due to less samples in the multivariate format
        'max_lr': { 'type': 'float', 'low': 5e-7, 'high': 1e-3, 'log': True},
        
        # Learning Rate Scheduler parameters
    })
    
    ### Model Config
    model_config['trainer_params'].update({
        "log_every_n_steps": 10,     #reduced due to less samples (and steps) in the multivariate format
    })
    




#%%#>-------------------- Execution  ---------------------------

if __name__ == "__main__":
    dl_pipe.main(
        config_dict=config_dict, 
        model_config=model_config, 
        param_search_space=param_search_space,
    )
    
