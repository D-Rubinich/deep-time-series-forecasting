"""
This file contains utility functions to help prepare and analyze the results 
collected from the backtesting.

The functions are designed to load and process backtest results from multiple CSV files,
consolidating them into a single DataFrame for both window-based and series-based backtests
and expanding columns in list format into individual columns for each forecast period for
easier analysis.
"""	

#%% #>-------------------- Imports and module exports --------------------

#* standard library
from pathlib import Path
import ast
import re

#* third-party libraries
import pandas as pd



__all__ = [
    'load_all_backtest_results',
    'load_window_backtest_results',
    'load_series_backtest_results',
    'get_mape_summary',
]

#%% #>---------------- Functions - load and process backtest results  -------------------------


def load_all_backtest_results(
    backtest_path: Path = '/workspace/data/backtest_results',
    check: bool = False,
    cols_to_split: list = ['ape', 'actual_values', 'forecast_values']
)-> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and process both window and series-based backtest results.
    
    Args:
        backtest_path (Path, optional): Directory containing files for window and series results
            Defaults to '/workspace/data/backtest_results'.
        check (bool, optional): If True, display summary information about the loaded data.
            Defaults to False.
        cols_to_split (list, optional): List of columns to be expanded into individual columns.
            Defaults to ['ape', 'actual_values', 'forecast_values'].
        
    Returns:
        Tuple of DataFrames (window_results, series_results) containing processed backtest data
    """
    df_window = load_window_backtest_results(backtest_path, check)
    df_series = load_series_backtest_results(backtest_path, check, cols_to_split)
    
    return df_window, df_series


def load_window_backtest_results(
    backtest_path: Path = '/workspace/data/backtest_results',
    check: bool = False
) -> pd.DataFrame:
    """Load and process window-based backtest results from CSV files.

    Args:
        backtest_path (Path, optional): Directory containing files matching '*window_results.csv'. 
            Defaults to '/workspace/data/backtest_results'.
        check (bool, optional): If True, display summary information about the loaded data.
            Defaults to False.

    Returns:
        pd.DataFrame: Combined DataFrame containing processed window-based backtest results with 
            expanded WAPE columns for each forecast period
    """
    ###* Get files and load as unified dataframe
    if isinstance(backtest_path, str):
        backtest_path = Path(backtest_path)
    files  = backtest_path.glob('*window_results.csv')
    df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

    ###* Adjust columns that are in list formats to individual columns
    # ensure list type
    df['wape_per_period'] = df['wape_per_period'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x) # convert string representation of list to actual list
    
    # Expand wape_per_period into individual columns
    n_periods = df['wape_per_period'].str.len().max()
    for i in range(n_periods):
        df[f'wape_f{i+1}'] = (
            df['wape_per_period']
            .apply(lambda x: x[i] if i < len(x) else None)
            .astype(float)
        )
    df.drop(columns=['wape_per_period'], inplace=True)
        
    ###* Check results
    if check:
        print(f"\n>>> Window level results:")
        print(f"Models in backtest: {df['model'].unique().tolist()}")
        print(f"Base dates: {df['base_date'].unique().tolist()}")
        print(f"Forecast start: {df['forecast_start'].unique().tolist()}")
        print(f"Forecast end:   {df['forecast_end'].unique().tolist()}")
    
    return df


def load_series_backtest_results(
    backtest_path: Path = '/workspace/data/backtest_results',
    check: bool = False,
    cols_to_split: list = ['ape', 'actual_values', 'forecast_values']
) -> pd.DataFrame:
    """ Load and process series-based backtest results from CSV files.

    Args:
        backtest_path (Path, optional): Directory containing files matching '*window_results.csv'. 
            Defaults to '/workspace/data/backtest_results'.
        check (bool, optional): If True, display summary information about the loaded data.
            Defaults to False.
        cols_to_split (list, optional): List of columns to be expanded into individual columns.
            Defaults to ['ape', 'actual_values', 'forecast_values'].

    Returns:
        pd.DataFrame: Combined DataFrame with each of ['ape','actual_values','forecast_values']
            expanded into '_f1','_f2',… columns and originals dropped.
    """
    
    ###* Get files and load as unified dataframe
    series_files = backtest_path.glob('*series_results.csv')
    df = pd.concat([pd.read_csv(file) for file in series_files], ignore_index=True)

    ###* Adjust columns that are in list formats to individual columns
    # helper to parse strings like "[1, 2, 3]" or "1 2 3"
    cols_to_split = cols_to_split if isinstance(cols_to_split, list) else [cols_to_split]
    parse_list = lambda s: [float(n) for n in re.split(r'[\s,]+', s.strip('[]')) if n] # adjustment for columns using space as separators

    # ensure list type
    for col in cols_to_split:
        df[col] = df[col].apply(lambda x: parse_list(x) if isinstance(x, str) else x)
        
        n_periods = df[col].str.len().max() 

        for i in range(n_periods):
            df[f'{col}_f{i+1}'] = (df[col].apply(lambda x: x[i] if i < len(x) else None)).astype(float)

    df.drop(columns=cols_to_split, inplace=True)   

    ###* Check results
    if check:
        print(f"\n>>> Series level results:")
        print(f"Series names: {list(df['series_name'].unique())}")
        print(f"Models in backtest: {df['model'].unique().tolist()}")
        print(f"Forecast start: {df['forecast_start'].unique().tolist()}")
        print(f"Forecast end:   {df['forecast_end'].unique().tolist()}")
    
    return df



#%% #>---------------- Functions - Tables Summary  -------------------------

def get_wape_summary(
    df: pd.DataFrame,
    group_by_col: str = "model",
    wape_cols_prefix: str = "wape_f",
    avg_col: str = "avg_wape_across_periods",
    mode: str = "both",  # options: "avg", "per_period", "both"
) -> pd.DataFrame:
    """
    Summarize WAPE results by model.

    Args:
        df (pd.DataFrame): DataFrame with WAPE columns.
        group_by_col (str): Column to group by (default: "model").
        wape_cols_prefix (str): Prefix for per-period WAPE columns (default: "wape_f").
        avg_col (str): Column name for average WAPE (default: "avg_wape_across_periods").
        mode (str): "avg", "per_period", or "both".

    Returns:
        pd.DataFrame: Summary table.
    """
    agg_dict = {}
    # Find all per-period WAPE columns
    wape_cols = [col for col in df.columns if col.startswith(wape_cols_prefix)]
    wape_cols = sorted(wape_cols, key=lambda x: int(re.findall(r'\d+', x)[0]))  # sort by period

    if mode in ("avg", "both") and avg_col in df.columns:
        agg_dict["avg_wape"] = (avg_col, "mean")
        agg_dict["std_avg_wape"] = (avg_col, "std")

    if mode in ("per_period", "both") and wape_cols:
        for col in wape_cols:
            agg_dict[f"{col}"] = (col, "mean")
        for col in wape_cols:
            agg_dict[f"std_{col}"] = (col, "std")

    summary_df = (
        df.groupby(group_by_col, as_index=False)
          .agg(**agg_dict)
          .round(4)
    )
    # Sort by avg_wape or first period
    if "avg_wape" in summary_df.columns:
        summary_df = summary_df.sort_values("avg_wape")
    elif wape_cols:
        summary_df = summary_df.sort_values(wape_cols[0])
    return summary_df.reset_index(drop=True)



def get_mape_summary(
    df: pd.DataFrame,
    group_by_col: str = "model",
    mape_col: str = "mape",
) -> pd.DataFrame:
    """
    Summarize MAPE results by model in series-level backtest results.

    Args:
        df (pd.DataFrame): DataFrame with MAPE column.
        group_by_col (str): Column to group by (default: "model").
        mape_col (str): Name of the MAPE column (default: "mape").

    Returns:
        pd.DataFrame: Summary table with mean, std, min, percentiles, and max MAPE per group.
    """
    summary_df = (
        df.groupby(group_by_col)[mape_col]
        .agg(
            mean_MAPE   = "mean",
            std_MAPE    = "std",
            min_MAPE    = "min",
            p25_MAPE    = lambda x: x.quantile(0.25),
            median_MAPE = "median",
            p75_MAPE    = lambda x: x.quantile(0.75),
            max_MAPE    = "max",
        )
        .round(4)
        .reset_index()
        .sort_values("mean_MAPE")
        .reset_index(drop=True)
    )
    return summary_df
    



