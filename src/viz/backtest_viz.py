"""
This file contains functions to generate visualizations for backtest results analysis.

"""	

#%% #>-------------------- Imports and module exports --------------------

#* standard library
from pathlib import Path


#* third-party libraries
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Figure
import pandas as pd
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output

__all__ = [
    'boxplot_wape_per_model',
    'lineplot_evolution_by_group',
    'barplot_wape_by_forecast_horizon',
    'violinplot_error_distribution',
    'scatterplot_actual_vs_forecast',
    'create_interactive_timeseries_plot',
]

#%% #>-------------------- Functions  -----------------------------

def boxplot_wape_per_model(
    df: pd.DataFrame,
    value_col_prefix: str = 'wape_f',
    value_name_legend: str = 'wape',
    group_col: str = "model",  
    color_palette: list[str] | None = None,
    graph_title: str = "WAPE per Model (all forecasts horizons)",
    xaxis_title: str = "Models",
    yaxis_title: str = "WAPE",
    y_range: list[float] = [0, 1],
    marker_size: int = 6,
    marker_opacity: float = 0.4,
    fig_width: int = 1100,
    fig_height: int = 600,
    show_legend: bool = False,
) -> Figure:
    """Creates a plotly boxplot visualizing the distribution of 'value_col' across different 'group_col'
    
    This function transforms wide-format data into long format and creates an interactive
    boxplot  showing the distribution of 'value_col' metrics (WAPE by default) across different 
    'group_col' (model by default) or other stratification variables. The plot includes individual 
    data points for detailed examination.
       
    Args:
        df (pd.DataFrame): DataFrame containing the forecast results with columns matching the value_col_prefix.
        value_col_prefix (str): Prefix of columns containing values to plot (default: 'wape_f').
        value_name_legend (str): Name to use for showing the value in the plot (default: 'wape').
        group_col (str): Column to stratify/group by, typically 'model' (default: 'model').
        color_palette (dict[str, str] | None): Optional list of color strings/codes. If provided, the 
            first N colors will be assigned in order to each unique value in `group_col`. If
            None, uses Plotly's `px.colors.qualitative.Dark24` palette.
        graph_title (str): Title for the plot (default: "WAPE per Model (all forecasts horizons)").
        xaxis_title (str): Title for the x-axis (default: "Models").
        yaxis_title (str): Title for the y-axis (default: "WAPE").
        y_range (list[float]): Range for the y-axis (default: [0.2, 1]).
        marker_size (int): Size of individual data points (default: 6).
        marker_opacity (float): Opacity of individual data points from 0 to 1 (default: 0.4).
        fig_width (int): Width of the figure in pixels (default: 1100).
        fig_height (int): Height of the figure in pixels (default: 600).
        show_legend (bool): Whether to display the color legend (default: False).

    Returns:
        Figure: A Plotly `Figure` object containing the styled box plot.
    
    Raises:
        ValueError: If no columns with the specified prefix are found in the DataFrame
    """
    #* Select and validate input of value columns to include
    value_cols = [col for col in df.columns if col.startswith(value_col_prefix)]
    if not value_cols:
        raise ValueError(f"No columns starting with prefix '{value_col_prefix}' found")
    
    #* Determine Color Mapping
    color_map_unique_values = df[group_col].unique().tolist()
    if color_palette is None:
        colors = px.colors.qualitative.Dark24[:len(color_map_unique_values)]
    else:
        colors = color_palette[:len(color_map_unique_values)]
    
    color_map = dict(zip(color_map_unique_values, colors))
    
    #* Melt the DataFrame to long format
    df_long = (
        df.melt(
            id_vars=group_col,
            value_vars=value_cols,
            var_name="step",
            value_name=value_name_legend
        )
    )
    
    #* Create the boxplot
    fig = px.box(
        df_long,
        x=group_col, 
        y=value_name_legend,
        color=group_col,
        color_discrete_map=color_map,
        points="all",                                   # show underlying points
        title=graph_title,
    )
    
    # Update trace
    fig.update_traces(
        boxpoints="all",
        jitter=0.25,
        pointpos=0,
        marker=dict(opacity=marker_opacity, size=marker_size)
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=50, r=50, t=70, b=50),
        width=fig_width,
        height=fig_height,
        plot_bgcolor='white',  
        paper_bgcolor='white', 
        yaxis_title=yaxis_title,
        yaxis=dict(
            range=y_range,
            tickformat='.0%',
            tickfont=dict(size=12),
            title_font=dict(size=14),
            gridcolor='lightgrey',
            gridwidth=1,
            griddash='dot',
            showline=True,  # Add axis line
            linecolor='grey', 
        ),
        xaxis_title=xaxis_title,
        xaxis=dict(
            tickfont=dict(size=12),
            title_font=dict(size=14),
            showline=True,  # Add axis line
            linecolor='grey', 
        ),
        title_font=dict(size=20),
        showlegend=show_legend,
    )
    
    
    return fig

def lineplot_evolution_by_group(
    df: pd.DataFrame,
    x_col: str = 'window',
    y_col: str = 'avg_wape_across_periods',
    group_col: str = "model",  
    color_palette: list[str] | None = None,
    markers: bool = True,
    graph_title: str = 'Evolution of Average WAPE by Model',
    xaxis_title: str = 'Backtest Window',
    yaxis_title: str = 'Average WAPE',
    y_range: list[float] = [0.2, 0.8],
    fig_width: int = 1100,
    fig_height: int = 600,
    show_legend: bool = True,
) -> Figure:
    """Creates a plotly line plot visualizing the evolution of metrics across different groups
    
    This function generates an interactive line plot that shows how metrics (like WAPE) 
    change across a sequence of values (like backtest windows), with different lines 
    representing different groups (like models). Each line is color-coded according to 
    its group, and optional markers highlight individual data points.
    
    Args:
        df (pd.DataFrame): DataFrame containing the required columns for plotting.
        x_col (str): DF Column name to use for x-axis values, typically a sequence identifier (default: 'window').
        y_col (str): DF Column name containing the metric to plot on y-axis (default: 'avg_wape_across_periods').
        group_col (str): Column name to use for grouping and color differentiation (default: 'model').
        color_palette (dict[str, str] | None): Optional list of color strings/codes. If provided, the 
            first N colors will be assigned in order to each unique value in `group_col`. If
            None, uses Plotly's `px.colors.qualitative.Dark24` palette.
        markers (bool): Whether to display markers at each data point (default: True).
        graph_title (str): Title for the plot (default: 'Evolution of Average WAPE by Model').
        xaxis_title (str): Title for the x-axis (default: 'Backtest Window').
        yaxis_title (str): Title for the y-axis (default: 'Average WAPE').
        y_range (list[float]): Range for the y-axis (default: [0.2, 0.8]).
        fig_width (int): Width of the figure in pixels (default: 1100).
        fig_height (int): Height of the figure in pixels (default: 600).
        show_legend (bool): Whether to display the color legend (default: True).
    
    Returns:
        Figure: A Plotly `Figure` object containing the styled line plot.
    
    Raises:
        ValueError: If any required columns (x_col, y_col, group_col) are missing from the DataFrame.
    """
    
    #* Validate inputs
    missing = [c for c in (x_col, y_col, group_col) if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    
    #* Determine Color Mapping
    color_map_unique_values = df[group_col].unique().tolist()
    if color_palette is None:
        colors = px.colors.qualitative.Dark24[:len(color_map_unique_values)]
    else:
        colors = color_palette[:len(color_map_unique_values)]
    
    color_map = dict(zip(color_map_unique_values, colors))
    
    #* Create the interactive line plot
    fig = px.line(
        df, 
        x=x_col,
        y=y_col,
        color=group_col,
        color_discrete_map=color_map,
        markers=markers,
        title=graph_title,
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=50, r=50, t=70, b=50),
        width=fig_width,
        height=fig_height,
        plot_bgcolor='white',  
        paper_bgcolor='white', 
        legend_title=None,
        legend=dict(
            orientation="v",  # horizontal legend
            yanchor="middle",
            y=0.5,  # position the legend below the plot
            xanchor="left",
            x=1.01,
            font=dict(size=14),
        ),
        yaxis_title=yaxis_title,
        yaxis=dict(
            range=y_range,
            tickformat='.0%',
            tickfont=dict(size=12),
            title_font=dict(size=14),
            gridcolor='lightgrey',
            gridwidth=1,
            griddash='dot',
            showline=True,  # Add axis line
            linecolor='grey', 
        ),
        xaxis_title=xaxis_title,
        xaxis=dict(
            tickfont=dict(size=12),
            title_font=dict(size=14),
            showline=True,  # Add axis line
            linecolor='grey', 
        ),
        title_font=dict(size=20),
        showlegend=show_legend,
    )

    return fig

def barplot_wape_by_forecast_horizon(
    df: pd.DataFrame,
    value_col_prefix: str = 'wape_f',
    value_name_legend: str = 'wape',    
    group_col: str = "model",
    horizon_order: list[str] = ["F1", "F2", "F3"],
    graph_title: str = "Average WAPE per Model and Forecast Horizon",
    xaxis_title: str = None,
    yaxis_title: str = "Average WAPE",
    y_range: list[float] = [0.1, 0.8],
    text_format: str = ".0%",
    fig_width: int = 1000,
    fig_height: int = 400,
) -> Figure:
    
    #* Select value columns to include
    value_cols = [col for col in df.columns if col.startswith(value_col_prefix)]
    if not value_cols:
        raise ValueError(f"No columns with prefix '{value_col_prefix}' found in DataFrame")
    
    #* Melt the DataFrame to long format
    df_long = (
        df.melt(
            id_vars=group_col,
            value_vars=value_cols,
            var_name="step",
            value_name=value_name_legend
        )
    )
    
    # Extract horizon number and convert to categorical
    df_long["step"] = "F" + df_long["step"].str.extract(r"(\d+)")[0]
    df_long["step"] = pd.Categorical(df_long["step"], categories=horizon_order)
    
    #* Calculate mean values per model and forecast horizon
    mean_wape = df_long.groupby([group_col, "step"], as_index=False)[value_name_legend].mean()
    
    #* Create the grouped bar chart
    fig = px.bar(
        mean_wape,
        x=group_col,
        y=value_name_legend,
        color="step",
        barmode="group",
        title=graph_title,
        text_auto=text_format,
        category_orders={"step": horizon_order},
    )
    
    fig.update_traces(
        textposition="outside",  
        textfont=dict(size=14),
        cliponaxis=False,
    )
    
    fig.update_layout(
        margin=dict(l=50, r=50, t=70, b=50),
        width=fig_width,
        height=fig_height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis_title=yaxis_title,
        yaxis=dict(
            range=y_range,
            tickformat=".0%",
            tickfont=dict(size=12),
            title_font=dict(size=14),
            gridcolor="lightgrey",
            gridwidth=1,
            griddash="dot",
            showline=True,
            linecolor="grey",
        ),
        xaxis_title=xaxis_title,
        xaxis=dict(
            tickfont=dict(size=12),
            title_font=dict(size=14),
            showline=True,
            linecolor="grey",
        ),
        title_font=dict(size=18),
    )
    
    return fig
    
def violinplot_error_distribution(
    df: pd.DataFrame,
    value_col_prefix: str = 'ape_f',
    value_name_legend: str = 'APE',
    group_col: str = "model",
    color_palette: list[str] | None = None,
    series_col: str = "series_name",
    graph_title: str = "Series Absolute Percentage Error (APE) Distribution per Model",
    xaxis_title: str = "Models",
    yaxis_title: str = "APE",
    y_range: list[float] = [0, 2],
    show_points: bool = True,
    show_box: bool = True,
    show_meanline: bool = True,
    point_size: int = 3,
    point_opacity: float = 0.7,
    jitter: float = 0.05,
    point_position: float = -0.1,
    fig_width: int = 1100,
    fig_height: int = 600,
    show_legend: bool = True,
) -> Figure:
    """
    Creates a violin plot showing the distribution of error metrics (e.g., APE) across different models.

    This function transforms wide-format error data into long format and generates an interactive
    Plotly violin plot. Each violin represents the distribution of the error metric for a given model,
    optionally displaying underlying data points, boxplots, and mean lines for enhanced interpretability.

    Args:
        df (pd.DataFrame): DataFrame containing the error metrics and grouping columns.
        value_col_prefix (str): Prefix of columns containing error values to plot (default: 'ape_f').
        value_name_legend (str): Name to use for the value in the plot legend and y-axis (default: 'APE').
        group_col (str): Column to group violins by, typically 'model' (default: 'model').
        color_palette (list[str] | None): Optional list of color strings/codes. If provided, the 
            first N colors will be assigned in order to each unique value in `group_col`. If
            None, uses Plotly's `px.colors.qualitative.Dark24` palette.
        series_col (str): Column indicating the series name for each data point (default: 'series_name').
        graph_title (str): Title for the plot (default: "Series Absolute Percentage Error (APE) Distribution per Model").
        xaxis_title (str): Title for the x-axis (default: "Models").
        yaxis_title (str): Title for the y-axis (default: "APE").
        y_range (list[float]): Range for the y-axis (default: [0, 2]).
        show_points (bool): Whether to display individual data points within each violin (default: True).
        show_box (bool): Whether to display a boxplot inside each violin (default: True).
        show_meanline (bool): Whether to display a mean line inside each violin (default: True).
        point_size (int): Size of individual data points (default: 3).
        point_opacity (float): Opacity of individual data points from 0 to 1 (default: 0.7).
        jitter (float): Amount of jitter (spread) applied to the points (default: 0.05).
        point_position (float): Position of the points relative to the violin (default: -0.1).
        fig_width (int): Width of the figure in pixels (default: 1100).
        fig_height (int): Height of the figure in pixels (default: 600).
        show_legend (bool): Whether to display the color legend (default: True).

    Returns:
        Figure: A Plotly `Figure` object containing the violin plot.

    Raises:
        ValueError: If required columns are missing from the DataFrame.
    """
    #* Select value columns to include
    value_cols = [col for col in df.columns if col.startswith(value_col_prefix)]
    
    #* Check if required columns exist
    missing_cols = [col for col in value_cols if col not in df.columns]
    missing_cols.extend([col for col in [group_col, series_col] if col not in df.columns])
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    
    #* Transform to long format
    df_long = pd.melt(
        df,
        id_vars=[group_col, series_col],
        value_vars=value_cols,
        var_name='forecast_step',
        value_name=value_name_legend
    )
    
    #* Extract forecast horizon (e.g., "ape_f1" → "F1")
    df_long["forecast_step"] = df_long["forecast_step"].str.extract(r"f(\d)").radd("F")
    
    #* Determine Color Mapping
    color_map_unique_values = df[group_col].unique().tolist()
    if color_palette is None:
        colors = px.colors.qualitative.Dark24[:len(color_map_unique_values)]
    else:
        colors = color_palette[:len(color_map_unique_values)]
    color_map = dict(zip(color_map_unique_values, colors))
    
    
    #* Create the violin plot
    fig = px.violin(
        df_long,
        x=group_col,
        y=value_name_legend,
        color=group_col,
        color_discrete_map=color_map,
        box=show_box,
        points="all" if show_points else False,
        title=graph_title
    )
    
    # Customize violins
    fig.update_traces(
        points='all' if show_points else False,
        jitter=jitter,
        pointpos=point_position,
        marker=dict(size=point_size, opacity=point_opacity),
        box_visible=show_box,
        meanline_visible=show_meanline
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=50, r=50, t=70, b=50),
        width=fig_width,
        height=fig_height,
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis_title=yaxis_title,
        yaxis=dict(
            range=y_range,
            tickformat=".0%",
            tickfont=dict(size=12),
            title_font=dict(size=14),
            gridcolor='lightgrey',
            gridwidth=1,
            griddash='dot'
        ),
        xaxis_title=xaxis_title,
        xaxis=dict(
            tickfont=dict(size=12),
            title_font=dict(size=14)
        ),
        showlegend=show_legend
    )
    
    
    return fig

def scatterplot_actual_vs_forecast(
    df: pd.DataFrame,
    actual_col_prefix: str = 'actual_values_f',
    forecast_col_prefix: str = 'forecast_values_f',
    group_col: str = "model",
    series_col: str = "series_name",
    window_col: str = "window",
    graph_title: str = "Scatter Plot of Actual vs Forecast by Model",
    add_diagonal_line: bool = True,
    diagonal_color: str = "red",
    fig_width: int = 1000,
    fig_height: int = 600,
    show_legend: bool = True,
    margin_padding: int = 30,
) -> Figure:
    """Creates an interactive plotly scatter plot comparing actual vs forecast values for each model

    This function reshapes the input DataFrame to long format, merges actual and forecast values,
    and generates a Plotly scatter plot. Each point represents a forecasted value for a given actual value,
    colored by model. The plot includes a dropdown menu to filter by series, a filter by model and an 
    optional 45-degree diagonal reference line for perfect forecasts.

    Args:
        df (pd.DataFrame): DataFrame containing actual and forecast columns, as well as grouping columns.
        actual_col_prefix (str): Prefix for columns containing actual values (default: 'actual_values_f').
        forecast_col_prefix (str): Prefix for columns containing forecast values (default: 'forecast_values_f').
        group_col (str): Column to group/color points by, typically 'model' (default: 'model').
        series_col (str): Column indicating the series name for each data point (default: 'series_name').
        window_col (str): Column indicating the backtest window for each data point (default: 'window').
        graph_title (str): Title for the plot (default: "Scatter Plot of Actual vs Forecast by Model").
        add_diagonal_line (bool): Whether to add a 45-degree diagonal reference line (default: True).
        diagonal_color (str): Color of the diagonal reference line (default: "red").
        fig_width (int): Width of the figure in pixels (default: 1000).
        fig_height (int): Height of the figure in pixels (default: 600).
        show_legend (bool): Whether to display the color legend (default: True).
        margin_padding (int): Padding added to axis limits for better visualization (default: 30).

    Returns:
        Figure: A Plotly `Figure` object containing the interactive scatter plot.

    Raises:
        ValueError: If required columns are missing from the DataFrame.
    """
    ###* Prepare DataFrame for graph
    # Select actual and forecast columns
    act_cols = [col for col in df.columns if col.startswith(actual_col_prefix)]
    fcst_cols = [col for col in df.columns if col.startswith(forecast_col_prefix)]

    # Get id columns
    id_vars = [group_col, series_col, window_col]
    
    # Check if required columns exist
    all_required_cols = id_vars + act_cols + fcst_cols
    missing_cols = [col for col in all_required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    
    ## Prepare unified long df of actuals and forecasts
    # Melt actuals
    df_act = df.melt(
        id_vars=id_vars,
        value_vars=act_cols,
        var_name='forecast_step',
        value_name='actual'
    )
    df_act['forecast_step'] = 'F' + df_act['forecast_step'].str.extract(f'{actual_col_prefix}(\\d+)')[0]
    
    # Melt forecasts
    df_fcst = df.melt(
        id_vars=id_vars,
        value_vars=fcst_cols,
        var_name='forecast_step',
        value_name='forecast'
    )
    df_fcst['forecast_step'] = 'F' + df_fcst['forecast_step'].str.extract(f'{forecast_col_prefix}(\\d+)')[0]
    
    # Merge into a single long DataFrame
    df_long = df_act.merge(df_fcst, on=id_vars + ['forecast_step'])

    
    ###* Figure creation
    # Create the base figure
    fig = px.scatter(
        df_long,
        x="actual",
        y="forecast",
        color=group_col, 
        title=graph_title
    )
    
    ###* Dynamic filtering with dropdowns
    # Create series dropdown menu options for dynamic filtering
    series_names = sorted(df_long[series_col].unique())
    series_buttons = []
    
    # Add "All Series" option
    series_buttons.append(
        dict(
            method='update',
            label='All Series',
            args=[
                {'x': [df_long[df_long[group_col] == model]['actual'] for model in df_long[group_col].unique()],
                 'y': [df_long[df_long[group_col] == model]['forecast'] for model in df_long[group_col].unique()]},
                {'title': graph_title}
            ]
        )
    )

    # Add individual series options
    for series_name in series_names:
        filtered_data = df_long[df_long[series_col] == series_name]
        series_buttons.append(
            dict(
                method='update',
                label=str(series_name),
                args=[
                    {'x': [filtered_data[filtered_data[group_col] == model]['actual'] for model in filtered_data[group_col].unique()],
                     'y': [filtered_data[filtered_data[group_col] == model]['forecast'] for model in filtered_data[group_col].unique()]},
                    {'title': graph_title}
                ]
            )
        )
    
    ###* Update layouts
    # Calculate max value for axis limits with some padding
    max_value = df_long[['actual', 'forecast']].max().max() + margin_padding
    
    fig.update_layout(
        margin=dict(l=50, r=50, t=100, b=50),  # Increased top margin for dropdown
        width=fig_width,
        height=fig_height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis_title="Forecast",
        yaxis=dict(
            range=[0, max_value],
            tickfont=dict(size=12),
            title_font=dict(size=14),
            gridcolor="lightgrey",
            gridwidth=1.1,
            griddash="dot",
            showline=True,
            linecolor="grey",
        ),
        xaxis_title="Actual",
        xaxis=dict(
            range=[0, max_value],
            tickfont=dict(size=12),
            title_font=dict(size=14),
            showline=True,
            linecolor="grey",
        ),
        title_font=dict(size=20),
        showlegend=show_legend,
        updatemenus=[
            dict(
                buttons=series_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.21,
                xanchor="right",
                y=0.45,
                yanchor="bottom",
                bgcolor="white",
                bordercolor="lightgrey"
            ),
        ],
        annotations=[
            dict(
                text="Select Series:",
                x=1.15,
                y=0.55,  
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12)
            ),
        ]
    )
    
    # Add a 45-degree diagonal line if requested
    if add_diagonal_line:
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=max_value, y1=max_value,
            line=dict(color=diagonal_color, dash="dot", width=1),
            xref="x",
            yref="y"
        )
    
    return fig

def create_interactive_timeseries_plot(
    df_org: pd.DataFrame,
    df_series: pd.DataFrame,
    series_col: str = "series_name",
    model_col: str = "model",
    step_col: str = "step",
    time_col_actual: str = "period",
    time_col_forecast: str = "forecast_period",
    actual_col: str = "actual",
    forecast_col: str = "forecast_value",
    fig_width: int = 900,
    fig_height: int = 500,
    actual_line_color: str = 'black',
    title_template: str = "Series {series} — Actual vs Forecast ({model})",
    dropdown_width: str = "300px",
    dropdown_description_series: str = "Series:",
    dropdown_description_model: str = "Model:",
    dropdown_description_steps: str = "Forecast Steps:",
    cut_off_date: str = '2017-01-01',
):
    """Creates an interactive time series dashboard for visualizing actual vs forecast values.
    
    This function generates a dashboard using ipywidgets and Plotly, allowing users to interactively
    select a time series, model, and forecast steps to visualize. The dashboard displays the actual
    historical values and the corresponding forecasts.
    
    Args:
        df_org (pd.DataFrame): DataFrame containing actual historical values with columns ['series_id', 'time_index', 'value'].
        df_series (pd.DataFrame): DataFrame containing forecast results in wide format, with forecast columns (e.g., 'forecast_values_f1', ...).
        series_col (str): Column name for the series identifier (default: "series_name").
        model_col (str): Column name for the model identifier (default: "model").
        step_col (str): Column name for the forecast step/horizon (default: "step").
        time_col_actual (str): Column name for the actuals' time index (default: "period").
        time_col_forecast (str): Column name for the forecasted time index (default: "forecast_period").
        actual_col (str): Column name for actual values (default: "actual").
        forecast_col (str): Column name for forecast values (default: "forecast_value").
        fig_width (int): Width of the plot in pixels (default: 900).
        fig_height (int): Height of the plot in pixels (default: 500).
        actual_line_color (str): Color for the actuals line (default: 'black').
        title_template (str): Template for the plot title, with placeholders for series and model (default: "Series {series} — Actual vs Forecast ({model})").
        dropdown_width (str): Width of the dropdown widgets (default: "300px").
        dropdown_description_series (str): Description label for the series dropdown (default: "Series:").
        dropdown_description_model (str): Description label for the model dropdown (default: "Model:").
        dropdown_description_steps (str): Description label for the forecast steps selector (default: "Forecast Steps:").
        cut_off_date (str): Minimum date (inclusive) for both actual and forecast data. Rows before this date are filtered out (default: '2017-01-01').

    Returns:
        widgets.HBox: An ipywidgets HBox containing the dashboard controls for interactive use in a Jupyter environment.
    """
    
    # Prepare data for plotting
    df_actuals, df_forecast = _prepare_data_for_timeseries_plot(
        df_actuals=df_org,
        df_series=df_series,
        forecast_col_prefix='forecast_values_f',
        cut_off_date=cut_off_date,
    )
    
    # Define layout dimensions
    WIDGET_WIDTH = f"{fig_width}px"
    
    # Create dropdowns and selector widgets
    series_dropdown = widgets.Dropdown(
        options=sorted(df_actuals[series_col].unique()),
        description=dropdown_description_series,
        layout=widgets.Layout(width=dropdown_width)
    )
    
    model_dropdown = widgets.Dropdown(
        options=sorted(df_forecast[model_col].unique()),
        description=dropdown_description_model,
        layout=widgets.Layout(width=dropdown_width)
    )
    
    # Create forecast step selector with labels
    step_options = [(f"F{step}", step) for step in sorted(df_forecast[step_col].unique())]
    steps_selector = widgets.SelectMultiple(
        options=step_options,
        value=list(sorted(df_forecast[step_col].unique())),
        description=dropdown_description_steps,
        layout=widgets.Layout(width=dropdown_width, height='60px')
    )
    
    # Pack controls into horizontal layout
    controls = widgets.HBox(
        [series_dropdown, model_dropdown, steps_selector],
        layout=widgets.Layout(
            width=WIDGET_WIDTH,
            justify_content='flex-start',
            align_items='center'
        )
    )
    
    # Define the plot update function
    def update_plot(selected_series, selected_model, selected_steps):
        clear_output(wait=True)
        display(controls)  # Show controls at the top
        
        # Filter data based on selections
        d_hist = df_actuals[df_actuals[series_col] == selected_series]
        d_fc = df_forecast[
            (df_forecast[series_col] == selected_series) &
            (df_forecast[model_col] == selected_model) &
            (df_forecast[step_col].isin(selected_steps))
        ]
        
        # Check if we have data
        if d_hist.empty:
            print(f"No actual data found for series '{selected_series}'")
            return
        
        if d_fc.empty:
            print(f"No forecast data found for the selected combination")
            return
        
        # Build the figure
        fig = go.Figure()
        
        # Add actuals trace
        fig.add_trace(go.Scatter(
            x=d_hist[time_col_actual], 
            y=d_hist[actual_col],
            mode='lines+markers', 
            name='Actual',
            line=dict(color=actual_line_color)
        ))
        
        # Add forecast traces by step
        for step in selected_steps:
            sub = d_fc[d_fc[step_col] == step]
            fig.add_trace(go.Scatter(
                x=sub[time_col_forecast], 
                y=sub[forecast_col],
                mode='lines+markers',
                name=f'Forecast F{step}'
            ))
        
        # Format the title
        title = title_template.format(series=selected_series, model=selected_model)
        
        # Style layout
        fig.update_layout(
            title=title,
            width=fig_width, 
            height=fig_height,
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor='white', 
            paper_bgcolor='white',
            xaxis=dict(
                title='Date', 
                showline=True, 
                linecolor='grey'
            ),
            yaxis=dict(
                title='Value', 
                showline=True, 
                linecolor='grey',
                gridcolor='lightgrey', 
                gridwidth=1, 
                griddash='dot'
            ),
            title_font=dict(size=20)
        )
        
        fig.show()
    
    # Attach observers to widgets
    series_dropdown.observe(
        lambda ev: update_plot(ev.new, model_dropdown.value, steps_selector.value),
        names='value'
    )
    
    model_dropdown.observe(
        lambda ev: update_plot(series_dropdown.value, ev.new, steps_selector.value),
        names='value'
    )
    
    steps_selector.observe(
        lambda ev: update_plot(series_dropdown.value, model_dropdown.value, ev.new),
        names='value'
    )
    
    # Initial render
    update_plot(series_dropdown.value, model_dropdown.value, steps_selector.value)
    
    # Return the control widget for potential later reference
    return controls

def _prepare_data_for_timeseries_plot(
    df_actuals: pd.DataFrame,
    df_series: pd.DataFrame,
    forecast_col_prefix: str = 'forecast_values_f',
    cut_off_date: str = '2017-01-01',
):
    """Prepares and aligns actual and forecast data for time series visualization.

    This function processes two input DataFrames: one containing actual historical values and 
    another containing forecasted values (wide format). It reshapes the forecast DataFrame to 
    long format, computes the forecast period for each forecast step, and filters both actual 
    and forecast data to include only records after a specified cut-off date.

    Args:
        df_actuals (pd.DataFrame): DataFrame with columns ['series_id', 'time_index', 'value'] 
            containing actual historical values.
        df_series (pd.DataFrame): DataFrame containing forecast results in wide format, with 
            forecast columns prefixed by `forecast_col_prefix`.
        forecast_col_prefix (str): Prefix for forecast columns in `df_series` (default: 'forecast_values_f').
        cut_off_date (str): Minimum date (inclusive) for both actual and forecast data. 
            Rows before this date are filtered out (default: '2017-01-01').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - df_act: DataFrame of actual values with columns ['series_name', 'period', 'actual'].
            - df_forecast: DataFrame of forecasts in long format with columns 
                ['model', 'series_name', 'base_date', 'window', 'forecast_value', 'step', 'forecast_period'].
    """
    
    ##* Get historical data from series
    df_act = df_actuals[['series_id','time_index','value']].copy()
    df_act.rename(columns={
        'value': 'actual',
        'series_id': 'series_name',
        'time_index': 'period'
        }, inplace=True
    )
    
        
    ##* Creates long format for forecasts
    # Get forecast columns
    forecast_cols = [c for c in df_series.columns if c.startswith(forecast_col_prefix)]
    n_steps = len(forecast_cols)

    # melt the DataFrame
    df_forecast = df_series.melt(
        id_vars=["model","series_name","base_date", "window","forecast_start","forecast_end"],
        value_vars=forecast_cols,
        var_name="horizon",
        value_name="forecast_value"
    )

    # compute forecast_period for each step
    df_forecast["forecast_start"] = pd.to_datetime(df_forecast["forecast_start"])
    df_forecast["forecast_end"] = pd.to_datetime(df_forecast["forecast_end"])
    df_forecast["step"] = df_forecast["horizon"].str[-1].astype(int)
    df_forecast["forecast_period"] = df_forecast.apply(
        lambda r: (r["forecast_start"] + pd.DateOffset(months=r["step"]-1)),
        axis=1
    )

    # Sort values, drop columns and reset_index
    df_forecast = (df_forecast.sort_values(by=["model", "series_name","base_date", "step"],)
                .drop(columns=['forecast_start', 'forecast_end', 'horizon'])
                .reset_index(drop=True)
    )

    ##* filter only recent data
    cut_off_date = pd.to_datetime(cut_off_date)
    df_act = df_act[df_act['period'] >= cut_off_date].copy()
    df_forecast = df_forecast[df_forecast['forecast_period'] >= cut_off_date].copy()
    
    return df_act, df_forecast

