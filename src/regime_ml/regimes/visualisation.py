"""
Visualization utilities for regime detection models.

Provides functions to visualize:
- Regime classifications over time
- Feature distributions by regime
- Transition matrices
- Regime periods and timelines
- Market performance by regime
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Optional, Dict, List
import yfinance as yf


def plot_regime_timeseries(
    features: pd.DataFrame,
    regimes: np.ndarray,
    proba: np.ndarray,
    regime_names: Optional[Dict[int, str]] = None,
    title: str = "Regime Classification Over Time"
) -> go.Figure:
    """
    Plot regime classification with features and probabilities.
    
    Creates a 3-panel plot showing:
    1. Discrete regime labels over time
    2. Underlying feature values
    3. Regime probabilities (stacked area chart)
    
    Args:
        features: Feature dataframe with DatetimeIndex
        regimes: Regime labels (n_samples,)
        proba: Regime probabilities (n_samples, n_regimes)
        regime_names: Optional dict mapping regime numbers to names
        title: Plot title
        
    Returns:
        Plotly figure
    """
    if regime_names is None:
        regime_names = {i: f"Regime {i}" for i in range(proba.shape[1])}
    
    # Map regimes to names for display
    regime_labels = [regime_names[r] for r in regimes]
    
    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['Regime Classification', 'Feature Values', 'Regime Probabilities'],
        row_heights=[0.2, 0.5, 0.3],
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # 1. Regime timeseries as colored background
    # Convert regime names to numeric for plotting
    regime_numeric = regimes.copy()
    
    # Use scatter with mode='lines' for step-like appearance
    fig.add_trace(
        go.Scatter(
            x=features.index,
            y=regime_numeric,
            mode='lines',
            name='Regime',
            line=dict(width=3, shape='hv'),  # 'hv' makes it step-like
            hovertemplate='%{text}<br>Date: %{x}<extra></extra>',
            text=regime_labels
        ),
        row=1, col=1
    )
    
    # Update y-axis to show regime names
    fig.update_yaxes(
        tickmode='array',
        tickvals=list(regime_names.keys()),
        ticktext=list(regime_names.values()),
        row=1, col=1
    )
    
    # 2. Feature values
    colors = px.colors.qualitative.Set2
    for i, col in enumerate(features.columns):
        fig.add_trace(
            go.Scatter(
                x=features.index,
                y=features[col],
                name=col,
                opacity=0.8,
                line=dict(width=1.5, color=colors[i % len(colors)])
            ),
            row=2, col=1
        )
    
    # Add horizontal line at y=0 for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1) # type: ignore
    
    # 3. Regime probabilities (stacked area)
    for i in range(proba.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=features.index,
                y=proba[:, i],
                name=f'P({regime_names[i]})',
                stackgroup='one',
                mode='lines',
                line=dict(width=0.5),
                fillcolor=colors[i % len(colors)]
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=1000,
        title=title,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Update x-axes
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    # Update y-axes
    fig.update_yaxes(title_text="Regime", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    fig.update_yaxes(title_text="Probability", row=3, col=1, range=[0, 1])
    
    return fig


def plot_regime_distributions(
    features: pd.DataFrame,
    regimes: np.ndarray,
    regime_names: Optional[Dict[int, str]] = None,
    title: str = "Feature Distributions by Regime"
) -> go.Figure:
    """
    Plot feature distributions for each regime using box plots.
    
    Args:
        features: Feature dataframe
        regimes: Regime labels (n_samples,)
        regime_names: Optional dict mapping regime numbers to names
        title: Plot title
        
    Returns:
        Plotly figure
    """
    if regime_names is None:
        regime_names = {i: f"Regime {i}" for i in np.unique(regimes)}
    
    # Combine features and regimes
    df_plot = features.copy()
    df_plot['regime'] = [regime_names[r] for r in regimes]
    
    # Melt for plotting
    df_melt = df_plot.melt(id_vars=['regime'], var_name='feature', value_name='value')
    
    # Create box plots
    fig = px.box(
        df_melt,
        x='regime',
        y='value',
        color='regime',
        facet_col='feature',
        facet_col_wrap=2,
        title=title,
        height=400 * ((len(features.columns) + 1) // 2)
    )
    
    # Update layout
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="Z-Score")
    
    # Add horizontal line at y=0 for reference on all facets
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
    
    return fig


def plot_transition_matrix(
    transition_matrix: np.ndarray,
    regime_names: Optional[Dict[int, str]] = None,
    title: str = "Regime Transition Probabilities"
) -> go.Figure:
    """
    Plot regime transition matrix as heatmap.
    
    Args:
        transition_matrix: Transition probability matrix (n_regimes, n_regimes)
        regime_names: Optional dict mapping regime numbers to names
        title: Plot title
        
    Returns:
        Plotly figure
    """
    n_regimes = transition_matrix.shape[0]
    
    if regime_names is None:
        regime_names = {i: f"Regime {i}" for i in range(n_regimes)}
    
    labels = [regime_names[i] for i in range(n_regimes)]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=transition_matrix,
        x=[f"To {label}" for label in labels],
        y=[f"From {label}" for label in labels],
        colorscale='Blues',
        text=np.round(transition_matrix, 3),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Probability")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Next Regime",
        yaxis_title="Current Regime",
        height=500,
        width=600
    )
    
    return fig


def plot_regime_periods(
    features: pd.DataFrame,
    regimes: np.ndarray,
    regime_names: Optional[Dict[int, str]] = None,
    min_duration_days: int = 30,
    title: str = "Regime Timeline"
) -> go.Figure:
    """
    Plot regime periods as a Gantt-style timeline.
    
    Args:
        features: Feature dataframe with DatetimeIndex
        regimes: Regime labels (n_samples,)
        regime_names: Optional dict mapping regime numbers to names
        min_duration_days: Only show regimes lasting at least this many days
        title: Plot title
        
    Returns:
        Plotly figure
    """
    if regime_names is None:
        regime_names = {i: f"Regime {i}" for i in np.unique(regimes)}
    
    # Find regime periods
    regime_periods = []
    current_regime = regimes[0]
    start_date = features.index[0]
    
    for i, r in enumerate(regimes[1:], 1):
        if r != current_regime:
            end_date = features.index[i-1]
            duration_days = (end_date - start_date).days
            
            if duration_days >= min_duration_days:
                regime_periods.append({
                    'regime': regime_names[current_regime],
                    'regime_num': current_regime,
                    'start': start_date,
                    'end': end_date,
                    'duration_days': duration_days
                })
            
            current_regime = r
            start_date = features.index[i]
    
    # Add last period
    end_date = features.index[-1]
    duration_days = (end_date - start_date).days
    if duration_days >= min_duration_days:
        regime_periods.append({
            'regime': regime_names[current_regime],
            'regime_num': current_regime,
            'start': start_date,
            'end': end_date,
            'duration_days': duration_days
        })
    
    periods_df = pd.DataFrame(regime_periods)
    
    # Create Gantt chart
    fig = px.timeline(
        periods_df,
        x_start='start',
        x_end='end',
        y='regime',
        color='regime',
        hover_data=['duration_days'],
        title=title
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Regime",
        height=400,
        showlegend=True
    )
    
    return fig


def plot_regime_confidence(
    features: pd.DataFrame,
    regimes: np.ndarray,
    proba: np.ndarray,
    regime_names: Optional[Dict[int, str]] = None,
    confidence_threshold: float = 0.7,
    title: str = "Regime Classification Confidence"
) -> go.Figure:
    """
    Plot regime confidence levels over time.
    
    Shows the maximum probability (confidence) for the predicted regime,
    highlighting periods where the model is uncertain.
    
    Args:
        features: Feature dataframe with DatetimeIndex
        regimes: Regime labels (n_samples,)
        proba: Regime probabilities (n_samples, n_regimes)
        regime_names: Optional dict mapping regime numbers to names
        confidence_threshold: Threshold for highlighting low confidence periods
        title: Plot title
        
    Returns:
        Plotly figure
    """
    if regime_names is None:
        regime_names = {i: f"Regime {i}" for i in range(proba.shape[1])}
    
    # Calculate confidence (max probability)
    confidence = proba.max(axis=1)
    regime_labels = [regime_names[r] for r in regimes]
    
    # Create figure
    fig = go.Figure()
    
    # Add confidence line
    fig.add_trace(go.Scatter(
        x=features.index,
        y=confidence,
        mode='lines',
        name='Confidence',
        line=dict(color='blue', width=1),
        fill='tozeroy',
        fillcolor='rgba(0, 100, 255, 0.2)',
        hovertemplate='Confidence: %{y:.2f}<br>Regime: %{text}<br>Date: %{x}<extra></extra>',
        text=regime_labels
    ))
    
    # Add confidence threshold line
    fig.add_hline(
        y=confidence_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Confidence Threshold ({confidence_threshold})",
        annotation_position="right"
    )
    
    # Highlight low confidence periods
    low_confidence_mask = confidence < confidence_threshold
    if low_confidence_mask.any():
        fig.add_trace(go.Scatter(
            x=features.index[low_confidence_mask],
            y=confidence[low_confidence_mask],
            mode='markers',
            name='Low Confidence',
            marker=dict(color='red', size=4, symbol='x'),
            hovertemplate='Low Confidence: %{y:.2f}<br>Date: %{x}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Confidence (Max Probability)",
        yaxis_range=[0, 1],
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_regime_summary_table(
    regimes: np.ndarray,
    regime_means: np.ndarray,
    feature_names: List[str],
    regime_names: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Create a summary table of regime characteristics.
    
    Args:
        regimes: Regime labels (n_samples,)
        regime_means: Mean feature values for each regime (n_regimes, n_features)
        feature_names: List of feature names
        regime_names: Optional dict mapping regime numbers to names
        
    Returns:
        Summary dataframe
    """
    n_regimes = regime_means.shape[0]
    
    if regime_names is None:
        regime_names = {i: f"Regime {i}" for i in range(n_regimes)}
    
    # Create means dataframe
    means_df = pd.DataFrame(
        regime_means,
        columns=feature_names,  # type: ignore
        index=[regime_names[i] for i in range(n_regimes)]  # type: ignore
    )
    
    # Add regime statistics
    unique, counts = np.unique(regimes, return_counts=True)
    regime_counts = dict(zip(unique, counts))
    
    means_df['Observations'] = [regime_counts.get(i, 0) for i in range(n_regimes)]
    means_df['Percentage'] = means_df['Observations'] / len(regimes) * 100
    
    # Calculate persistence
    regime_lengths = []
    current_regime = regimes[0]
    current_length = 1
    regime_duration_map = {i: [] for i in range(n_regimes)}
    
    for r in regimes[1:]:
        if r == current_regime:
            current_length += 1
        else:
            regime_duration_map[current_regime].append(current_length)
            current_regime = r
            current_length = 1
    regime_duration_map[current_regime].append(current_length)
    
    means_df['Avg_Duration'] = [
        np.mean(regime_duration_map[i]) if regime_duration_map[i] else 0
        for i in range(n_regimes)
    ]
    
    return means_df.round(2)


def plot_ticker_by_regime(
    ticker: str,
    regime_index: pd.DatetimeIndex,
    regimes: np.ndarray,
    regime_names: Optional[Dict[int, str]] = None,
    price_type: str = 'Close',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    title: Optional[str] = None
) -> go.Figure:
    """
    Plot a ticker's price over time, colored by regime as scatter points.
    
    Downloads ticker data from yfinance and overlays regime classifications
    to visualize how market performance aligns with different regimes.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'SPY', '^GSPC', 'QQQ')
        regime_index: DatetimeIndex from the regime classification
        regimes: Regime labels (n_samples,)
        regime_names: Optional dict mapping regime numbers to names
        price_type: Price column to plot ('Close', 'Adj Close', 'Open', etc.)
        start_date: Optional start date (YYYY-MM-DD), defaults to regime_index start
        end_date: Optional end date (YYYY-MM-DD), defaults to regime_index end
        title: Plot title, defaults to f"{ticker} Price by Regime"
        
    Returns:
        Plotly figure with price chart colored by regime
        
    Example:
        >>> fig = plot_ticker_by_regime('SPY', features.index, regimes, regime_names)
        >>> fig.show()
    """
    if regime_names is None:
        regime_names = {i: f"Regime {i}" for i in np.unique(regimes)}
    
    # Set date range
    if start_date is None:
        start_date = regime_index.min().strftime('%Y-%m-%d')  # type: ignore
    if end_date is None:
        end_date = regime_index.max().strftime('%Y-%m-%d')  # type: ignore
    
    if title is None:
        title = f"{ticker} Price by Regime"
    
    # Download ticker data
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)  # type: ignore
    
    if ticker_data.empty:  # type: ignore
        raise ValueError(f"No data downloaded for ticker {ticker}")
    
    # Handle MultiIndex columns (yfinance sometimes returns these)
    if isinstance(ticker_data.columns, pd.MultiIndex):  # type: ignore
        ticker_data.columns = ticker_data.columns.get_level_values(0)  # type: ignore
    
    # Ensure we have the requested price type
    if price_type not in ticker_data.columns:  # type: ignore
        print(f"Warning: {price_type} not found, using Close instead")
        price_type = 'Close'
    
    # Create regime dataframe
    regime_df = pd.DataFrame({
        'regime': regimes,
        'regime_name': [regime_names[r] for r in regimes]
    }, index=regime_index)
    
    # Align ticker data with regime data (forward fill regimes to match ticker dates)
    combined = ticker_data[[price_type]].copy()  # type: ignore
    combined = combined.join(regime_df, how='left')
    
    # Forward fill regimes for dates that fall between regime classification dates
    combined['regime'] = combined['regime'].ffill()
    combined['regime_name'] = combined['regime_name'].ffill()
    
    # Drop rows where we don't have regime data (before first regime or after last)
    combined = combined.dropna(subset=['regime'])
    
    # Convert regime to int for color mapping
    combined['regime'] = combined['regime'].astype(int)
    
    # Create single figure
    fig = go.Figure()
    
    # Get color palette
    colors = px.colors.qualitative.Set2
    regime_colors = {i: colors[i % len(colors)] for i in regime_names.keys()}
    
    # Plot price as scatter points colored by regime
    for regime_num, regime_name in regime_names.items():
        regime_mask = combined['regime'] == regime_num
        regime_data = combined[regime_mask]
        
        if not regime_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=regime_data.index,
                    y=regime_data[price_type],
                    mode='markers',
                    name=regime_name,
                    marker=dict(
                        color=regime_colors[regime_num],
                        size=4,
                        line=dict(width=0)
                    ),
                    hovertemplate=f'{regime_name}<br>Price: %{{y:.2f}}<br>Date: %{{x}}<extra></extra>',
                    legendgroup=regime_name
                )
            )
    
    # Update layout
    fig.update_layout(
        height=600,
        title=title,
        xaxis_title="Date",
        yaxis_title=f"{ticker} Price ($)",
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig
