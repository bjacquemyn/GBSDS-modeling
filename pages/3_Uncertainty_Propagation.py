"""
Uncertainty Propagation - Monte Carlo Simulation

Simulate realistic uncertainty in both baseline distribution and odds ratio,
then observe how this propagates to outcome metrics.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ProportionalOddsModel
from src.components.sidebar import load_settings


def sample_dirichlet(expected_props: np.ndarray, concentration: float, n_samples: int) -> np.ndarray:
    """
    Sample baseline distributions from a Dirichlet distribution.
    
    Args:
        expected_props: Expected proportions (should sum to 100)
        concentration: Higher values = less spread around expected values
        n_samples: Number of samples to draw
        
    Returns:
        Array of shape (n_samples, n_categories) with proportions summing to 100
    """
    # Convert to probabilities and scale by concentration
    alpha = (expected_props / 100.0) * concentration
    # Ensure minimum alpha to avoid numerical issues
    alpha = np.maximum(alpha, 0.01)
    samples = np.random.dirichlet(alpha, size=n_samples) * 100
    return samples


def sample_lognormal_or(expected_or: float, cv: float, n_samples: int) -> np.ndarray:
    """
    Sample odds ratios from a log-normal distribution.
    
    Args:
        expected_or: Expected odds ratio (median)
        cv: Coefficient of variation (higher = more spread)
        n_samples: Number of samples to draw
        
    Returns:
        Array of sampled odds ratios
    """
    # For lognormal: if X ~ LogNormal(mu, sigma), then E[X] = exp(mu + sigma^2/2)
    # We want the median to be expected_or, so mu = log(expected_or)
    mu = np.log(expected_or)
    sigma = cv  # Simplified: sigma directly controls spread
    return np.random.lognormal(mu, sigma, n_samples)


def run_simulation(
    baseline_props: np.ndarray,
    expected_or: float,
    baseline_concentration: float,
    or_cv: float,
    n_simulations: int,
    walking_cutoff: int = 2
) -> dict:
    """
    Run Monte Carlo simulation for uncertainty propagation.
    
    Returns dict with simulation results.
    """
    # Sample baselines and ORs
    baseline_samples = sample_dirichlet(baseline_props, baseline_concentration, n_simulations)
    or_samples = sample_lognormal_or(expected_or, or_cv, n_simulations)
    
    # Track outcomes
    baseline_walking_rates = []
    treatment_walking_rates = []
    risk_differences = []
    treatment_proportions = []  # Track full treatment distributions
    
    for i in range(n_simulations):
        try:
            model = ProportionalOddsModel(baseline_samples[i], walking_cutoff=walking_cutoff)
            result = model.calculate_shift(or_samples[i])
            
            baseline_walking_rates.append(model.baseline_walking_rate)
            treatment_walking_rates.append(result.new_walking)
            risk_differences.append(result.risk_difference)
            treatment_proportions.append(result.new_proportions)
        except ValueError:
            # Skip invalid samples (unlikely but possible)
            continue
    
    return {
        "baseline_samples": baseline_samples,
        "or_samples": or_samples,
        "baseline_walking": np.array(baseline_walking_rates),
        "treatment_walking": np.array(treatment_walking_rates),
        "risk_difference": np.array(risk_differences),
        "treatment_proportions": np.array(treatment_proportions),
        "n_valid": len(risk_differences)
    }


def create_distribution_plot(data: np.ndarray, title: str, xlabel: str, color: str = "#636EFA") -> go.Figure:
    """Create a histogram with statistics overlay."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=50,
        marker_color=color,
        opacity=0.7,
        name="Distribution"
    ))
    
    # Add vertical lines for percentiles
    p5, p50, p95 = np.percentile(data, [5, 50, 95])
    
    for val, label, dash in [(p5, "5th %ile", "dot"), (p50, "Median", "solid"), (p95, "95th %ile", "dot")]:
        fig.add_vline(x=val, line_dash=dash, line_color="red", line_width=2,
                      annotation_text=f"{label}: {val:.1f}%", annotation_position="top")
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title="Frequency",
        template="plotly_white",
        height=350,
        showlegend=False
    )
    
    return fig


def create_or_distribution_plot(data: np.ndarray, expected_or: float) -> go.Figure:
    """Create histogram for odds ratio distribution."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=50,
        marker_color="#EF553B",
        opacity=0.7,
        name="OR Distribution"
    ))
    
    # Add vertical lines
    p5, p50, p95 = np.percentile(data, [5, 50, 95])
    
    fig.add_vline(x=expected_or, line_dash="solid", line_color="blue", line_width=2,
                  annotation_text=f"Expected: {expected_or:.2f}", annotation_position="top left")
    fig.add_vline(x=p5, line_dash="dot", line_color="red", line_width=1,
                  annotation_text=f"5th: {p5:.2f}", annotation_position="bottom left")
    fig.add_vline(x=p95, line_dash="dot", line_color="red", line_width=1,
                  annotation_text=f"95th: {p95:.2f}", annotation_position="bottom right")
    
    fig.update_layout(
        title="Sampled Odds Ratio Distribution",
        xaxis_title="Odds Ratio",
        yaxis_title="Frequency",
        template="plotly_white",
        height=350,
        showlegend=False
    )
    
    return fig


def create_waterfall_uncertainty(results: dict, baseline_props: np.ndarray, expected_or: float) -> go.Figure:
    """Create a visualization showing uncertainty contributions."""
    
    # Calculate deterministic result (no uncertainty)
    try:
        deterministic_model = ProportionalOddsModel(baseline_props)
        deterministic_result = deterministic_model.calculate_shift(expected_or)
        deterministic_rd = deterministic_result.risk_difference
    except:
        deterministic_rd = np.median(results["risk_difference"])
    
    # Overall uncertainty
    rd_std = np.std(results["risk_difference"])
    rd_5, rd_95 = np.percentile(results["risk_difference"], [5, 95])
    
    fig = go.Figure()
    
    # Box plot of risk difference
    fig.add_trace(go.Box(
        y=results["risk_difference"],
        name="Risk Difference",
        boxmean='sd',
        marker_color="#636EFA",
        boxpoints='outliers'
    ))
    
    # Add reference line for deterministic value
    fig.add_hline(y=deterministic_rd, line_dash="dash", line_color="red",
                  annotation_text=f"Deterministic: {deterministic_rd:.1f}%")
    
    fig.update_layout(
        title="Risk Difference Distribution (Uncertainty Range)",
        yaxis_title="Risk Difference (%)",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig


def create_scatter_sensitivity(results: dict) -> go.Figure:
    """Create scatter plots with LOWESS regression lines for binned data."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Impact of Baseline Severity", "Impact of Odds Ratio"),
                        horizontal_spacing=0.15)
    
    # Get data aligned
    or_values = results["or_samples"][:len(results["risk_difference"])]
    baseline_walking = results["baseline_walking"]
    risk_diff = results["risk_difference"]
    
    # Color scales
    or_colorscale = [[0, '#2ecc71'], [0.5, '#f1c40f'], [1, '#e74c3c']]
    baseline_colorscale = [[0, '#e74c3c'], [0.5, '#f1c40f'], [1, '#2ecc71']]
    
    # Number of bins for grouping
    n_bins = 6
    
    # --- Left plot: Baseline walking vs Risk Difference, binned by OR ---
    # Add scatter points first (background layer)
    fig.add_trace(
        go.Scatter(
            x=baseline_walking,
            y=risk_diff,
            mode='markers',
            marker=dict(
                size=3,
                opacity=0.3,
                color=or_values,
                colorscale=or_colorscale,
                cmin=np.min(or_values),
                cmax=np.max(or_values),
                colorbar=dict(
                    title="Odds Ratio",
                    x=0.42,
                    len=0.8,
                    thickness=15
                )
            ),
            name="Data Points",
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    # Bin by OR and fit LOWESS curves
    or_percentiles = np.percentile(or_values, np.linspace(0, 100, n_bins + 1))
    or_bin_indices = np.digitize(or_values, or_percentiles[1:-1])
    
    for bin_idx in range(n_bins):
        mask = or_bin_indices == bin_idx
        if np.sum(mask) < 50:  # Need enough points for LOWESS
            continue
            
        x_bin = baseline_walking[mask]
        y_bin = risk_diff[mask]
        or_bin_mean = np.mean(or_values[mask])
        
        # Sort by x for LOWESS
        sort_idx = np.argsort(x_bin)
        x_sorted = x_bin[sort_idx]
        y_sorted = y_bin[sort_idx]
        
        # Fit LOWESS
        try:
            lowess_result = lowess(y_sorted, x_sorted, frac=0.3, return_sorted=True)
            x_smooth = lowess_result[:, 0]
            y_smooth = lowess_result[:, 1]
            
            # Get color for this bin (interpolate on OR scale)
            or_norm = (or_bin_mean - np.min(or_values)) / (np.max(or_values) - np.min(or_values))
            # Interpolate RGB from green -> yellow -> red
            if or_norm < 0.5:
                r = int(46 + (241 - 46) * (or_norm * 2))
                g = int(204 + (196 - 204) * (or_norm * 2))
                b = int(113 + (15 - 113) * (or_norm * 2))
            else:
                r = int(241 + (231 - 241) * ((or_norm - 0.5) * 2))
                g = int(196 + (76 - 196) * ((or_norm - 0.5) * 2))
                b = int(15 + (60 - 15) * ((or_norm - 0.5) * 2))
            line_color = f'rgb({r},{g},{b})'
            
            fig.add_trace(
                go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode='lines+text',
                    line=dict(width=3, color=line_color),
                    name=f"OR ≈ {or_bin_mean:.2f}",
                    text=[None] * (len(x_smooth) - 1) + [f"OR={or_bin_mean:.1f}"],
                    textposition='top right',
                    textfont=dict(size=10, color=line_color, family='Arial Black'),
                    hovertemplate=f"OR bin: {or_bin_mean:.2f}<br>Baseline: %{{x:.1f}}%<br>RD: %{{y:.1f}}%<extra></extra>",
                    showlegend=False
                ),
                row=1, col=1
            )
        except Exception:
            pass  # Skip if LOWESS fails
    
    # --- Right plot: OR vs Risk Difference, binned by Baseline Walking ---
    # Add scatter points first (background layer)
    fig.add_trace(
        go.Scatter(
            x=or_values,
            y=risk_diff,
            mode='markers',
            marker=dict(
                size=3,
                opacity=0.3,
                color=baseline_walking,
                colorscale=baseline_colorscale,
                cmin=np.min(baseline_walking),
                cmax=np.max(baseline_walking),
                colorbar=dict(
                    title="Baseline<br>Walking %",
                    x=1.02,
                    len=0.8,
                    thickness=15
                )
            ),
            name="Data Points",
            hoverinfo='skip'
        ),
        row=1, col=2
    )
    
    # Bin by baseline walking and fit LOWESS curves
    baseline_percentiles = np.percentile(baseline_walking, np.linspace(0, 100, n_bins + 1))
    baseline_bin_indices = np.digitize(baseline_walking, baseline_percentiles[1:-1])
    
    for bin_idx in range(n_bins):
        mask = baseline_bin_indices == bin_idx
        if np.sum(mask) < 50:  # Need enough points for LOWESS
            continue
            
        x_bin = or_values[mask]
        y_bin = risk_diff[mask]
        baseline_bin_mean = np.mean(baseline_walking[mask])
        
        # Sort by x for LOWESS
        sort_idx = np.argsort(x_bin)
        x_sorted = x_bin[sort_idx]
        y_sorted = y_bin[sort_idx]
        
        # Fit LOWESS
        try:
            lowess_result = lowess(y_sorted, x_sorted, frac=0.3, return_sorted=True)
            x_smooth = lowess_result[:, 0]
            y_smooth = lowess_result[:, 1]
            
            # Get color for this bin (interpolate on baseline scale - reversed: high = green)
            baseline_norm = (baseline_bin_mean - np.min(baseline_walking)) / (np.max(baseline_walking) - np.min(baseline_walking))
            # Interpolate RGB from red -> yellow -> green (reversed)
            if baseline_norm < 0.5:
                r = int(231 + (241 - 231) * (baseline_norm * 2))
                g = int(76 + (196 - 76) * (baseline_norm * 2))
                b = int(60 + (15 - 60) * (baseline_norm * 2))
            else:
                r = int(241 + (46 - 241) * ((baseline_norm - 0.5) * 2))
                g = int(196 + (204 - 196) * ((baseline_norm - 0.5) * 2))
                b = int(15 + (113 - 15) * ((baseline_norm - 0.5) * 2))
            line_color = f'rgb({r},{g},{b})'
            
            fig.add_trace(
                go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode='lines+text',
                    line=dict(width=3, color=line_color),
                    name=f"Baseline ≈ {baseline_bin_mean:.1f}%",
                    text=[None] * (len(x_smooth) - 1) + [f"{baseline_bin_mean:.0f}%"],
                    textposition='top right',
                    textfont=dict(size=10, color=line_color, family='Arial Black'),
                    hovertemplate=f"Baseline bin: {baseline_bin_mean:.1f}%<br>OR: %{{x:.2f}}<br>RD: %{{y:.1f}}%<extra></extra>",
                    showlegend=False
                ),
                row=1, col=2
            )
        except Exception:
            pass  # Skip if LOWESS fails
    
    fig.update_xaxes(title_text="Baseline Walking Rate (%)", row=1, col=1)
    fig.update_xaxes(title_text="Odds Ratio", row=1, col=2)
    fig.update_yaxes(title_text="Risk Difference (%)", row=1, col=1)
    fig.update_yaxes(title_text="Risk Difference (%)", row=1, col=2)
    
    fig.update_layout(
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5,
            font=dict(size=9)
        ),
        title_text="Sensitivity Analysis: LOWESS Regression by Binned Values"
    )
    
    return fig


def create_success_probability_plot(results: dict, thresholds: list) -> go.Figure:
    """Create bar chart showing probability of achieving various thresholds."""
    
    probs = []
    for thresh in thresholds:
        prob = np.mean(results["risk_difference"] >= thresh) * 100
        probs.append(prob)
    
    colors = ['#2ecc71' if p >= 80 else '#f39c12' if p >= 50 else '#e74c3c' for p in probs]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[f"≥{t}%" for t in thresholds],
        y=probs,
        marker_color=colors,
        text=[f"{p:.0f}%" for p in probs],
        textposition='outside'
    ))
    
    fig.add_hline(y=80, line_dash="dash", line_color="gray",
                  annotation_text="80% threshold", annotation_position="right")
    
    fig.update_layout(
        title="Probability of Achieving Risk Difference Targets",
        xaxis_title="Risk Difference Threshold",
        yaxis_title="Probability of Success (%)",
        yaxis_range=[0, 105],
        template="plotly_white",
        height=350
    )
    
    return fig


def create_uncertainty_grotta(results: dict, baseline_props: np.ndarray, labels: list) -> go.Figure:
    """
    Create a Grotta bar chart showing baseline vs treatment with uncertainty bands.
    
    Uses coherent distributions from specific simulations that produced
    the 5th, 50th (median), and 95th percentile risk differences.
    Bars are grouped by scenario with dotted lines connecting disability level boundaries.
    """
    # Color scheme for disability levels (green = good, red = bad)
    colors = [
        "#228B22",  # 0: ForestGreen (Healthy)
        "#90EE90",  # 1: LightGreen (Minor Symptoms)
        "#ADFF2F",  # 2: GreenYellow (Walk Independent)
        "#FFFF00",  # 3: Yellow (Walk Assisted)
        "#FFA500",  # 4: Orange (Bedridden)
        "#FF4500",  # 5: OrangeRed (Ventilated)
    ]
    
    # Find the simulations that produced specific percentiles of risk difference
    risk_diffs = results["risk_difference"]
    
    # Get percentile values
    rd_5 = np.percentile(risk_diffs, 5)
    rd_50 = np.percentile(risk_diffs, 50)
    rd_95 = np.percentile(risk_diffs, 95)
    
    # Find the simulation indices closest to these percentiles
    idx_5 = np.argmin(np.abs(risk_diffs - rd_5))
    idx_50 = np.argmin(np.abs(risk_diffs - rd_50))
    idx_95 = np.argmin(np.abs(risk_diffs - rd_95))
    
    # Get the complete distributions from those specific simulations
    treatment_at_5 = results["treatment_proportions"][idx_5]
    treatment_at_50 = results["treatment_proportions"][idx_50]
    treatment_at_95 = results["treatment_proportions"][idx_95]
    
    baseline_at_5 = results["baseline_samples"][idx_5]
    baseline_at_50 = results["baseline_samples"][idx_50]
    baseline_at_95 = results["baseline_samples"][idx_95]
    
    fig = go.Figure()
    
    # Group by scenario: Treatment and Control adjacent for each percentile
    # Order from top: 95th scenario, Median scenario, 5th scenario
    groups = [
        f"95th %ile: Treatment (RD={rd_95:.1f}%)",
        f"95th %ile: IVIg Control",
        f"Median: Treatment (RD={rd_50:.1f}%)", 
        f"Median: IVIg Control",
        f"5th %ile: Treatment (RD={rd_5:.1f}%)",
        f"5th %ile: IVIg Control",
    ]
    
    group_data = [
        (groups[0], treatment_at_95, 0.8),
        (groups[1], baseline_at_95, 0.8),
        (groups[2], treatment_at_50, 1.0),
        (groups[3], baseline_at_50, 1.0),
        (groups[4], treatment_at_5, 0.8),
        (groups[5], baseline_at_5, 0.8),
    ]
    
    # Add stacked bars for each scenario
    for group_idx, (group_name, proportions, opacity) in enumerate(group_data):
        for i, label in enumerate(labels):
            fig.add_trace(go.Bar(
                y=[group_name],
                x=[proportions[i]],
                name=label if group_idx == 0 else None,  # Only show legend once
                showlegend=(group_idx == 0),
                legendgroup=label,
                orientation='h',
                marker_color=colors[i],
                marker_line=dict(color='#333333', width=1),
                opacity=opacity,
                text=[f"{proportions[i]:.1f}%" if proportions[i] >= 5 else ""],
                textposition='inside',
                textfont=dict(
                    color="white" if i in [0, 4, 5] else "black",
                    size=10,
                    family="Arial Black"
                ),
                hovertemplate=f"{group_name}<br>{label}: %{{x:.1f}}%<extra></extra>",
            ))
    
    # Add annotations for walking independence
    walking_rates = {
        groups[0]: np.sum(treatment_at_95[:3]),
        groups[1]: np.sum(baseline_at_95[:3]),
        groups[2]: np.sum(treatment_at_50[:3]),
        groups[3]: np.sum(baseline_at_50[:3]),
        groups[4]: np.sum(treatment_at_5[:3]),
        groups[5]: np.sum(baseline_at_5[:3]),
    }
    
    fig.update_layout(
        barmode='stack',
        title=dict(
            text="Outcome Distribution with Monte Carlo Uncertainty",
            font=dict(size=16, color='white')
        ),
        xaxis=dict(
            title=dict(text="Percentage of Patients (%)", font=dict(color='white')),
            range=[0, 100],
            ticksuffix="%",
            tickfont=dict(color='white'),
            gridcolor='rgba(255, 255, 255, 0.2)',
        ),
        yaxis=dict(
            title="",
            categoryorder='array',
            categoryarray=list(reversed(groups)),
            tickfont=dict(size=10, color='white')
        ),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02,
            font=dict(size=10, color='white'),
            bgcolor='rgba(0, 0, 0, 0.6)'
        ),
        margin=dict(l=20, r=180, t=60, b=40),
        height=500,
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        hovermode='closest',
        annotations=[
            dict(
                x=105,
                y=group,
                text=f"Walk: {rate:.1f}%",
                showarrow=False,
                xanchor='left',
                font=dict(size=9, color='white'),
            )
            for group, rate in walking_rates.items()
        ]
    )
    
    return fig


def create_uncertainty_grotta_simple(results: dict, baseline_props: np.ndarray, labels: list) -> go.Figure:
    """
    Create a simplified Grotta chart showing just baseline and treatment medians,
    with error bars indicating the 90% CI for each category.
    """
    colors = [
        "#228B22", "#90EE90", "#ADFF2F", 
        "#FFFF00", "#FFA500", "#FF4500"
    ]
    
    # Calculate statistics
    baseline_median = np.median(results["baseline_samples"], axis=0)
    treatment_median = np.median(results["treatment_proportions"], axis=0)
    treatment_5 = np.percentile(results["treatment_proportions"], 5, axis=0)
    treatment_95 = np.percentile(results["treatment_proportions"], 95, axis=0)
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.15,
        subplot_titles=("IVIg Control (Reference)", "Treatment Arm (with 90% CI)")
    )
    
    # Control arm (baseline median)
    cumulative = 0
    for i, label in enumerate(labels):
        fig.add_trace(
            go.Bar(
                x=[baseline_median[i]],
                y=["Control"],
                orientation='h',
                name=label,
                legendgroup=label,
                showlegend=True,
                marker_color=colors[i],
                marker_line=dict(color='white', width=1),
                text=[f"{baseline_median[i]:.1f}%" if baseline_median[i] >= 5 else ""],
                textposition='inside',
                textfont=dict(color="white" if i in [0, 5] else "black", size=10),
                hovertemplate=f"{label}: %{{x:.1f}}%<extra></extra>",
            ),
            row=1, col=1
        )
        cumulative += baseline_median[i]
    
    # Treatment arm (with uncertainty)
    for i, label in enumerate(labels):
        # Main bar (median)
        fig.add_trace(
            go.Bar(
                x=[treatment_median[i]],
                y=["Treatment"],
                orientation='h',
                name=label,
                legendgroup=label,
                showlegend=False,
                marker_color=colors[i],
                marker_line=dict(color='white', width=1),
                text=[f"{treatment_median[i]:.1f}%" if treatment_median[i] >= 5 else ""],
                textposition='inside',
                textfont=dict(color="white" if i in [0, 5] else "black", size=10),
                hovertemplate=(
                    f"{label}<br>"
                    f"Median: {treatment_median[i]:.1f}%<br>"
                    f"90% CI: [{treatment_5[i]:.1f}%, {treatment_95[i]:.1f}%]"
                    f"<extra></extra>"
                ),
            ),
            row=2, col=1
        )
    
    # Add vertical lines showing walking independence threshold position
    control_walking = np.sum(baseline_median[:3])
    treatment_walking = np.sum(treatment_median[:3])
    treatment_walking_5 = np.sum(treatment_5[:3])
    treatment_walking_95 = np.sum(treatment_95[:3])
    
    fig.update_layout(
        barmode='stack',
        title=dict(
            text="Grotta Chart: Outcome Distribution with Uncertainty",
            font=dict(size=16)
        ),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02,
            font=dict(size=10)
        ),
        height=350,
        margin=dict(l=20, r=180, t=80, b=40),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    
    # Update x-axes
    fig.update_xaxes(range=[0, 100], ticksuffix="%", row=1, col=1)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title="Percentage of Patients (%)", row=2, col=1)
    
    # Add walking rate annotations
    fig.add_annotation(
        x=105, y="Control",
        text=f"Walk: {control_walking:.1f}%",
        showarrow=False, xanchor='left',
        font=dict(size=10, color='#666'),
        xref='x', yref='y'
    )
    
    fig.add_annotation(
        x=105, y="Treatment",
        text=f"Walk: {treatment_walking:.1f}%<br><span style='font-size:9px'>[{treatment_walking_5:.1f}%-{treatment_walking_95:.1f}%]</span>",
        showarrow=False, xanchor='left',
        font=dict(size=10, color='#666'),
        xref='x2', yref='y2'
    )
    
    return fig


def main():
    settings = load_settings()
    labels = settings.get("disability_scale", {}).get("labels", [
        "0: Healthy", "1: Minor Symptoms", "2: Walk Independent",
        "3: Walk Assisted", "4: Bedridden", "5: Ventilated"
    ])
    
    st.header("🎲 Uncertainty Propagation")
    st.markdown("""
    **Monte Carlo Simulation**: Explore how uncertainty in baseline distribution and odds ratio 
    propagates to outcome predictions. This helps assess confidence in trial projections.
    """)
    
    # Check if baseline is available
    if "baseline_props" not in st.session_state:
        st.warning("⚠️ Please configure baseline proportions on the **Home** page first.")
        st.stop()
    
    baseline_props = np.array(st.session_state.baseline_props)
    
    st.divider()
    
    # Simulation Parameters
    st.subheader("⚙️ Simulation Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Treatment Effect")
        expected_or = st.slider(
            "Expected Odds Ratio",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Central estimate for the treatment effect"
        )
        or_cv = st.slider(
            "OR Uncertainty (CV)",
            min_value=0.05,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Coefficient of variation: higher = more uncertainty. 0.2 = 95% CI roughly [0.7×, 1.4×] of expected"
        )
    
    with col2:
        st.markdown("##### Baseline Uncertainty")
        baseline_concentration = st.slider(
            "Baseline Precision",
            min_value=10.0,
            max_value=500.0,
            value=100.0,
            step=10.0,
            help="Higher values = less variation around expected baseline. 100 = moderate uncertainty, 500 = high confidence"
        )
    
    with col3:
        st.markdown("##### Simulation Settings")
        n_simulations = st.select_slider(
            "Number of Simulations",
            options=[1000, 5000, 10000, 25000, 50000],
            value=10000,
            help="More simulations = more stable results, but slower"
        )
    
    # Run simulation button
    st.divider()
    
    if st.button("🚀 Run Monte Carlo Simulation", type="primary", use_container_width=True):
        with st.spinner(f"Running {n_simulations:,} simulations..."):
            # Set seed for reproducibility within session
            np.random.seed(42)
            
            results = run_simulation(
                baseline_props=baseline_props,
                expected_or=expected_or,
                baseline_concentration=baseline_concentration,
                or_cv=or_cv,
                n_simulations=n_simulations
            )
            
            st.session_state.mc_results = results
            st.session_state.mc_params = {
                "expected_or": expected_or,
                "or_cv": or_cv,
                "baseline_concentration": baseline_concentration,
                "n_simulations": n_simulations,
                "baseline_props": baseline_props
            }
    
    # Display results if available
    if "mc_results" in st.session_state:
        results = st.session_state.mc_results
        params = st.session_state.mc_params
        
        st.divider()
        st.subheader("📊 Simulation Results")
        
        # GBS Disability Scale Reference
        with st.expander("📋 GBS Disability Scale Reference", expanded=False):
            st.markdown("""
            The **GBS Disability Scale** is a 7-point scale (0-6) used to measure functional status 
            in Guillain-Barré Syndrome patients. This simulation uses a simplified 6-level version (0-5):
            
            | Level | Description | Functional Status |
            |:-----:|:------------|:------------------|
            | **0** | Healthy | No symptoms at all |
            | **1** | Minor symptoms | Able to run; minor symptoms or signs |
            | **2** | Walk independently | Able to walk ≥10 meters without assistance |
            | **3** | Walk with assistance | Able to walk ≥10 meters with help (walker, cane, person) |
            | **4** | Bedridden/chairbound | Unable to walk; confined to bed or wheelchair |
            | **5** | Requiring ventilation | Needs mechanical ventilation for breathing |
            
            ---
            
            ### What does "Walking Independently" mean?
            
            In this analysis, **"walking independently"** refers to patients at levels **0, 1, or 2** — 
            those who can walk at least 10 meters without any physical assistance from another person 
            or device.
            
            This is the **primary clinical endpoint** used in most GBS trials because:
            - It represents meaningful functional recovery
            - It's objective and easy to assess
            - It directly impacts patient quality of life and independence
            
            The **Risk Difference** shown in this simulation represents the additional percentage 
            of patients who achieve walking independence with the new treatment compared to IVIg alone.
            """)
        
        
        # Key summary metrics
        rd_mean = np.mean(results["risk_difference"])
        rd_median = np.median(results["risk_difference"])
        rd_5, rd_95 = np.percentile(results["risk_difference"], [5, 95])
        rd_std = np.std(results["risk_difference"])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Risk Difference", f"{rd_mean:.1f}%")
        col2.metric("Median Risk Difference", f"{rd_median:.1f}%")
        col3.metric("90% CI", f"[{rd_5:.1f}%, {rd_95:.1f}%]")
        col4.metric("Std Dev", f"±{rd_std:.1f}%")
        
        # Probability of success metrics
        st.markdown("##### Probability of Success")
        success_cols = st.columns(5)
        thresholds = [5, 10, 15, 20, 25]
        for i, thresh in enumerate(thresholds):
            prob = np.mean(results["risk_difference"] >= thresh) * 100
            success_cols[i].metric(f"P(RD ≥ {thresh}%)", f"{prob:.0f}%")
        
        # Definitions section
        with st.expander("📖 Understanding These Metrics", expanded=False):
            st.markdown("""
            ### What is Risk Difference?
            
            **Risk Difference** measures how much better (or worse) patients do with the new treatment 
            compared to the standard treatment (IVIg).
            
            - It's the **percentage of additional patients** who can walk independently after treatment
            - Example: A risk difference of **+15%** means that for every 100 patients treated:
              - With standard IVIg: ~50 patients might walk independently
              - With the new drug: ~65 patients would walk independently
              - **15 extra patients benefit** from the new treatment
            
            **How to interpret the values:**
            - **Mean/Median**: The "best guess" for what improvement we expect
            - **90% CI**: The range where we're 90% confident the true improvement falls
            - **Std Dev**: How spread out the possible outcomes are
            
            ---
            
            ### What is Probability of Success?
            
            **Probability of Success** answers the question: *"How likely is it that the trial will 
            show at least X% improvement?"*
            
            - It's calculated by running thousands of simulated trials
            - We count how many achieve the target, then convert to a percentage
            
            **Example interpretation:**
            - **P(RD ≥ 10%) = 85%** means:
              - In 85 out of 100 possible trial scenarios
              - The new treatment would show at least 10% more patients walking
            - **P(RD ≥ 20%) = 40%** means:
              - There's only a 40% chance of hitting this ambitious target
              - Consider whether a 20% target is realistic
            
            **Rule of thumb:**
            - 🟢 **≥80%**: High confidence — target is achievable
            - 🟡 **50-79%**: Moderate confidence — some risk of falling short  
            - 🔴 **<50%**: Low confidence — target may be too ambitious
            """)
        
        st.divider()
        
        # Visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Outcome Distribution", 
            "🎯 Input Distributions",
            "🔍 Sensitivity Analysis",
            "✅ Success Probability"
        ])
        
        with tab1:
            # Uncertainty Grotta Chart Section
            st.markdown("#### Grotta Chart with Uncertainty")
            st.markdown("""
            This stacked bar chart shows how the disability distribution shifts from control to treatment, 
            with **uncertainty bands** showing the range of possible outcomes from the Monte Carlo simulation.
            """)
            
            # Show the detailed uncertainty Grotta chart
            fig_grotta = create_uncertainty_grotta(results, params["baseline_props"], labels)
            st.plotly_chart(fig_grotta, use_container_width=True)
            
            st.caption("""
            **Reading this chart:** 
            - The median bars show the most likely outcome distribution
            - The 5th and 95th percentile bars show the range of plausible outcomes (90% of scenarios fall within this range)
            - Faded bars indicate uncertainty bounds; compare their walking independence rates on the right
            """)
            
            st.divider()
            
            st.markdown("#### Risk Difference Distribution")
            st.markdown("The distribution of absolute improvement in walking independence across all simulations.")
            fig = create_distribution_plot(
                results["risk_difference"],
                "Risk Difference Distribution (Treatment - Control)",
                "Risk Difference (%)",
                "#636EFA"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Also show walking rates
            col1, col2 = st.columns(2)
            with col1:
                fig_baseline = create_distribution_plot(
                    results["baseline_walking"],
                    "Baseline Walking Rate (Control)",
                    "Walking Rate (%)",
                    "#00CC96"
                )
                st.plotly_chart(fig_baseline, use_container_width=True)
            with col2:
                fig_treatment = create_distribution_plot(
                    results["treatment_walking"],
                    "Treatment Walking Rate",
                    "Walking Rate (%)",
                    "#AB63FA"
                )
                st.plotly_chart(fig_treatment, use_container_width=True)
        
        with tab2:
            st.markdown("#### Input Parameter Distributions")
            st.markdown("These distributions show the simulated uncertainty in your inputs.")
            
            # OR distribution
            fig_or = create_or_distribution_plot(
                results["or_samples"][:results["n_valid"]], 
                params["expected_or"]
            )
            st.plotly_chart(fig_or, use_container_width=True)
            
            # Baseline samples visualization - show full distribution per level
            st.markdown("##### Sampled Baseline Distributions")
            st.markdown("Distribution of sampled baseline proportions for each GBS disability level:")
            
            # Create violin/box plot for each disability level
            fig_baselines = go.Figure()
            
            # Colors for each disability level
            level_colors = [
                "#228B22",  # 0: ForestGreen (Healthy)
                "#90EE90",  # 1: LightGreen (Minor Symptoms)
                "#ADFF2F",  # 2: GreenYellow (Walk Independent)
                "#FFFF00",  # 3: Yellow (Walk Assisted)
                "#FFA500",  # 4: Orange (Bedridden)
                "#FF4500",  # 5: OrangeRed (Ventilated)
            ]
            
            # Add violin plot for each disability level
            for i, label in enumerate(labels):
                level_samples = results["baseline_samples"][:, i]
                
                fig_baselines.add_trace(go.Violin(
                    x=[label] * len(level_samples),
                    y=level_samples,
                    name=label,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=level_colors[i],
                    line_color='#333333',
                    opacity=0.7,
                    showlegend=False,
                    points=False,  # Don't show individual points for performance
                ))
            
            # Add expected baseline as scatter points with connecting line
            fig_baselines.add_trace(go.Scatter(
                x=labels,
                y=params["baseline_props"],
                mode='markers+lines',
                name="Expected Baseline",
                marker=dict(size=14, color='red', symbol='diamond', line=dict(color='white', width=2)),
                line=dict(color='red', width=3, dash='dash')
            ))
            
            fig_baselines.update_layout(
                title="Baseline Distribution Uncertainty per GBS Disability Level",
                xaxis_title="GBS Disability Level",
                yaxis_title="Proportion (%)",
                template="plotly_white",
                height=450,
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5
                ),
                xaxis=dict(tickangle=-15)
            )
            st.plotly_chart(fig_baselines, use_container_width=True)
            
            # Add summary statistics table
            st.markdown("##### Distribution Statistics per Level")
            stats_data = []
            for i, label in enumerate(labels):
                level_samples = results["baseline_samples"][:, i]
                stats_data.append({
                    "GBS Level": label,
                    "Expected": f"{params['baseline_props'][i]:.1f}%",
                    "Median": f"{np.median(level_samples):.1f}%",
                    "5th %ile": f"{np.percentile(level_samples, 5):.1f}%",
                    "95th %ile": f"{np.percentile(level_samples, 95):.1f}%",
                    "Std Dev": f"±{np.std(level_samples):.1f}%"
                })
            st.dataframe(stats_data, use_container_width=True, hide_index=True)
        
        with tab3:
            st.markdown("#### Sensitivity Analysis")
            st.markdown("""
            These scatter plots help you understand which input drives more of the outcome uncertainty:
            - **Left**: How does baseline severity affect the risk difference?
            - **Right**: How does the odds ratio affect the risk difference?
            """)
            
            fig_scatter = create_scatter_sensitivity(results)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Correlation analysis
            corr_baseline = np.corrcoef(results["baseline_walking"], results["risk_difference"])[0, 1]
            corr_or = np.corrcoef(results["or_samples"][:len(results["risk_difference"])], results["risk_difference"])[0, 1]
            
            st.markdown("##### Correlation with Risk Difference")
            col1, col2 = st.columns(2)
            col1.metric("Baseline Walking Rate", f"r = {corr_baseline:.3f}")
            col2.metric("Odds Ratio", f"r = {corr_or:.3f}")
            
            # Interpretation
            dominant = "Odds Ratio" if abs(corr_or) > abs(corr_baseline) else "Baseline Distribution"
            st.info(f"**Insight**: {dominant} has a stronger influence on outcome uncertainty in this scenario.")
        
        with tab4:
            st.markdown("#### Probability of Achieving Target Improvements")
            st.markdown("What is the probability that the trial will show at least a given improvement?")
            
            fig_success = create_success_probability_plot(results, [5, 10, 15, 20, 25])
            st.plotly_chart(fig_success, use_container_width=True)
            
            # Custom threshold analysis
            st.markdown("##### Custom Threshold Analysis")
            custom_threshold = st.number_input(
                "Enter custom risk difference threshold (%)",
                min_value=0.0,
                max_value=50.0,
                value=15.0,
                step=1.0
            )
            
            prob_custom = np.mean(results["risk_difference"] >= custom_threshold) * 100
            if prob_custom >= 80:
                st.success(f"✅ Probability of achieving ≥{custom_threshold}% improvement: **{prob_custom:.1f}%**")
            elif prob_custom >= 50:
                st.warning(f"⚠️ Probability of achieving ≥{custom_threshold}% improvement: **{prob_custom:.1f}%**")
            else:
                st.error(f"❌ Probability of achieving ≥{custom_threshold}% improvement: **{prob_custom:.1f}%**")
        
        # Methodology expander
        with st.expander("📚 Methodology"):
            st.markdown(f"""
            ### Monte Carlo Uncertainty Propagation
            
            This simulation samples from probability distributions for both inputs:
            
            **1. Baseline Distribution Uncertainty**
            - Sampled from a **Dirichlet distribution** centered on your expected baseline
            - Concentration parameter: **{params['baseline_concentration']:.0f}** (higher = less variation)
            - This reflects uncertainty in the true population distribution
            
            **2. Odds Ratio Uncertainty**
            - Sampled from a **Log-Normal distribution** centered on OR = **{params['expected_or']:.2f}**
            - Coefficient of variation: **{params['or_cv']:.2f}** (higher = more spread)
            - Log-normal ensures OR is always positive and captures asymmetric confidence intervals
            
            **3. Simulation Process**
            - For each of **{params['n_simulations']:,}** iterations:
              1. Sample a baseline distribution
              2. Sample an odds ratio
              3. Apply proportional odds model to calculate treatment distribution
              4. Compute walking rate and risk difference
            - Results show the distribution of possible outcomes
            
            **Interpretation**
            - The 90% CI represents the range where 90% of simulated outcomes fall
            - "Probability of success" shows the fraction of simulations achieving targets
            - Sensitivity analysis reveals which input drives more uncertainty
            """)


if __name__ == "__main__":
    main()
else:
    main()
