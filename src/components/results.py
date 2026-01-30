"""Results display components."""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Optional

from ..models import ShiftResult


def render_metrics(result: ShiftResult, mode: str = "forward") -> None:
    """Render key metrics in columns.
    
    Args:
        result: ShiftResult from the proportional odds model.
        mode: Either "forward" (show OR) or "goal" (show target).
    """
    col1, col2, col3 = st.columns(3)
    
    if mode == "forward":
        col1.metric(
            "Odds Ratio",
            f"{result.odds_ratio:.2f}",
            help="Common odds ratio applied across all thresholds"
        )
    else:
        col1.metric(
            "Required Odds Ratio",
            f"{result.odds_ratio:.3f}",
            help="OR needed to achieve target improvement"
        )
    
    col2.metric(
        "Walking Independent (Control)",
        f"{result.control_walking:.1f}%",
        help="GBS Score 0-2 in IVIg arm"
    )
    
    delta_color = "normal" if result.risk_difference >= 0 else "inverse"
    col3.metric(
        "Walking Independent (Treatment)",
        f"{result.new_walking:.1f}%",
        delta=f"{result.risk_difference:+.1f}%",
        delta_color=delta_color,
        help="GBS Score 0-2 in investigational arm"
    )


def render_comparison_table(
    control_props: np.ndarray,
    new_props: np.ndarray,
    labels: list[str],
    odds_ratio: float = None
) -> pd.DataFrame:
    """Render and return the detailed comparison table.
    
    Args:
        control_props: Control group proportions (%).
        new_props: Treatment group proportions (%).
        labels: Category labels.
        odds_ratio: The odds ratio used (optional, for display).
    
    Returns:
        DataFrame with comparison data.
    """
    st.subheader("📋 Detailed Distribution Table")
    
    # Calculate cumulative percentages
    control_cum = np.cumsum(control_props)
    new_cum = np.cumsum(new_props)
    
    # Calculate odds: P(Y <= k) / P(Y > k)
    # Avoid division by zero for the last category (cumulative = 100%)
    def calculate_odds(cum_pct):
        """Calculate odds from cumulative percentage."""
        odds = []
        for cp in cum_pct:
            if cp >= 100.0 or cp <= 0.0:
                odds.append(np.nan)  # undefined for 0% or 100%
            else:
                odds.append(cp / (100.0 - cp))
        return odds
    
    control_odds = calculate_odds(control_cum)
    new_odds = calculate_odds(new_cum)
    
    # Calculate odds ratio at each threshold
    calculated_or = []
    for co, no in zip(control_odds, new_odds):
        if np.isnan(co) or np.isnan(no) or co == 0:
            calculated_or.append(np.nan)
        else:
            calculated_or.append(no / co)
    
    df = pd.DataFrame({
        "Category": labels,
        "Control %": control_props,
        "Treatment %": new_props,
        "Δ %": new_props - control_props,
        "Cum. Control %": control_cum,
        "Cum. Treatment %": new_cum,
        "Odds (Control)": control_odds,
        "Odds (Treatment)": new_odds,
        "Odds Ratio": calculated_or
    })
    
    # Style the dataframe
    def color_diff(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: green"
        elif val < 0:
            return "color: red"
        return ""
    
    styled_df = df.style.format({
        "Control %": "{:.2f}",
        "Treatment %": "{:.2f}",
        "Δ %": "{:+.2f}",
        "Cum. Control %": "{:.2f}",
        "Cum. Treatment %": "{:.2f}",
        "Odds (Control)": "{:.3f}",
        "Odds (Treatment)": "{:.3f}",
        "Odds Ratio": "{:.3f}"
    }, na_rep="—").map(color_diff, subset=["Δ %"])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    return df


def render_export_buttons(
    df: pd.DataFrame,
    chart_fig,
    filename_prefix: str = "gbs_analysis"
) -> None:
    """Render export buttons for data and charts.
    
    Args:
        df: DataFrame to export.
        chart_fig: Plotly figure to export.
        filename_prefix: Prefix for exported filenames.
    """
    st.subheader("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    # CSV Export
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"{filename_prefix}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Excel Export
    with col2:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        
        st.download_button(
            label="📊 Download Excel",
            data=buffer.getvalue(),
            file_name=f"{filename_prefix}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # Chart PNG Export
    with col3:
        if chart_fig is not None:
            img_bytes = chart_fig.to_image(format="png", scale=2)
            st.download_button(
                label="🖼️ Download Chart",
                data=img_bytes,
                file_name=f"{filename_prefix}_chart.png",
                mime="image/png",
                use_container_width=True
            )
