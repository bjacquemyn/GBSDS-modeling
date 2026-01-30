"""
Forward Simulation Page - Approach A

Adjust the Odds Ratio to visualize how the outcome distribution shifts.
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ProportionalOddsModel
from src.visualization import GrottaChart
from src.components import render_metrics, render_comparison_table, render_export_buttons
from src.components.sidebar import load_settings


def main():
    settings = load_settings()
    labels = settings.get("disability_scale", {}).get("labels", [
        "0: Healthy", "1: Minor Symptoms", "2: Walk Independent",
        "3: Walk Assisted", "4: Bedridden", "5: Ventilated"
    ])
    sim_config = settings.get("simulation", {})
    
    st.header("🔬 Forward Simulation (Sensitivity Analysis)")
    st.markdown("""
    **Approach A**: Define an Odds Ratio and observe how the disability distribution shifts.
    
    Use this mode to explore different treatment effect sizes and understand 
    the potential impact on patient outcomes.
    """)
    
    # Check if baseline is available in session state
    if "baseline_props" not in st.session_state:
        st.warning("⚠️ Please configure baseline proportions on the **Home** page first.")
        st.stop()
    
    baseline_props = st.session_state.baseline_props
    
    # OR Slider
    st.subheader("Treatment Effect (Odds Ratio)")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        odds_ratio = st.slider(
            "Common Odds Ratio (OR)",
            min_value=sim_config.get("or_min", 0.5),
            max_value=sim_config.get("or_max", 5.0),
            value=sim_config.get("or_default", 1.5),
            step=sim_config.get("or_step", 0.1),
            help="OR > 1 shifts toward better outcomes; OR < 1 shifts toward worse outcomes"
        )
    with col2:
        st.markdown("")
        st.markdown("")
        if odds_ratio > 1:
            st.success(f"OR {odds_ratio:.2f} → Better outcomes")
        elif odds_ratio < 1:
            st.error(f"OR {odds_ratio:.2f} → Worse outcomes")
        else:
            st.info("OR 1.00 → No change")
    
    # Calculate shift
    model = ProportionalOddsModel(baseline_props)
    result = model.calculate_shift(odds_ratio)
    
    st.divider()
    
    # Metrics
    render_metrics(result, mode="forward")
    
    st.divider()
    
    # Grotta Chart
    st.subheader("📊 Outcome Distribution Comparison")
    chart = GrottaChart(labels)
    fig = chart.create_comparison(
        baseline_props,
        result.new_proportions,
        title=f"GBS Disability Outcomes: OR = {odds_ratio:.2f}"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Binary Endpoint Analysis
    st.subheader("🎯 Binary Endpoint: Ability to Walk Unaided")
    st.markdown("""
    This section shows how the common odds ratio translates to the binary endpoint 
    **"ability to walk unaided"** (GBS-DS levels 0-2 combined).
    """)
    
    # Calculate binary endpoint statistics
    control_walking = sum(baseline_props[:3])  # Levels 0, 1, 2
    treatment_walking = sum(result.new_proportions[:3])
    
    # Calculate odds for binary endpoint
    control_odds = control_walking / (100 - control_walking) if control_walking < 100 else float('inf')
    treatment_odds = treatment_walking / (100 - treatment_walking) if treatment_walking < 100 else float('inf')
    
    # Odds ratio for binary endpoint (should match the common OR)
    binary_or = treatment_odds / control_odds if control_odds > 0 else float('inf')
    
    # Additional clinically relevant metrics
    risk_diff = treatment_walking - control_walking
    relative_risk = treatment_walking / control_walking if control_walking > 0 else float('inf')
    nnt = 100 / risk_diff if risk_diff > 0 else float('inf')
    
    # Display in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Control Arm (IVIg)")
        st.metric("Walking Unaided (%)", f"{control_walking:.1f}%")
        st.metric("Odds", f"{control_odds:.3f}", 
                  help="P(walk) / P(not walk)")
    
    with col2:
        st.markdown("##### Treatment Arm")
        st.metric("Walking Unaided (%)", f"{treatment_walking:.1f}%", 
                  delta=f"{risk_diff:+.1f}%")
        st.metric("Odds", f"{treatment_odds:.3f}",
                  help="P(walk) / P(not walk)")
    
    # Summary table
    st.markdown("##### Summary Measures")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Odds Ratio", f"{binary_or:.3f}",
                           help="Should equal the common OR from the proportional odds model")
    summary_cols[1].metric("Risk Difference", f"{risk_diff:+.1f}%",
                           help="Absolute increase in walking rate")
    summary_cols[2].metric("Relative Risk", f"{relative_risk:.2f}",
                           help="Treatment rate / Control rate")
    summary_cols[3].metric("NNT", f"{nnt:.1f}" if nnt < 100 else "—",
                           help="Number Needed to Treat for one additional patient to walk")
    
    # Interpretation
    with st.expander("📖 Interpretation Guide"):
        st.markdown(f"""
        **Current Analysis:**
        - With OR = **{odds_ratio:.2f}**, patients in the investigational arm are 
          **{odds_ratio:.1f}x** more likely to achieve any given disability threshold or better.
        - The absolute increase in walking independence is **{result.risk_difference:+.1f}%**.
        
        **Clinical Context:**
        - **OR 1.5**: Moderate treatment effect (typical for IVIg superiority studies)
        - **OR 2.0**: Strong treatment effect (ambitious target)
        - **OR 3.0+**: Very strong effect (rare in practice)
        """)
    
    st.divider()
    
    # Comparison Table
    df = render_comparison_table(baseline_props, result.new_proportions, labels)
    
    st.divider()
    
    # Export
    render_export_buttons(df, fig, f"gbs_forward_or_{odds_ratio:.2f}")


if __name__ == "__main__":
    main()
else:
    main()
