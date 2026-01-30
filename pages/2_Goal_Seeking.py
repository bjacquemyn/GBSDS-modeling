"""
Goal Seeking Page - Approach B

Set a target walking recovery improvement and calculate the required efficacy (OR).
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
    goal_config = settings.get("goal_seeking", {})
    
    st.header("🎯 Goal Seeking (Trial Powering)")
    st.markdown("""
    **Approach B**: Define a target clinical benefit and calculate the required treatment efficacy.
    
    Use this mode when designing a trial to determine what effect size (OR) 
    you need to detect based on clinically meaningful endpoints.
    """)
    
    # Check if baseline is available in session state
    if "baseline_props" not in st.session_state:
        st.warning("⚠️ Please configure baseline proportions on the **Home** page first.")
        st.stop()
    
    baseline_props = st.session_state.baseline_props
    
    # Create model
    model = ProportionalOddsModel(baseline_props)
    
    # Show current baseline
    st.info(f"""
    **Current Baseline (IVIg Control)**  
    Walking Independent Rate (Score 0-2): **{model.baseline_walking_rate:.1f}%**
    """)
    
    # Target input
    st.subheader("Define Clinical Target")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        target_risk_diff = st.slider(
            "Target Absolute Increase in Walking (%)",
            min_value=goal_config.get("target_min", 1.0),
            max_value=goal_config.get("target_max", 50.0),
            value=goal_config.get("target_default", 20.0),
            step=goal_config.get("target_step", 0.5),
            help="The absolute percentage point increase in patients achieving walking independence"
        )
    
    with col2:
        projected_rate = model.baseline_walking_rate + target_risk_diff
        st.metric(
            "Projected Walking Rate",
            f"{projected_rate:.1f}%",
            delta=f"+{target_risk_diff:.1f}%"
        )
    
    # Calculate required OR
    required_or = model.find_or_for_target(target_risk_diff / 100.0)
    result = model.calculate_shift(required_or)
    
    st.divider()
    
    # Key finding
    st.subheader("📌 Required Treatment Effect")
    
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Target Improvement",
        f"+{target_risk_diff:.1f}%",
        help="Desired increase in walking independence"
    )
    col2.metric(
        "Required Odds Ratio",
        f"{required_or:.3f}",
        help="The OR your treatment must achieve"
    )
    col3.metric(
        "Achieved Walking Rate",
        f"{result.new_walking:.1f}%",
        delta=f"{result.risk_difference:+.1f}%"
    )
    
    # Feasibility assessment
    if required_or < 1.3:
        st.success("✅ **Highly Achievable**: This target requires a modest treatment effect.")
    elif required_or < 2.0:
        st.info("ℹ️ **Moderate Challenge**: This target is realistic for an effective treatment.")
    elif required_or < 3.0:
        st.warning("⚠️ **Ambitious Target**: This requires a strong treatment effect.")
    else:
        st.error("🚨 **Very Difficult**: This target may be unrealistic. Consider revising assumptions.")
    
    st.divider()
    
    # Grotta Chart
    st.subheader("📊 Projected Outcome Distribution")
    chart = GrottaChart(labels)
    fig = chart.create_comparison(
        baseline_props,
        result.new_proportions,
        title=f"Goal: +{target_risk_diff:.0f}% Walking (Required OR = {required_or:.2f})"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Binary Endpoint Analysis
    st.subheader("🎯 Binary Endpoint: Ability to Walk Unaided")
    st.markdown("""
    This section shows how the required odds ratio translates to the binary endpoint 
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
    
    # Power analysis context
    with st.expander("📖 Trial Design Implications"):
        st.markdown(f"""
        **For Your Target of +{target_risk_diff:.1f}% Walking Independence:**
        
        | Parameter | Value |
        |-----------|-------|
        | Required OR | {required_or:.3f} |
        | Baseline Walking | {model.baseline_walking_rate:.1f}% |
        | Expected in Treatment Arm | {result.new_walking:.1f}% |
        | Absolute Risk Difference | {result.risk_difference:.1f}% |
        | Number Needed to Treat (NNT) | {100/result.risk_difference:.1f} |
        
        **Sample Size Considerations:**
        - Larger effect sizes (higher OR) require smaller sample sizes
        - Consider your confidence in the baseline assumptions
        - Factor in expected dropout rates
        """)
    
    st.divider()
    
    # Comparison Table
    df = render_comparison_table(baseline_props, result.new_proportions, labels)
    
    st.divider()
    
    # Export
    render_export_buttons(df, fig, f"gbs_goal_{target_risk_diff:.0f}pct")


if __name__ == "__main__":
    main()
else:
    main()
