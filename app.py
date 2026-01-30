"""
GBS Clinical Trial Outcome Modeler

Main application entry point. This serves as the home page and
initializes the control arm configuration for other pages.
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ProportionalOddsModel
from src.visualization import GrottaChart
from src.components.sidebar import render_control_sidebar, load_settings


def main():
    # Page configuration
    st.set_page_config(
        layout="wide",
        page_title="GBS Clinical Trial Modeler",
        page_icon="🧬",
        initial_sidebar_state="expanded"
    )
    
    # Load settings
    settings = load_settings()
    app_config = settings.get("app", {})
    
    # Title and description
    st.title("🧬 " + app_config.get("title", "GBS Clinical Trial Outcome Modeler"))
    st.markdown(app_config.get("subtitle", "Dynamic modeling using the Proportional Odds Model"))
    
    # Sidebar: Control arm configuration
    control_props = render_control_sidebar()
    
    # Store in session state for other pages
    st.session_state.control_props = control_props
    
    # Main content
    st.header("📋 Overview")
    
    st.markdown("""
    This application models treatment effects in **Guillain-Barré Syndrome (GBS)** 
    clinical trials using the **Proportional Odds Model**.
    
    ### How to Use This Tool
    
    1. **Configure Control Arm** (sidebar): Set the expected disability distribution 
       in the IVIg control arm
    2. **Choose Your Approach**:
       - **Forward Simulation**: Explore different odds ratios to see outcome shifts
       - **Goal Seeking**: Define a target improvement and find the required efficacy
    """)
    
    # Quick preview of current control arm
    st.subheader("Current Control Arm Configuration")
    
    labels = settings.get("disability_scale", {}).get("labels", [
        "0: Healthy", "1: Minor Symptoms", "2: Walk Independent",
        "3: Walk Assisted", "4: Bedridden", "5: Ventilated"
    ])
    
    # Show control arm distribution
    chart = GrottaChart(labels)
    fig = chart.create_single(
        control_props,
        "IVIg (Control Arm)",
        "Control Arm Disability Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    model = ProportionalOddsModel(control_props)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Walking Independent", f"{model.control_walking_rate:.1f}%", help="Score 0-2")
    col2.metric("Walk Assisted", f"{control_props[3]:.1f}%", help="Score 3")
    col3.metric("Bedridden", f"{control_props[4]:.1f}%", help="Score 4")
    col4.metric("Ventilated", f"{control_props[5]:.1f}%", help="Score 5")
    
    st.divider()
    
    # Navigation hint
    st.info("""
    👈 **Next Step**: Use the **sidebar navigation** to select either:
    - **Forward Simulation** - Explore "what if" scenarios with different ORs
    - **Goal Seeking** - Calculate required efficacy for a target outcome
    """)
    
    # Methodology section
    with st.expander("📚 Methodology: Proportional Odds Model"):
        st.markdown("""
        ### The Proportional Odds Assumption
        
        The proportional odds model assumes that a single **common odds ratio (OR)** 
        applies across all cumulative probability thresholds of an ordinal outcome.
        
        For a GBS disability scale with categories 0-5:
        
        - OR applies to P(Y ≤ 0), P(Y ≤ 1), P(Y ≤ 2), etc.
        - An OR > 1 means patients are more likely to achieve better outcomes
        - The assumption is that the treatment effect is uniform across the scale
        
        ### Mathematical Formulation
        
        For each cumulative probability threshold k:
        
        ```
        odds_treatment(k) = odds_control(k) × OR
        ```
        
        Where:
        - `odds = P(Y ≤ k) / P(Y > k)`
        
        ### Limitations
        
        - Assumes treatment effect is consistent across all disability levels
        - May not capture differential effects at specific thresholds
        - Should be validated against clinical trial data when available
        
        ### References
        
        - Agresti, A. (2010). Analysis of Ordinal Categorical Data
        - Hughes RAC, et al. Immunotherapy for GBS: Cochrane Review
        """)


if __name__ == "__main__":
    main()
