"""Sidebar component for control arm configuration."""

import streamlit as st
import numpy as np
import yaml
from pathlib import Path
from typing import Optional


def load_presets(config_path: Optional[Path] = None) -> dict:
    """Load clinical presets from YAML configuration.
    
    Args:
        config_path: Path to presets.yaml. Defaults to config/presets.yaml.
    
    Returns:
        Dictionary of preset configurations.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "presets.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        # Return default preset if file not found
        return {
            "presets": {
                "severe_gbs": {
                    "name": "Severe GBS (Default)",
                    "proportions": [0.0, 4.0, 4.0, 16.0, 43.0, 33.0]
                }
            },
            "default_preset": "severe_gbs"
        }


def load_settings(config_path: Optional[Path] = None) -> dict:
    """Load application settings from YAML configuration.
    
    Args:
        config_path: Path to settings.yaml.
    
    Returns:
        Dictionary of settings.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Return defaults
        return {
            "disability_scale": {
                "labels": [
                    "0: Healthy",
                    "1: Minor Symptoms", 
                    "2: Walk Independent",
                    "3: Walk Assisted",
                    "4: Bedridden",
                    "5: Ventilated"
                ]
            }
        }


def render_control_sidebar() -> np.ndarray:
    """Render the control arm configuration sidebar.
    
    Returns:
        NumPy array of control arm proportions (percentages).
    """
    settings = load_settings()
    presets_config = load_presets()
    
    labels = settings.get("disability_scale", {}).get("labels", [
        "0: Healthy", "1: Minor Symptoms", "2: Walk Independent",
        "3: Walk Assisted", "4: Bedridden", "5: Ventilated"
    ])
    
    presets = presets_config.get("presets", {})
    default_preset_key = presets_config.get("default_preset", "severe_gbs")
    
    st.sidebar.header("📊 Control Arm Configuration")
    st.sidebar.markdown("Configure the IVIg (Control) arm distribution.")
    
    # Preset selector
    st.sidebar.subheader("Quick Presets")
    preset_names = {key: preset["name"] for key, preset in presets.items()}
    preset_options = ["Custom"] + list(preset_names.values())
    
    # Initialize session state for preset tracking
    if "current_preset" not in st.session_state:
        st.session_state.current_preset = preset_names.get(default_preset_key, "Custom")
    
    selected_preset_name = st.sidebar.selectbox(
        "Load Preset",
        options=preset_options,
        index=preset_options.index(st.session_state.current_preset) 
              if st.session_state.current_preset in preset_options else 0,
        key="preset_selector"
    )
    
    # Detect if preset selection changed
    preset_changed = (selected_preset_name != st.session_state.current_preset)
    st.session_state.current_preset = selected_preset_name
    
    # Get preset proportions if not custom
    if selected_preset_name != "Custom":
        # Find the preset key by name
        preset_key = next(
            (k for k, v in presets.items() if v["name"] == selected_preset_name),
            default_preset_key
        )
        preset_data = presets.get(preset_key, {})
        default_props = preset_data.get("proportions", [0, 4, 4, 16, 43, 33])
        
        # When preset changes, update the session state for each control widget
        # This is necessary because Streamlit caches widget values by key
        if preset_changed:
            for i, val in enumerate(default_props):
                st.session_state[f"control_{i}"] = float(val)
        
        # Show preset description
        if "description" in preset_data:
            st.sidebar.info(preset_data["description"])
    else:
        # Use session state or defaults for custom
        if "custom_control" in st.session_state:
            default_props = st.session_state.custom_control
        else:
            default_props = [0.0, 4.0, 4.0, 16.0, 43.0, 33.0]
    
    st.sidebar.divider()
    st.sidebar.subheader("Manual Adjustment")
    
    # Individual category inputs
    control_inputs = []
    for i, label in enumerate(labels):
        val = st.sidebar.number_input(
            f"{label} (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(default_props[i]),
            step=1.0,
            key=f"control_{i}"
        )
        control_inputs.append(val)
    
    # Store custom values
    st.session_state.custom_control = control_inputs
    
    # Validation
    total_pct = sum(control_inputs)
    
    st.sidebar.divider()
    if 99.9 <= total_pct <= 100.1:
        st.sidebar.success(f"✓ Total: {total_pct:.1f}%")
    else:
        st.sidebar.error(f"⚠ Total: {total_pct:.1f}% (must equal 100%)")
        st.stop()
    
    # Calculate and display walking rate
    walking_rate = sum(control_inputs[:3])
    st.sidebar.metric(
        "Control Walking Rate",
        f"{walking_rate:.1f}%",
        help="Percentage with GBS Score 0-2 (walking independent)"
    )
    
    return np.array(control_inputs)
