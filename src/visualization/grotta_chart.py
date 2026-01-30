"""
Interactive Grotta Bar Chart Visualization using Plotly.

Grotta charts display stacked horizontal bar charts comparing
ordinal outcome distributions between treatment groups.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional

from .styles import (
    DISABILITY_COLORS,
    get_text_color,
    CHART_MARGINS,
    CHART_HEIGHT,
    FONT_FAMILY,
    TITLE_FONT_SIZE,
    LABEL_FONT_SIZE,
    ANNOTATION_FONT_SIZE,
)


class GrottaChart:
    """Interactive Grotta bar chart for comparing treatment outcomes.
    
    Creates stacked horizontal bar charts showing the distribution
    of patients across disability categories for two treatment groups.
    """
    
    def __init__(
        self,
        labels: list[str],
        colors: Optional[list[str]] = None,
        control_name: str = "IVIg (Active Control)",
        treatment_name: str = "Investigational"
    ):
        """Initialize the Grotta chart.
        
        Args:
            labels: List of category labels (e.g., ["0: Healthy", ...]).
            colors: Optional custom color list. Uses default if not provided.
            control_name: Display name for control arm.
            treatment_name: Display name for treatment arm.
        """
        self.labels = labels
        self.colors = colors or DISABILITY_COLORS
        self.control_name = control_name
        self.treatment_name = treatment_name
    
    def create_comparison(
        self,
        control_props: np.ndarray,
        treatment_props: np.ndarray,
        title: str = "GBS Disability Outcome Distribution"
    ) -> go.Figure:
        """Create a comparison Grotta chart.
        
        Args:
            control_props: Control group proportions (percentages).
            treatment_props: Treatment group proportions (percentages).
            title: Chart title.
        
        Returns:
            Plotly Figure object.
        """
        groups = [self.control_name, self.treatment_name]
        data = np.vstack([control_props, treatment_props])
        
        fig = go.Figure()
        
        # Track cumulative positions for stacking
        cumulative = np.zeros(len(groups))
        
        # Add bars for each category
        for i, label in enumerate(self.labels):
            # Hover text with detailed info
            hover_text = [
                f"<b>{groups[j]}</b><br>"
                f"{label}: {data[j, i]:.1f}%<br>"
                f"Cumulative: {sum(data[j, :i+1]):.1f}%"
                for j in range(len(groups))
            ]
            
            fig.add_trace(go.Bar(
                y=groups,
                x=data[:, i],
                name=label,
                orientation='h',
                marker_color=self.colors[i],
                marker_line=dict(color='white', width=1),
                text=[f"{v:.1f}%" if v >= 4 else "" for v in data[:, i]],
                textposition='inside',
                textfont=dict(
                    color=[get_text_color(i)] * len(groups),
                    size=ANNOTATION_FONT_SIZE,
                    family=FONT_FAMILY
                ),
                hovertext=hover_text,
                hoverinfo='text',
            ))
        
        # Update layout with dark theme
        fig.update_layout(
            barmode='stack',
            title=dict(
                text=title,
                font=dict(size=TITLE_FONT_SIZE, family=FONT_FAMILY, color='white')
            ),
            xaxis=dict(
                title=dict(text="Percentage of Patients (%)", font=dict(color='white')),
                range=[0, 100],
                ticksuffix="%",
                tickfont=dict(color='white'),
                gridcolor='rgba(255, 255, 255, 0.2)',
                gridwidth=1,
            ),
            yaxis=dict(
                title="",
                tickfont=dict(size=LABEL_FONT_SIZE, color='white')
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
            margin=CHART_MARGINS,
            height=CHART_HEIGHT,
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#16213e',
            hovermode='closest',
        )
        
        return fig
    
    def create_single(
        self,
        proportions: np.ndarray,
        group_name: str,
        title: str = "GBS Disability Distribution"
    ) -> go.Figure:
        """Create a single-group Grotta chart.
        
        Args:
            proportions: Group proportions (percentages).
            group_name: Display name for the group.
            title: Chart title.
        
        Returns:
            Plotly Figure object.
        """
        fig = go.Figure()
        
        cumulative = 0
        for i, label in enumerate(self.labels):
            fig.add_trace(go.Bar(
                y=[group_name],
                x=[proportions[i]],
                name=label,
                orientation='h',
                marker_color=self.colors[i],
                marker_line=dict(color='white', width=1),
                text=[f"{proportions[i]:.1f}%" if proportions[i] >= 4 else ""],
                textposition='inside',
                textfont=dict(
                    color=[get_text_color(i)],
                    size=ANNOTATION_FONT_SIZE,
                    family=FONT_FAMILY
                ),
                hovertemplate=f"{label}: %{{x:.1f}}%<extra></extra>",
            ))
            cumulative += proportions[i]
        
        # Update layout with dark theme
        fig.update_layout(
            barmode='stack',
            title=dict(text=title, font=dict(size=TITLE_FONT_SIZE, color='white')),
            xaxis=dict(
                title=dict(text="Percentage of Patients (%)", font=dict(color='white')),
                range=[0, 100],
                ticksuffix="%",
                tickfont=dict(color='white'),
                gridcolor='rgba(255, 255, 255, 0.2)',
            ),
            yaxis=dict(title="", tickfont=dict(color='white')),
            legend=dict(
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='left',
                x=1.02,
                font=dict(color='white'),
                bgcolor='rgba(0, 0, 0, 0.6)'
            ),
            margin=CHART_MARGINS,
            height=200,
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#16213e',
        )
        
        return fig
