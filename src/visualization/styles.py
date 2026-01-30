"""Color schemes and styling for GBS visualizations."""

# GBS Disability Scale Colors (good outcomes green -> bad outcomes red)
DISABILITY_COLORS = [
    "#228B22",  # 0: ForestGreen (Healthy)
    "#90EE90",  # 1: LightGreen (Minor Symptoms)
    "#ADFF2F",  # 2: GreenYellow (Walk Independent)
    "#FFFF00",  # 3: Yellow (Walk Assisted)
    "#FFA500",  # 4: Orange (Bedridden)
    "#FF4500",  # 5: OrangeRed (Ventilated)
]

# Alternative color palette (colorblind-friendly)
DISABILITY_COLORS_ACCESSIBLE = [
    "#1a9850",  # 0: Dark green
    "#91cf60",  # 1: Light green
    "#d9ef8b",  # 2: Yellow-green
    "#fee08b",  # 3: Light orange
    "#fc8d59",  # 4: Orange
    "#d73027",  # 5: Red
]


def get_text_color(category_index: int) -> str:
    """Get appropriate text color for a given category.
    
    Returns white for dark backgrounds, black for light backgrounds.
    
    Args:
        category_index: Index of the disability category (0-5).
    
    Returns:
        Hex color string for text.
    """
    # Dark backgrounds (0, 5) need white text
    if category_index in [0, 5]:
        return "#FFFFFF"
    return "#000000"


# Chart layout constants
CHART_MARGINS = dict(l=20, r=150, t=60, b=40)
CHART_HEIGHT = 300
BAR_HEIGHT = 0.6

# Font settings
FONT_FAMILY = "Arial, sans-serif"
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 11
ANNOTATION_FONT_SIZE = 10
