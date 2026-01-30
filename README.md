# GBS Clinical Trial Outcome Modeler

A dynamic Guillain-Barré Syndrome clinical trial modeling application using the Proportional Odds Model.

## Features

- **Forward Simulation**: Adjust the odds ratio to visualize outcome shifts
- **Goal Seeking**: Set a walking recovery target to calculate required efficacy
- **Interactive Grotta Charts**: Visualize disability distributions with Plotly
- **Detailed Comparison Tables**: Export results for analysis
- **Clinical Presets**: Pre-configured scenarios for common GBS presentations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Project Structure

```
GBSDS modeling/
├── app.py                        # Main entry point
├── config/
│   ├── settings.yaml             # App configuration
│   └── presets.yaml              # Clinical scenario presets
├── src/
│   ├── models/                   # Core statistical logic
│   ├── visualization/            # Grotta charts and styling
│   └── components/               # Reusable UI components
└── pages/                        # Streamlit multi-page navigation
```

## Methodology

The application uses the **Proportional Odds Model** to simulate treatment effects:

- **Odds Ratio (OR)**: A common OR is applied across all cumulative probabilities
- **Walking Independence**: Defined as GBS Disability Score ≤ 2
- **Active Control**: IVIg (standard of care) as the baseline comparator

## References

- Hughes RAC, et al. Immunotherapy for Guillain-Barré syndrome. Cochrane Database Syst Rev. 2014.
