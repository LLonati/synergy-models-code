# Synergy Models

A Python package for analyzing and visualizing drug synergy using the ZIP (Zero Interaction Potency) model with robust statistical analysis through bootstrapping.


## Key Features
**Monotherapy Model Fitting**: Fits dose-response curves for single-drug treatments using 2-parameter logistic models
**Synergy Quantification**: Calculates delta scores based on the ZIP model to quantify synergy/antagonism
**Potency Shift Analysis**: Quantifies how one drug modifies the potency of another
**Bootstrap Analysis**: Provides robust statistical confidence intervals and p-values
**Multi-cell Line Comparison**: Tools to compare synergy patterns across different cell lines
**Visualization**: tools for synergy analysis results

## Overview
This package provides tools for quantifying and visualizing synergistic or antagonistic effects in drug combination experiments, particularly focused on EcAII and X-ray therapy combinations. It implements robust statistical methods through bootstrap resampling to provide confidence intervals and statistical significance for synergy scores.

### Notes on δ-score
Note that a delta score has a unit of percentage inhibition (e.g., δ = 0.3 corresponds to 30% of response beyond expectation). Consequently, the delta scores are directly comparable within and between drug combinations.

### Implementation Notes
**Bootstrap Distribution**: parameters are sampled from multivariate normal distributions, which preserves the correlation structure between parameters (e.g., EC50 and Hill).
**Confidence Intervals**: 95% confidence intervals are calculated using the empirical quantiles from the bootstrap distribution rather than assuming normality.
**Statistical Significance**: p-values are calculated by counting the proportion of bootstrap samples that cross zero, which gives you a direct measure of statistical significance.


## Installation
```bash
# Clone the repository
git clone https://github.com/LLonati/synergy-models-code.git
cd synergy-models-code

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Basic usage

```python
from src.monotherapy import fit_monotherapy
from src.synergy import calculate_delta_scores, bootstrap_delta_scores
import pandas as pd

# Load data
data = pd.read_csv('examples/example_data.csv')

# Calculate inhibition
data['inhibition'] = 1 - data['live_normalized']

# Fit monotherapy models
ecaii_params = fit_monotherapy(data[data['dose_X'] == 0], 'dose_E', 'inhibition')
xray_params = fit_monotherapy(data[data['dose_E'] == 0], 'dose_X', 'inhibition')

# Calculate synergy scores
results = calculate_delta_scores(data, ecaii_params, xray_params, params_shifts)

bootstrap_results, bootstrap_raw_iter = bootstrap_delta_scores(data, params_drug1=params_ecaii, params_drug2=params_xray, drug_col1='dose_E', drug_col2='dose_X')
```

Basic analysis from command line.

```bash
python synergy_analysis.py --cell-lines A549 BT549 --bootstrap --iterations 1000
```

Generate Visualizations from Existing Results

```bash
python synergy_analysis.py --use-existing --fontsize 18 --linewidth 4 --markersize 8
```
Create Poster-Friendly Figures

```bash
python synergy_analysis.py --use-existing --fontsize 24 --linewidth 6 --markersize 12 --figsize 24,16
```
 
### Command-Line Arguments

- `--data-dir`: directory containing input data files (default: 'data/processed')
- `--output-dir`: directory to save results (default: 'results')
- `--cell-lines`: List of cell lines to analyze, separated by a single space. Each cell name must correspond to a .csv dataset named '{cell-name}_synergy_normalized.csv'. (default: 'A549 BT549 786')
- `--bootstrap`: Enable bootstrap analysis (default: True)
- `--iterations`: Number of bootstrap iterations (default: 1000)
- `--use-existing`: Use existing analysis results instead of recomputing
- `--fontsize`: Base font size for figures (default: 18)
- `--linewidth`: Line width for plots (default: 4)
- `--markersize`: Marker size for plots (default: 8)
- `--figsize`: Figure size as width height (e.g. "18, 12")


### Input Data Structure

Input data should be in csv format with following columns:
- `dose_E`: EcaII dose
- `dose_X`: X-ray dose
- `live_normalized` or `inhibition`: Cell viability or inhibition values
- `rep`: Replicate identifier

### Output Files

The package generates:

- Parameter files in `results/parameters/{cell_line}/`
- Figures in `results/figures/{cell_line}/`
- Bootstrap results in CSV format
- Summary logs in `real_data_test.log`

## Core Modules

- `src/monotherapy.py`: Functions for fitting dose-response curves
- `src/synergy.py`: Implementation of ZIP synergy models and bootstrap analysis for calculating delta scores
- `src/visualization.py`: Visualization functions for creating plots and heatmaps
- `synergy_analysis.py`: Main execution script with command-line interface

## Dependencies
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
tqdm

## Citation
If you use this package in your research, please cite:
Citation information will be available in the next release. Please check back soon.