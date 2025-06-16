from src.monotherapy import fit_monotherapy
from src.synergy import calculate_delta_scores, bootstrap_delta_scores
from src.visualization import create_synergy_heatmap_base
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('examples/example_data.csv')

# Calculate inhibition
data['inhibition'] = 1 - data['live_normalized']

# Fit monotherapy models
ecaii_params = fit_monotherapy(data[data['dose_X'] == 0], 'dose_E', 'inhibition')
xray_params = fit_monotherapy(data[data['dose_E'] == 0], 'dose_X', 'inhibition')

# Calculate synergy scores
results = calculate_delta_scores(data, ecaii_params, xray_params)

create_synergy_heatmap_base(results)
plt.savefig('./examples/synergy_heatmap.png')