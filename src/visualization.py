import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
from src.monotherapy import logistic_4PL
from src.synergy import potency_shift_model
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('synergy_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _vectorized_4PL(doses, ec50, hill):
    """Vectorized 4PL calculation for efficiency."""
    doses = np.asarray(doses)
    return np.array([logistic_4PL(d, ec50, hill) for d in doses])


def _create_dose_range(dose_min, dose_max, n_points=100, use_log=True):
    """Create dose range with appropriate scaling."""
    if use_log and dose_max / dose_min > 10:
        return np.logspace(np.log10(max(dose_min, 1e-3)), np.log10(dose_max), n_points)
    return np.linspace(dose_min, dose_max, n_points)


def _format_axis(ax, xlabel, ylabel, title=None, fontsize=12, grid_alpha=0.3):
    """Apply common axis formatting."""
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize + 2)
    ax.grid(True, alpha=grid_alpha)
    ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)


def _add_text_box(ax, text, x=0.05, y=0.95, fontsize=12, va='top', ha='left'):
    """Add text annotation box to axis."""
    ax.text(x, y, text, transform=ax.transAxes, verticalalignment=va,
            horizontalalignment=ha, fontsize=fontsize,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    

def _plot_monotherapy_for_cell_line(ax, results, dose_col, other_dose_col, params_key, 
                                     params_dict, color_dict, cell_line,
                                     marker='o', markersize=6, linewidth=2):
    """Helper to plot monotherapy data for one drug across cell lines."""
    mono_data = results[results[other_dose_col] == 0]
    grouped = mono_data.groupby(dose_col)['inhibition'].agg(['mean', 'sem']).reset_index()
    
    ax.errorbar(
        grouped[dose_col], grouped['mean'], yerr=grouped['sem'],
        fmt=marker, markersize=markersize, linewidth=linewidth,
        elinewidth=linewidth, capsize=markersize / 2,
        label=cell_line, color=color_dict[cell_line], alpha=0.8
    )
    
    # Plot fitted curve if params available
    if cell_line in params_dict and params_key in params_dict[cell_line]:
        params = params_dict[cell_line][params_key]
        dose_min, dose_max = mono_data[dose_col].min(), mono_data[dose_col].max()
        use_log = 'ecaii' in params_key.lower()
        dose_range = _create_dose_range(dose_min, dose_max, use_log=use_log)
        y_fit = _vectorized_4PL(dose_range, params['EC50'], params['Hill'])
        ax.plot(dose_range, y_fit, '--', color=color_dict[cell_line], linewidth=linewidth)


def _build_param_text(params_dict, cell_lines, params_key, unit):
    """Build parameter annotation text for multiple cell lines."""
    lines = []
    for cell_line in cell_lines:
        if cell_line in params_dict and params_key in params_dict[cell_line]:
            p = params_dict[cell_line][params_key]
            lines.append(f"{cell_line}: EC50={p['EC50']:.2f}±{p['errors'][0]:.2f} {unit}, "
                        f"Hill={p['Hill']:.2f}±{p['errors'][1]:.2f}")
    return '\n'.join(lines)


def _pivot_and_sort(data, value_col, ascending=False):
    """Pivot data and sort index."""
    pivot = data.pivot_table(values=value_col, index='dose_E', columns='dose_X')
    return pivot.sort_index(ascending=ascending)

# =============================================================================
# MAIN PLOTTING FUNCTIONS
# =============================================================================

def plot_monotherapy_curve(data, drug_col, effect_col, params, style=None, title=None, x_logscale=True):
    """Plot monotherapy dose-response data with fitted curve."""
    # Validate that params contains required keys
    required_keys = ['EC50', 'Hill', 'errors', 'r_squared']
    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        raise ValueError(f"Missing required keys in params: {', '.join(missing_keys)}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if style is None or style == 'bar':
        sns.lineplot(data=data, x=drug_col, y=effect_col, label='Experimental Data',
                     marker='o', alpha=0.7, linestyle='none', err_style="bars",
                     errorbar=("se", 2), err_kws={'capsize': 5}, ax=ax)
    elif style == 'scatter':
        for rep in data['rep'].unique():
            rep_data = data[data['rep'] == rep]
            ax.scatter(rep_data[drug_col], rep_data[effect_col], label=f'Rep {rep}', alpha=0.7)

    # Generate smooth curve from fitted parameters
    x_range = _create_dose_range(data[drug_col].min(), data[drug_col].max(), use_log=True)
    y_fit = _vectorized_4PL(x_range, params['EC50'], params['Hill'])
    ax.plot(x_range, y_fit, 'r-', linewidth=2)

    # Add parameter annotations
    param_text = (f"EC50 = {params['EC50']:.3f} ± {params['errors'][0]:.3f}\n"
                  f"Hill = {params['Hill']:.3f} ± {params['errors'][1]:.3f}\n"
                  f"R² = {params['r_squared']:.3f}")
    _add_text_box(ax, param_text)

    if x_logscale:
        ax.set_xscale('log')
    _format_axis(ax, 'Dose', 'Inhibition Fraction', title or 'Dose-Response Curve')
    
    return fig


def plot_cell_line_monotherapy_comparison(results_dict, params_dict, cell_lines,
                                          fontsize=12, linewidth=2, markersize=6, figsize=(18, 8)):
    """
    Create a figure comparing monotherapy responses across cell lines.
    
    Parameters:
    results_dict: dict, mapping cell lines to their results DataFrames
    params_dict: dict, mapping cell lines to their parameter dictionaries
    cell_lines: list, cell lines to include
    fontsize, linewidth, markersize: parameters for plot styling
    figsize: tuple, size of the figure
    
    Returns:
    matplotlib.figure.Figure: The created figure with monotherapy comparisons
    """
    # Create color mapping for consistent colors
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(cell_lines)))
    color_dict = {cell: color for cell, color in zip(cell_lines, colors)}

    # Create figure with two subplots for EcAII and X-ray
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Configuration for both drugs
    drug_configs = [
        {'ax': ax1, 
         'dose_col': 'dose_E',
         'other_col': 'dose_X',
         'params_key': 'params_ecaii',
         'marker': 'o', 
         'title': 'EcAII Monotherapy Response by Cell Line',
         'xlabel': 'EcAII concentration (U/ml)',
         'unit': 'U/ml',
         'log_scale': True},
        {'ax': ax2,
         'dose_col': 'dose_X',
         'other_col': 'dose_E',
         'params_key': 'params_xray',
         'marker': 's',
         'title': 'X-ray Monotherapy Response by Cell Line',
         'xlabel': 'X-ray dose (Gy)',
         'unit': 'Gy',
         'log_scale': False},
    ]
    
    for config in drug_configs:
        ax = config['ax']
        for cell_line in cell_lines:
            if cell_line not in results_dict:
                continue
            _plot_monotherapy_for_cell_line(
                ax, results_dict[cell_line], config['dose_col'], config['other_col'],
                config['params_key'], params_dict, color_dict, cell_line,
                marker=config['marker'], markersize=markersize, linewidth=linewidth
            )
        
        if config['log_scale']:
            ax.set_xscale('log')
        
        _format_axis(ax, config['xlabel'], 'Inhibition Fraction', config['title'], fontsize)
        ax.legend(fontsize=fontsize - 2)
        
        param_text = _build_param_text(params_dict, cell_lines, config['params_key'], config['unit'])
        _add_text_box(ax, param_text, y=0.9, fontsize=fontsize - 1)
        ax.tick_params(width=2, length=6)
    
    fig.suptitle('Comparison of Monotherapy Response Across Cell Lines',
                 fontsize=fontsize + 4, y=0.98, fontweight='bold')
    plt.tight_layout()
    
    return fig


def create_synergy_heatmap_base(results, value_to_plot='delta_score', cell_line=None,
                                cmap='RdBu', vmin=-30, vmax=30, ax=None, annot=True,
                                fontsize=12, mask_zeros=True, figsize=(10, 8)):
    """
    Plot heatmap of delta scores.
    
    Parameters:
    results: DataFrame with dose-response data and delta scores
    value_to_plot: str, column name in results to use for heatmap values (default: 'delta_score')
    cell_line: str, name of cell line for title
    cmap: str, colormap to use (default: 'RdBu')
    vmin, vmax: float, minimum and maximum values for colormap
    ax: matplotlib axis, optional axis to plot on
    annot: bool, whether to annotate heatmap cells with delta scores
    fontsize: int, font size for annotations
    figsize: tuple, size of the figure if ax is not provided
    mask_zeros: bool, whether to mask zero delta scores for monotherapy points in the heatmap

    Returns:
    fig: matplotlib figure (or None if ax is provided)
    ax: matplotlib axis with the plot
    heatmap_data: pivot table with the data used for the heatmap
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
       
    # Pivot data to create dose matrix
    heatmap_data = results.pivot_table(
        values=value_to_plot, 
        index='dose_E', 
        columns='dose_X'
    )
    # Sort index in reverse order to match contour plot orientation
    heatmap_data = _pivot_and_sort(results, value_to_plot)
    # Convert to percentage for display (multiply by 100)
    display_data = heatmap_data * 100
    # Create a mask for monotherapy points if requested
    mask = (heatmap_data == 0) if mask_zeros else None
    
    # Plot heatmap
    sns.heatmap(
        display_data, cmap=cmap, vmin=vmin, vmax=vmax, center=0,
        annot=annot, fmt='.1f', annot_kws={'size': fontsize},
        cbar_kws={'label': 'Delta Score % (Antagonism/Synergy)'},
        ax=ax, mask=mask
    )
    
    # Set titles and labels
    title = f'{cell_line or ""} Synergy Delta Scores (ZIP Model)'
    _format_axis(ax, 'X-ray dose (Gy)', 'EcAII concentration (U/ml)', title, fontsize)

    # Set colorbar properties after creating the heatmap
    cbar = ax.collections[0].colorbar  # Get the colorbar
    cbar.ax.tick_params(labelsize=fontsize - 2)  # Set tick label font size
    cbar.set_label('Delta Score % (Antagonism/Synergy)', fontsize=fontsize - 2)  # Set label font size

    return fig, ax, heatmap_data


def annotate_heatmap_with_ci(ax, bootstrap_results, significance_levels=None, fontsize_delta=10,
                             fontsize_ci=8, clear_existing=False, cell_line=None):
    """
    Add confidence interval and significance annotations to a synergy heatmap.
    
    Parameters:
    ax: matplotlib axis with the heatmap
    bootstrap_results: DataFrame with bootstrap results including CIs and p-values
    significance_levels: list of floats, p-value thresholds for significance markers
    fontsize_delta: int, font size for delta score values
    fontsize_ci: int, font size for confidence intervals

    Returns:
    ax: matplotlib axis with annotations added
"""
    if significance_levels is None:
        significance_levels = [0.05, 0.01, 0.005]

    significance_levels = sorted(significance_levels, reverse=True)

    # Clear existing annotations if any
    if clear_existing:
        for txt in ax.texts:
            txt.remove()

    if cell_line and ax.get_title() == '':
        ax.set_title(f'{cell_line} Synergy Delta Scores with 95% CI (ZIP Model)')
    elif ax.get_title() == '':
        ax.set_title('Synergy Delta Scores with 95% CI (ZIP Model)')

    # Pivot the bootstrap results data
    delta_pivot = bootstrap_results.pivot_table(values='delta_score_mean', index='dose_E', columns='dose_X')
    lower_pivot = bootstrap_results.pivot_table(values='delta_score_lower', index='dose_E', columns='dose_X')
    upper_pivot = bootstrap_results.pivot_table(values='delta_score_upper', index='dose_E', columns='dose_X')
    pval_pivot = bootstrap_results.pivot_table(values='p_adjusted', index='dose_E', columns='dose_X')
 

    # Sort indices in reverse order to match contour plot orientation
    pivots = {col: _pivot_and_sort(bootstrap_results, col) 
              for col in ['delta_score_mean', 'delta_score_lower', 'delta_score_upper', 'p_adjusted']}
    # Get the row and column indices and values
    rows, cols = pivots['delta_score_mean'].index, pivots['delta_score_mean'].columns
    
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            delta = pivots['delta_score_mean'].iloc[i, j]
            if pd.isna(delta):
                continue
                
            lower = pivots['delta_score_lower'].iloc[i, j] * 100
            upper = pivots['delta_score_upper'].iloc[i, j] * 100
            pval = pivots['p_adjusted'].iloc[i, j]
            
            ax.text(j + 0.5, i + 0.7, f"({lower:.1f}, {upper:.1f})",
                    ha='center', va='center', fontsize=fontsize_ci, color='black')
            
            asterisk_count = sum(1 for threshold in significance_levels if pval < threshold)
            if asterisk_count > 0:
                ax.text(j + 0.85, i + 0.15, '*' * asterisk_count,
                        color='black', fontsize=fontsize_delta + 2, fontweight='bold',
                        ha='center', va='center')
    return ax


def plot_synergy_heatmap_comparison(results_dict, bootstrap_dict=None, cell_lines=None,
                                    fontsize=12, figsize=(18, 8)):
    """
    Create a figure comparing synergy heatmaps across cell lines.
    
    Parameters:
    results_dict: dict, mapping cell lines to their results DataFrames
    bootstrap_dict: dict, mapping cell lines to their bootstrap results (optional)
    cell_lines: list, cell lines to include
    fontsize: int, font size for plot text
    figsize: tuple, size of the figure
    
    Returns:
    matplotlib.figure.Figure: The created figure with synergy heatmap comparisons
    """
    if cell_lines is None:
        cell_lines = list(results_dict.keys())
        
    # Create figure for delta score comparison
    fig = plt.figure(figsize=figsize)
    n_cell_lines = len([c for c in cell_lines if c in results_dict])
    n_cols = min(3, n_cell_lines)
    n_rows = (n_cell_lines + n_cols - 1) // n_cols

    plot_idx = 0
    for cell_line in cell_lines:
        if cell_line not in results_dict:
            continue

        plot_idx += 1
        ax = fig.add_subplot(n_rows, n_cols, plot_idx)            
        results = results_dict[cell_line]
        combo_data = results[(results['dose_E'] > 0) & (results['dose_X'] > 0)]
        
        if combo_data.empty:
            ax.text(0.5, 0.5, f"No combination data for {cell_line}",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(cell_line, fontsize=fontsize + 1, fontweight='bold')
            continue

        _, ax, _ = create_synergy_heatmap_base(
            combo_data, value_to_plot='delta_score', cell_line=cell_line,
            ax=ax, annot=True, fontsize=fontsize - 2
        )
        if bootstrap_dict and cell_line in bootstrap_dict:
            try:
                annotate_heatmap_with_ci(ax, bootstrap_dict[cell_line],
                                         significance_levels=[0.05, 0.01],
                                         fontsize_delta=fontsize - 2, fontsize_ci=fontsize - 4,
                                         cell_line=cell_line)
            except Exception as e:
                logger.error(f"Error annotating heatmap for {cell_line}: {e}")
        
        ax.set_title(cell_line, fontsize=fontsize + 2, fontweight='bold')
    
    fig.suptitle('Comparison of Synergy Patterns Across Cell Lines', fontsize=fontsize + 4, y=0.98)
    plt.tight_layout()
    return fig



def plot_detailed_bootstrap_results(bootstrap_results, original_deltas, cell_line=None,
                                    bootstrap_raw_iter=None, cmap='RdBu', vmin=-30, vmax=30, fontsize=12,
                                    figsize=(16, 12)):
    """
    Create detailed visualization of bootstrap results including confidence intervals.
    
    Parameters:
    bootstrap_results: DataFrame with bootstrap analysis results
    original_deltas: DataFrame with original delta scores
    cell_line: str, name of cell line for title
    bootstrap_raw_iter: array with raw bootstrap iterations (optional)
    cmap: str, colormap for heatmap
    vmin, vmax: float, color scale limits
    fontsize: int, font size for plot titles and labels

    Returns:
    fig: matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2)
    
    # Panel 1. Heatmap with significance markers (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create base heatmap
    _, ax1, _ = create_synergy_heatmap_base(
        bootstrap_results, value_to_plot='delta_score_mean', cell_line=cell_line,
        cmap=cmap, vmin=vmin, vmax=vmax, ax=ax1, annot=True, fontsize=fontsize
    )
    annotate_heatmap_with_ci(ax1, bootstrap_results, significance_levels=[0.05, 0.01, 0.005])

    # Add confidence interval annotations
    ax1 = annotate_heatmap_with_ci(ax1, bootstrap_results, clear_existing=False, cell_line=cell_line, significance_levels= [0.05, 0.01, 0.005])
    ax1.set_title('Synergy Delta Scores with Significance', fontsize=fontsize+2)

    # Panel 2. Distribution plot for selected dose combinations (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_bootstrap_distributions(ax2, bootstrap_results, fontsize)

    # Panel 3. Diagnostic plot of bootstrap iterations (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_convergence_diagnostic(ax3, bootstrap_raw_iter, fontsize)

    # Panel 4. Actual vs bootstrap comparison
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_actual_vs_bootstrap(ax4, bootstrap_results, original_deltas, fontsize)
    
    fig.suptitle(f'{cell_line or ""} Bootstrap Analysis (95% CI)', fontsize=fontsize + 4)
    plt.tight_layout()

    return fig


def _plot_bootstrap_distributions(ax, bootstrap_results, fontsize):
    """Helper for bootstrap distribution subplot."""
    top_synergy = bootstrap_results.nlargest(5, 'delta_score_mean')
    top_antagonism = bootstrap_results.nsmallest(5, 'delta_score_mean')
    selected_combos = pd.concat([top_synergy, top_antagonism])
    
    for idx, (_, combo) in enumerate(selected_combos.iterrows()):
        mean, lower, upper = combo['delta_score_mean'], combo['delta_score_lower'], combo['delta_score_upper']
        std = (upper - lower) / (2 * 1.96)
        x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
        y = stats.norm.pdf(x, mean, std)
        y = y / np.max(y) * 0.8
        
        ax.plot(x, y + idx)
        ax.plot([lower, upper], [idx, idx], 'k-', linewidth=2)
        ax.plot(mean, idx, 'ko', markersize=6)
    
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax.set_yticks(range(len(selected_combos)))
    ax.set_yticklabels([f"E:{r['dose_E']}, X:{r['dose_X']}" for _, r in selected_combos.iterrows()])
    _format_axis(ax, 'Delta Score Distribution', '', 'Bootstrap Distributions for Key Dose Combinations', fontsize)


def _plot_convergence_diagnostic(ax, bootstrap_raw_iter, fontsize):
    """Helper for convergence diagnostic subplot."""
    if bootstrap_raw_iter is None:
        ax.text(0.5, 0.5, "Bootstrap iteration data not available", ha='center', va='center', fontsize=fontsize)
        return
    
    global_deltas = np.mean(bootstrap_raw_iter, axis=0)
    n_iters = len(global_deltas)
    
    ax3b = ax.twinx()
    running_avg = np.cumsum(global_deltas) / np.arange(1, n_iters + 1)
    running_std = np.array([np.std(global_deltas[:i + 1]) for i in range(n_iters)])
    running_sem = running_std / np.sqrt(np.arange(1, n_iters + 1))
    
    ax.plot(running_avg, 'b-', linewidth=2, label='Running Average')
    ax.fill_between(range(n_iters), running_avg - 1.96 * running_sem,
                    running_avg + 1.96 * running_sem, color='blue', alpha=0.2, label='95% CI')
    ax.plot(global_deltas, 'o', alpha=0.1, markersize=2, color='gray')
    
    final_mean = np.mean(global_deltas)
    ci_lower, ci_upper = np.percentile(global_deltas, [2.5, 97.5])
    ax.axhline(y=final_mean, color='red', linestyle='-', label='Mean')
    ax.axhspan(ci_lower, ci_upper, color='red', alpha=0.1, label='95% CI')
    
    pct_change = np.abs(np.diff(running_avg) / running_avg[:-1] * 100)
    ax3b.plot(range(1, n_iters), pct_change, 'g-', alpha=0.5, label='% Change')
    ax3b.axhline(y=0.1, color='green', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Bootstrap Iteration', fontsize=fontsize)
    ax.set_ylabel('Global Delta Score', fontsize=fontsize)
    ax3b.set_ylabel('Percent Change (%)', fontsize=fontsize - 1, color='green')
    ax.set_title('Bootstrap Convergence Diagnostic', fontsize=fontsize + 2)
    ax.legend(loc='upper right', fontsize=fontsize - 2)
    ax.grid(True, alpha=0.3)


def _plot_actual_vs_bootstrap(ax, bootstrap_results, original_deltas, fontsize):
    """Helper for actual vs bootstrap comparison subplot."""
    combo_deltas = original_deltas[(original_deltas['dose_E'] > 0) & (original_deltas['dose_X'] > 0)]
    merged = pd.merge(bootstrap_results, combo_deltas.drop_duplicates(subset=['dose_E', 'dose_X']),
                      on=['dose_E', 'dose_X'], suffixes=('_bootstrap', ''))
    
    ax.scatter(merged['delta_score'], merged['delta_score_mean'], alpha=0.7, s=50,
               c=np.abs(merged['delta_score'] - merged['delta_score_mean']), cmap='viridis')
    
    ax.errorbar(merged['delta_score'], merged['delta_score_mean'],
                yerr=[merged['delta_score_mean'] - merged['delta_score_lower'],
                      merged['delta_score_upper'] - merged['delta_score_mean']],
                fmt='none', ecolor='gray', alpha=0.3)
    
    vals = np.concatenate([merged['delta_score'], merged['delta_score_mean']])
    margin = (vals.max() - vals.min()) * 0.1
    ax.plot([vals.min() - margin, vals.max() + margin], 
            [vals.min() - margin, vals.max() + margin], 'k--', alpha=0.7)
    
    corr = np.corrcoef(merged['delta_score'], merged['delta_score_mean'])[0, 1]
    _add_text_box(ax, f"Correlation: {corr:.3f}")
    _format_axis(ax, 'Actual Delta Score', 'Bootstrap Mean Delta Score',
                 'Actual vs. Bootstrap Delta Scores', fontsize)


def panel_synergy_heatmap(results, params_drug1, params_drug2, params_shifts,
                          bootstrap_results=None, cell_line=None, grid_density=100,
                          cmap='RdBu', vmin=-30, vmax=30, fontsize=12):
    """
    Create a 4-panels figure for synergy visualization.
    
    Parameters:
    results: DataFrame with experimental data and calculated delta scores
    params_drug1: dict, fitted parameters for drug 1
    params_drug2: dict, fitted parameters for drug 2
    params_shifts: dict, containing shift parameters for both directions
    bootstrap_results: DataFrame with bootstrap analysis results (optional)
    cell_line: str, name of cell line for title
    grid_density: int, density of interpolation grid
    cmap: str, colormap to use
    vmin, vmax: float, limits for colormap
    
    Returns:
    fig: matplotlib figure
    """
    # Create figure with specific layout
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    # Get experimental doses
    exp_doses_E = np.sort(results['dose_E'].unique())
    exp_doses_X = np.sort(results['dose_X'].unique())

    # Create finer grid for smooth interpolation
    dose_E_min, dose_E_max = np.min(exp_doses_E[exp_doses_E > 0]), np.max(exp_doses_E)
    dose_X_min, dose_X_max = np.min(exp_doses_X[exp_doses_X > 0]), np.max(exp_doses_X)
    
    fine_doses_E = _create_dose_range(dose_E_min, dose_E_max + 1, grid_density, dose_E_max / dose_E_min > 10)
    fine_doses_X = _create_dose_range(dose_X_min, dose_X_max + 1, grid_density, dose_X_max / dose_X_min > 10)
    fine_doses_E = np.sort(np.concatenate([[0], fine_doses_E]))
    fine_doses_X = np.sort(np.concatenate([[0], fine_doses_X]))

    # Create mesh grid for interpolation
    X_fine, Y_fine = np.meshgrid(fine_doses_X, fine_doses_E)
    Z_delta = _calculate_delta_surface(fine_doses_E, fine_doses_X, params_drug1, params_drug2, params_shifts)

    # PANEL 1: Monotherapy dose-response curves with dual x-axis
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_panel1_monotherapy(ax1, results, params_drug1, params_drug2, fontsize)

    # Panel 2: Contour landscape
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_panel2_contour(ax2, X_fine, Y_fine, Z_delta, results, vmin, vmax, cmap, 
                         dose_E_min, dose_E_max, dose_X_min, dose_X_max, fontsize, fig)
    
    # Panel 3: Heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    if bootstrap_results is not None:
        _, ax3, _ = create_synergy_heatmap_base(bootstrap_results, 'delta_score_mean', cell_line,
                                                 cmap, vmin, vmax, ax3, True)
        annotate_heatmap_with_ci(ax3, bootstrap_results)
        ax3.set_title('Synergy with Statistical Significance', fontsize=fontsize + 2)
    else:
        _, ax3, _ = create_synergy_heatmap_base(results, 'delta_score_mean', cell_line,
                                                 cmap, vmin, vmax, ax3)
        ax3.set_title('Synergy Delta Scores', fontsize=fontsize + 2)
    
    # Panel 4: Delta by dose
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_panel4_delta_by_dose(ax4, results, dose_E_max / dose_E_min > 10, fontsize)
    
    title = f'{cell_line} Synergy Analysis (ZIP Model)' if cell_line else 'Synergy Analysis (ZIP Model)'
    fig.suptitle(title, fontsize=fontsize + 2, y=0.98)
    plt.tight_layout()
    return fig

def _calculate_delta_surface(doses_E, doses_X, params1, params2, params_shifts):
    """Calculate delta score surface for all dose combinations."""
    Z_delta = np.zeros((len(doses_E), len(doses_X)))
    
    for i, dose_e in enumerate(doses_E):
        for j, dose_x in enumerate(doses_X):
            if dose_e == 0 and dose_x == 0:
                continue
            
            y1 = 0 if dose_e == 0 else logistic_4PL(dose_e, params1['EC50'], params1['Hill'])
            y2 = 0 if dose_x == 0 else logistic_4PL(dose_x, params2['EC50'], params2['Hill'])
            y_zip = y1 + y2 - y1 * y2
            
            if dose_e > 0 and dose_x > 0:
                y_c_1to2 = potency_shift_model(dose_e, params_shifts['E_to_X']['EC50'],
                                               params_shifts['E_to_X']['Hill'], y2)
                y_c_2to1 = potency_shift_model(dose_x, params_shifts['X_to_E']['EC50'],
                                               params_shifts['X_to_E']['Hill'], y1)
                Z_delta[i, j] = (y_c_1to2 + y_c_2to1) / 2 - y_zip
    
    return Z_delta


def _plot_panel1_monotherapy(ax, results, params1, params2, fontsize):
    """Plot monotherapy panel with dual axes."""
    mono_E = results[results['dose_X'] == 0]
    mono_X = results[results['dose_E'] == 0]
    
    sns.lineplot(data=mono_E, x='dose_E', y='inhibition', marker='o', linestyle='none',
                 err_style="bars", errorbar=("se", 1), label='EcAII', color='tab:orange', ax=ax)
    
    dose_e_range = _create_dose_range(max(mono_E['dose_E'].min(), 1e-2), mono_E['dose_E'].max())
    ax.plot(dose_e_range, _vectorized_4PL(dose_e_range, params1['EC50'], params1['Hill']),
            '-', color='tab:orange', linewidth=2)
    
    ax.set_xscale('log')
    ax.set_xlabel('EcAII concentration (U/ml)', color='tab:orange', fontsize=fontsize)
    ax.tick_params(axis='x', labelcolor='tab:orange')
    ax.set_ylabel('Inhibition Fraction', fontsize=fontsize)
    
    ax1b = ax.twiny()
    sns.lineplot(data=mono_X, x='dose_X', y='inhibition', marker='s', linestyle='none',
                 err_style="bars", errorbar=("se", 1), label='X-ray', color='tab:purple',
                 ax=ax1b, legend=False)
    
    dose_x_range = np.linspace(mono_X['dose_X'].min(), mono_X['dose_X'].max(), 100)
    ax1b.plot(dose_x_range, _vectorized_4PL(dose_x_range, params2['EC50'], params2['Hill']),
              '-', color='tab:purple', linewidth=2)
    
    ax1b.set_xlabel('X-ray dose (Gy)', color='tab:purple', fontsize=fontsize)
    ax1b.tick_params(axis='x', labelcolor='tab:purple')
    
    param_text = (f"EcAII: EC50 = {params1['EC50']:.2f}, Hill = {params1['Hill']:.2f}\n"
                  f"X-ray: EC50 = {params2['EC50']:.2f}, Hill = {params2['Hill']:.2f}")
    _add_text_box(ax, param_text, fontsize=fontsize - 1)
    
    ax.set_title('Monotherapy Dose-Response Curves', fontsize=fontsize + 2)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=fontsize - 1)
    ax.grid(True, alpha=0.3)


def _plot_panel2_contour(ax, X, Y, Z, results, vmin, vmax, cmap, 
                         dose_E_min, dose_E_max, dose_X_min, dose_X_max, fontsize, fig):
    """Plot contour landscape panel."""
    masked_Z = np.ma.masked_invalid(Z)
    levels = np.linspace(vmin / 100, vmax / 100, 21)
    
    cs = ax.contourf(X, Y, masked_Z * 100, levels=levels * 100, cmap=cmap, alpha=0.9)
    contour = ax.contour(X, Y, masked_Z * 100, levels=np.linspace(vmin / 100, vmax / 100, 9) * 100,
                         colors='black', alpha=0.5, linewidths=0.7)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    
    combo = results[(results['dose_E'] > 0) & (results['dose_X'] > 0)]
    ax.scatter(combo['dose_X'], combo['dose_E'], color='black', s=20, marker='o')
    
    if dose_E_max / dose_E_min > 10:
        ax.set_yscale('log')
    if dose_X_max / dose_X_min > 10:
        ax.set_xscale('log')
    
    fig.colorbar(cs, ax=ax, label='Delta Score %')
    _format_axis(ax, 'X-ray dose (Gy)', 'EcAII concentration (U/ml)', 
                 'Synergy Landscape with Contours', fontsize)


def _plot_panel4_delta_by_dose(ax, results, use_log, fontsize):
    """Plot delta by dose panel."""
    combo = results[(results['dose_E'] > 0) & (results['dose_X'] > 0)].copy()
    
    ecaii_delta = combo.groupby('dose_E')['delta_score'].agg(['mean', 'sem']).reset_index()
    ax.errorbar(ecaii_delta['dose_E'], ecaii_delta['mean'], yerr=ecaii_delta['sem'],
                fmt='o-', color='blue', label='By EcAII dose', capsize=5)
    
    if use_log:
        ax.set_xscale('log')
    ax.set_xlabel('Dose', fontsize=fontsize)
    ax.set_ylabel('Mean Delta Score by EcAII dose', color='blue', fontsize=fontsize)
    ax.tick_params(axis='y', labelcolor='blue')
    
    ax4b = ax.twinx()
    xray_delta = combo.groupby('dose_X')['delta_score'].agg(['mean', 'sem']).reset_index()
    ax4b.errorbar(xray_delta['dose_X'], xray_delta['mean'], yerr=xray_delta['sem'],
                  fmt='s--', color='red', label='By X-ray dose', capsize=5)
    ax4b.set_ylabel('Mean Delta Score by X-ray dose', color='red', fontsize=fontsize)
    ax4b.tick_params(axis='y', labelcolor='red')
    
    global_mean, global_sem = combo['delta_score'].mean(), combo['delta_score'].sem()
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=global_mean, color='purple', linestyle='-', alpha=0.7)
    ax.axhspan(global_mean - global_sem, global_mean + global_sem, color='purple', alpha=0.1)
    
    _add_text_box(ax, f"Global δ = {global_mean:.3f} ± {global_sem:.3f}", x=0.05, y=0.05, va='bottom')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.set_title('Global Delta Score by Dose', fontsize=fontsize + 2)
    ax.grid(True, alpha=0.3)


def poster_synergy_heatmap(results, params_drug1, params_drug2, params_shifts,
                          bootstrap_results=None, cell_line=None, cmap='RdBu', vmin=-30, vmax=30):
    """
    Create poster-friendly synergy visualization with larger elements for better visibility.
    """    
    # Increase fontsize for poster visibility
    POSTER_FONTSIZE = 18
    TITLE_FONTSIZE = 24
    SUBTITLE_FONTSIZE = 20
    LINEWIDTH = 3
    MARKERSIZE = 12


def kde_bootstrap_distribution(global_deltas, n_iters, ax=None):
    # Distribution of bootstrap values
    # Use KDE to show evolution of distribution    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    # Choose 4-5 milestone iterations to show distribution evolution
    milestones = [int(n_iters * p) for p in [0.1, 0.25, 0.5, 0.75, 1.0]]
    
    # Add small density plots at the bottom of the axis
    density_height = 0.15  # Height of density plots as fraction of y-axis
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    density_y_base = y_min - (density_height * y_range)
    
    x_values = np.linspace(min(global_deltas), max(global_deltas), 100)
    
    for i, milestone in enumerate(milestones):
        if milestone == 0:  # Skip if milestone is 0
            continue
            
        # Get data up to this milestone
        data_subset = global_deltas[:milestone]
        
        # Calculate KDE
        try:
            kde = gaussian_kde(data_subset)
            density = kde(x_values)
            # Scale density to desired height
            density_scaled = density * (density_height * y_range) / np.max(density)
            # Plot density
            ax.fill_between(x_values, density_y_base, 
                            density_y_base + density_scaled,
                            alpha=0.3, color=plt.cm.viridis(i/len(milestones)))
            
            # Add milestone marker
            ax.text(x_values[0], density_y_base + 0.02 * y_range, 
                    f"{milestone}", fontsize=8, ha='left')
        except:
            # Skip if KDE fails (e.g., with identical values)
            pass
    # Adjust limits to include density plots
    plt.ylim(bottom=density_y_base - 0.05 * y_range)
    return fig, ax


def plot_delta_score_distributions(results_dict, cell_lines=None, fontsize=12, figsize=(10, 6)):
    """
    Create violin plots comparing the distribution of delta scores across cell lines.
    
    Parameters:
    results_dict: dict, mapping cell lines to their results DataFrames
    cell_lines: list, cell lines to include (if None, use all in results_dict)
    fontsize: int, font size for plot text
    figsize: tuple, size of the figure
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    if cell_lines is None:
        cell_lines = list(results_dict.keys())
        
    # Create a dataframe for plotting
    plot_data = []
    
    for cell_line in cell_lines:
        if cell_line not in results_dict:
            continue
            
        combo = results_dict[cell_line]
        combo = combo[(combo['dose_E'] > 0) & (combo['dose_X'] > 0)]
        for _, row in combo.iterrows():
            plot_data.append({'Cell Line': cell_line, 'Delta Score': row['delta_score'] * 100})
    
    plot_df = pd.DataFrame(plot_data)
    if plot_df.empty:
        return plt.figure(figsize=figsize)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=plot_df, x='Cell Line', y='Delta Score', ax=ax, inner='box', cut=0)
    sns.swarmplot(data=plot_df, x='Cell Line', y='Delta Score', ax=ax,
                  color='white', edgecolor='auto', size=2, alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    
    valid_lines = [cl for cl in cell_lines if cl in results_dict]
    for i, cell_line in enumerate(valid_lines):
        mean_delta = plot_df[plot_df['Cell Line'] == cell_line]['Delta Score'].mean()
        ax.text(i, mean_delta, f"Mean: {mean_delta:.2f}", ha='center', va='bottom',
                fontsize=fontsize - 2, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Statistical tests
    if len(valid_lines) > 1:
        groups = [plot_df[plot_df['Cell Line'] == cl]['Delta Score'] for cl in valid_lines]
        if len(groups) > 2:
            _, p_val = stats.f_oneway(*groups)
            test_name = "ANOVA"
        else:
            _, p_val = stats.ttest_ind(*groups)
            test_name = "t-test"
        ax.text(0.5, 0.01, f"{test_name}: p={p_val:.4f}", ha='center', transform=ax.transAxes,
                fontsize=fontsize - 2, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    _format_axis(ax, 'Cell Line', 'Delta Score (%)', 
                 'Distribution of Synergy Scores Across Cell Lines', fontsize)
    
    ax.axhline(y=10, color='green', linestyle=':', alpha=0.7)
    ax.axhline(y=-10, color='red', linestyle=':', alpha=0.7)
    ax.text(ax.get_xlim()[1], 10, 'Synergy', ha='right', va='bottom', fontsize=fontsize - 2)
    ax.text(ax.get_xlim()[1], -10, 'Antagonism', ha='right', va='top', fontsize=fontsize - 2)
    
    plt.tight_layout()
    return fig



def plot_potency_shift_comparison(params_dict, cell_lines=None, fontsize=12, figsize=(12, 6)):
    """
    Compare potency shift parameters across cell lines.
    
    Parameters:
    params_dict: dict, mapping cell lines to their parameter dictionaries
    cell_lines: list, cell lines to include (if None, use all in params_dict)
    fontsize: int, font size for plot text
    figsize: tuple, size of the figure
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    if cell_lines is None:
        cell_lines = list(params_dict.keys())
    
    # Extract potency shift parameters for each cell line
    data = {'e_to_x_ec50': [], 'e_to_x_hill': [], 'x_to_e_ec50': [], 'x_to_e_hill': [], 'labels': []}
    
    for cell_line in cell_lines:
        if cell_line not in params_dict or 'potency_shifts' not in params_dict[cell_line]:
            continue
        shifts = params_dict[cell_line]['potency_shifts']
        if 'E_to_X' in shifts and 'X_to_E' in shifts:
            data['e_to_x_ec50'].append(shifts['E_to_X']['EC50'])
            data['e_to_x_hill'].append(shifts['E_to_X']['Hill'])
            data['x_to_e_ec50'].append(shifts['X_to_E']['EC50'])
            data['x_to_e_hill'].append(shifts['X_to_E']['Hill'])
            data['labels'].append(cell_line)
    
    if not data['labels']:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No potency shift data available", ha='center', va='center',
                transform=ax.transAxes, fontsize=fontsize)
        ax.set_title("Potency Shift Parameters", fontsize=fontsize + 2)
        return fig
    

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Set bar width and positions
    bar_width = 0.35
    indices = np.arange(len(data['labels']))
    
    for ax, ec50_data, hill_data, title, ylabel in [
        (ax1, (data['e_to_x_ec50'], data['x_to_e_ec50']), None, 'EC50 Potency Shift Parameters', 'EC50'),
        (ax2, None, (data['e_to_x_hill'], data['x_to_e_hill']), 'Hill Coefficient Potency Shift Parameters', 'Hill Coefficient')
    ]:
        values = ec50_data if ec50_data else hill_data
        ax.bar(indices - bar_width / 2, values[0], bar_width, label='EcAII to X-ray', color='skyblue')
        ax.bar(indices + bar_width / 2, values[1], bar_width, label='X-ray to EcAII', color='salmon')
        ax.set_xticks(indices)
        ax.set_xticklabels(data['labels'], fontsize=fontsize - 1)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize + 1)
        ax.legend(fontsize=fontsize - 2)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Comparison of Potency Shift Parameters Across Cell Lines',
                 fontsize=fontsize + 4, y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig