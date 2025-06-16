import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
from scipy import stats
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


def set_scale_for_positive_doses(doses, ax, scale_type='x'):
    """Set log scale for axes if doses span multiple orders of magnitude."""
    positive_doses = [d for d in doses if d > 0]
    if positive_doses and max(positive_doses) / min(positive_doses) > 10:
        getattr(ax, f'set_{scale_type}scale')('log')


def plot_monotherapy_curve(data, drug_col, effect_col, params, style=None, title=None, x_logscale=True):
    """Plot monotherapy dose-response data with fitted curve."""
    # Validate that params contains required keys
    required_keys = ['EC50', 'Hill', 'errors', 'r_squared']
    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        raise ValueError(f"Missing required keys in params: {', '.join(missing_keys)}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot experimental points
    if style is None or style == 'bar':
        sns.lineplot(data=data, x=drug_col, y=effect_col, label='Experimental Data',
                     marker='o', alpha=0.7, linestyle='none', err_style="bars",
                     errorbar=("se", 2), err_kws={'capsize': 5}, ax=ax)
    elif style == 'scatter':
        for rep in data['rep'].unique():
            rep_data = data[data['rep'] == rep]
            ax.scatter(rep_data[drug_col], rep_data[effect_col], label=f'Rep {rep}', alpha=0.7)

    # Generate smooth curve from fitted parameters
    x_range = np.logspace(
        np.log10(max(data[drug_col].min(), 1e-3)), 
        np.log10(data[drug_col].max()), 
        100
    )

    y_fit = [logistic_4PL(x, params['EC50'], params['Hill']) for x in x_range]
    ax.plot(x_range, y_fit, 'r-', linewidth=2)

    # Add parameter annotations
    param_text = f"EC50 = {params['EC50']:.3f} ± {params['errors'][0]:.3f}\n"
    param_text += f"Hill = {params['Hill']:.3f} ± {params['errors'][1]:.3f}\n"
    param_text += f"R² = {params['r_squared']:.3f}"
    
    ax.text(0.05, 0.95, param_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    if x_logscale:
        ax.set_xscale('log')
    ax.set_xlabel('Dose')
    ax.set_ylabel('Inhibition Fraction')
    ax.set_title(title or 'Dose-Response Curve')
    #ax.legend()
    ax.grid(True, alpha=0.3)
    
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
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # EcAII monotherapy comparison
    ax1 = axes[0]

    for cell_line in cell_lines:
        if cell_line not in results_dict:
            continue    
        results = results_dict[cell_line]
        mono_E = results[results['dose_X'] == 0]
        grouped_data = mono_E.groupby('dose_E')['inhibition'].agg(['mean', 'sem']).reset_index()
        
        # Plot experimental points with error bars
        ax1.errorbar(
            grouped_data['dose_E'], grouped_data['mean'], 
            yerr=grouped_data['sem'],
            fmt='o', markersize=markersize, 
            linewidth=linewidth, elinewidth=linewidth, capsize=markersize/2,
            label=cell_line, color=color_dict[cell_line], alpha=0.8
        )

        # Generate fitted curve
        if cell_line in params_dict and 'params_ecaii' in params_dict[cell_line]:
            ecaii_params = params_dict[cell_line]['params_ecaii']
            dose_e_min = max(mono_E['dose_E'].min(), 1e-3)
            dose_e_max = mono_E['dose_E'].max()
            dose_e_range = np.logspace(np.log10(dose_e_min), np.log10(dose_e_max), 100)
            y_e_fit = [logistic_4PL(x, ecaii_params['EC50'], ecaii_params['Hill']) for x in dose_e_range]
            ax1.plot(dose_e_range, y_e_fit, '--', color=color_dict[cell_line], 
                    linewidth=linewidth, label=None)
    
    # Configure EcAII plot
    ax1.set_xscale('log')
    ax1.set_title('EcAII Monotherapy Response by Cell Line', fontsize=fontsize+2, fontweight='bold')
    ax1.set_xlabel('EcAII concentration (U/ml)', fontsize=fontsize)
    ax1.set_ylabel('Inhibition Fraction', fontsize=fontsize)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=fontsize-2)

    # Add parameter annotations for EcAII
    param_text = ""
    for cell_line in cell_lines:
        if cell_line in params_dict and 'params_ecaii' in params_dict[cell_line]:
            ecaii_params = params_dict[cell_line]['params_ecaii']
            param_text += f"{cell_line}: EC50={ecaii_params['EC50']:.2f}±{ecaii_params['errors'][0]:.2f} U/ml, Hill={ecaii_params['Hill']:.2f}±{ecaii_params['errors'][1]:.2f}\n"

    ax1.text(0.05, 0.9, param_text, transform=ax1.transAxes,
            verticalalignment='center', horizontalalignment='left', fontsize=fontsize-1,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    ax1.tick_params(axis='both', which='major', labelsize=fontsize, width=2, length=6)

    # X-ray monotherapy comparison
    ax2 = axes[1]

    for cell_line in cell_lines:
        if cell_line not in results_dict:
            continue
        results = results_dict[cell_line]
        mono_X = results[results['dose_E'] == 0]
        grouped_data = mono_X.groupby('dose_X')['inhibition'].agg(['mean', 'sem']).reset_index()

        # Plot experimental points with error bars
        ax2.errorbar(
            grouped_data['dose_X'], grouped_data['mean'], 
            yerr=grouped_data['sem'],
            fmt='s', markersize=markersize, 
            linewidth=linewidth, elinewidth=linewidth, capsize=markersize/2,
            label=cell_line, color=color_dict[cell_line], alpha=0.8
        )

        # Generate fitted curve
        if cell_line in params_dict and 'params_xray' in params_dict[cell_line]:
            xray_params = params_dict[cell_line]['params_xray']
            dose_x_min = mono_X['dose_X'].min()
            dose_x_max = mono_X['dose_X'].max()
            dose_x_range = np.linspace(dose_x_min, dose_x_max, 100)
            y_x_fit = [logistic_4PL(x, xray_params['EC50'], xray_params['Hill']) for x in dose_x_range]
            ax2.plot(dose_x_range, y_x_fit, '--', color=color_dict[cell_line], 
                    linewidth=linewidth, label=None)
    
    # Configure X-ray plot
    ax2.set_title('X-ray Monotherapy Response by Cell Line', fontsize=fontsize+2, fontweight='bold')
    ax2.set_xlabel('X-ray dose (Gy)', fontsize=fontsize)
    ax2.set_ylabel('Inhibition Fraction', fontsize=fontsize)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=fontsize-2)

    # Add parameter annotations for X-ray
    param_text = ""
    for cell_line in cell_lines:
        if cell_line in params_dict and 'params_xray' in params_dict[cell_line]:
            xray_params = params_dict[cell_line]['params_xray']
            param_text += f"{cell_line}: EC50={xray_params['EC50']:.2f}±{xray_params['errors'][0]:.2f} Gy, Hill={xray_params['Hill']:.2f}±{xray_params['errors'][1]:.2f}\n"

    ax2.text(0.05, 0.9, param_text, transform=ax2.transAxes,
            verticalalignment='center', horizontalalignment='left', fontsize=fontsize-1,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    ax2.tick_params(axis='both', which='major', labelsize=fontsize, width=2, length=6)

    # Set overall title
    fig.suptitle('Comparison of Monotherapy Response Across Cell Lines', 
                fontsize=fontsize+4, y=0.98, fontweight='bold')
    plt.tight_layout()
    
    return fig


def create_synergy_heatmap_base(results, value_to_plot='delta_score', cell_line=None, cmap='RdBu', vmin=-30, vmax=30, ax=None, annot=True,
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
    heatmap_data = heatmap_data.sort_index(ascending=False)
    
    # Convert to percentage for display (multiply by 100)
    display_data = heatmap_data * 100

    # Create a mask for monotherapy points if requested
    mask = None
    if mask_zeros:
        # Create a mask where delta scores are exactly zero (typical for monotherapy)
        mask = heatmap_data == 0
    
    # Plot heatmap
    sns.heatmap(
        display_data, 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax, 
        center=0,
        annot=annot, 
        fmt='.1f', 
        annot_kws={'size': fontsize},
        cbar_kws={'label': 'Delta Score % (Antagonism/Synergy)'},
        ax=ax,
        mask=mask
    )
    
    # Set titles and labels
    title = f'{cell_line or ""} Synergy Delta Scores (ZIP Model)'
    ax.set_title(title, fontsize=fontsize+2)
    ax.set_xlabel('X-ray dose (Gy)', fontsize=fontsize)
    ax.set_ylabel('EcAII concentration (U/ml)', fontsize=fontsize)

    # Ensure tick labels are readable
    ax.tick_params(axis='both', labelsize=fontsize-2)

    # Set colorbar properties after creating the heatmap
    cbar = ax.collections[0].colorbar  # Get the colorbar
    cbar.ax.tick_params(labelsize=fontsize-2)  # Set tick label font size
    cbar.set_label('Delta Score % (Antagonism/Synergy)', fontsize=fontsize-2)  # Set label font size

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
    delta_pivot = delta_pivot.sort_index(ascending=False)
    lower_pivot = lower_pivot.sort_index(ascending=False)
    upper_pivot = upper_pivot.sort_index(ascending=False)
    pval_pivot = pval_pivot.sort_index(ascending=False)

    # Get the row and column indices and values
    rows = delta_pivot.index
    cols = delta_pivot.columns

    # Add annotations for each cell
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            if pd.notna(delta_pivot.iloc[i, j]):
                delta = delta_pivot.iloc[i, j] * 100
                lower = lower_pivot.iloc[i, j] * 100
                upper = upper_pivot.iloc[i, j] * 100
                pval = pval_pivot.iloc[i, j]

                text_color = 'black' # if abs(delta) > 0.15 else 'black'

                # Add CI below the delta score
                ax.text(j + 0.5, i + 0.7, f"({lower:.1f}, {upper:.1f})", 
                        ha='center', va='center', 
                        fontsize=fontsize_ci,
                        color=text_color)

                # Add significance markers based on p-value thresholds
                # Determine the number of asterisks based on p-value
                significance_levels = sorted(significance_levels, reverse=True)
                asterisk_count = sum(1 for threshold in significance_levels if pval < threshold)
                        
                # Add asterisks if significant
                if asterisk_count > 0:
                    ax.text(j + 0.85, i + 0.15, '*' * asterisk_count, 
                            color='black', fontsize=fontsize_delta+2, fontweight='bold', 
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
    
    # Determine common dose grid for comparison
    common_doses_E = set()
    common_doses_X = set()
    
    for cell_line in cell_lines:
        if cell_line not in results_dict:
            continue
            
        results = results_dict[cell_line]
        combo_data = results[(results['dose_E'] > 0) & (results['dose_X'] > 0)]
        
        if not combo_data.empty:
            common_doses_E.update(combo_data['dose_E'].unique())
            common_doses_X.update(combo_data['dose_X'].unique())
    
    common_doses_E = sorted(common_doses_E)
    common_doses_X = sorted(common_doses_X)
    
    # Create subplots for each cell line's delta score heatmap
    n_cell_lines = len(cell_lines)
    n_cols = min(3, n_cell_lines)  # Max 3 columns
    n_rows = (n_cell_lines + n_cols - 1) // n_cols  # Ceiling division
    
    for i, cell_line in enumerate(cell_lines):
        if cell_line not in results_dict:
            continue
            
        results = results_dict[cell_line]
        
        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        # Get combo data
        combo_data = results[(results['dose_E'] > 0) & (results['dose_X'] > 0)]
        
        if combo_data.empty:
            ax.text(0.5, 0.5, f"No combination data for {cell_line}", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(cell_line, fontsize=fontsize+1, fontweight='bold')
            continue
        
        # Create base heatmap from bootstrap results
        _, ax, _ = create_synergy_heatmap_base(
            combo_data,
            value_to_plot='delta_score',
            cell_line=cell_line,
            ax=ax,
            annot=True,
            fontsize=fontsize-2,
            figsize=figsize
        )
                
        if bootstrap_dict and cell_line in bootstrap_dict:
            bootstrap_results = bootstrap_dict[cell_line]
            
            # Add significance markers
            try:
                annotate_heatmap_with_ci(
                    ax, 
                    bootstrap_results, 
                    significance_levels=[0.05, 0.01],
                    fontsize_delta=fontsize-2,
                    fontsize_ci=fontsize-4,
                    clear_existing=False,
                    cell_line=cell_line
                )
            except Exception as e:
                logger.error(f"Error annotating heatmap for {cell_line}: {e}")

        ax.set_title(cell_line, fontsize=fontsize+2, fontweight='bold')
        ax.set_xlabel('X-ray dose (Gy)', fontsize=fontsize)
        ax.set_ylabel('EcAII concentration (U/ml)', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2, width=2, length=6)

    # Set overall title
    fig.suptitle('Comparison of Synergy Patterns Across Cell Lines', fontsize=fontsize+4, y=0.98)
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
    
    # 1. Heatmap with significance markers (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create base heatmap
    _, ax1, _ = create_synergy_heatmap_base(
        bootstrap_results,
        value_to_plot='delta_score_mean',
        cell_line=cell_line,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax1,
        annot=True,
        fontsize=fontsize,
    )

    # Add confidence interval annotations
    ax1 = annotate_heatmap_with_ci(ax1, bootstrap_results, clear_existing=False, cell_line=cell_line, significance_levels= [0.05, 0.01, 0.005])
    ax1.set_title('Synergy Delta Scores with Significance', fontsize=fontsize+2)

    # 2. Distribution plot for selected dose combinations (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    # Select top 5 synergistic and top 5 antagonistic combinations
    top_synergy = bootstrap_results.nlargest(5, 'delta_score_mean')
    top_antagonism = bootstrap_results.nsmallest(5, 'delta_score_mean')
    selected_combos = pd.concat([top_synergy, top_antagonism])
 
    # Create visualization of delta score distributions
    for idx, row in enumerate(selected_combos.iterrows()):
        _, combo_data = row
        dose_e = combo_data['dose_E']
        dose_x = combo_data['dose_X']
        mean = combo_data['delta_score_mean']
        lower = combo_data['delta_score_lower']
        upper = combo_data['delta_score_upper']

        # Create a normal distribution approximation between the CI bounds
        std = (upper - lower) / (2 * 1.96)  # Approximate from 95% CI
        # Generate points for this distribution
        x = np.linspace(mean - 3*std, mean + 3*std, 100)
        y = scipy.stats.norm.pdf(x, mean, std)
        
        # Scale y values for better visualization
        y = y / np.max(y) * 0.8
        
        # Label
        label = f"E:{dose_e}, X:{dose_x}"
        
        # Plot distribution curve
        ax2.plot(x, y + idx, label=label)

        # Add CI markers
        ax2.plot([lower, upper], [idx, idx], 'k-', linewidth=2)
        ax2.plot(mean, idx, 'ko', markersize=6)

    # Add vertical line at 0
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add labels and formatting
    ax2.set_yticks(range(len(selected_combos)))
    ax2.set_yticklabels([f"E:{row['dose_E']}, X:{row['dose_X']}" for _, row in selected_combos.iterrows()])
    ax2.set_xlabel('Delta Score Distribution', fontsize=fontsize)
    ax2.set_title('Bootstrap Distributions for Key Dose Combinations', fontsize=fontsize+2)
    ax2.grid(True, axis='x', alpha=0.3)

    # 3. Diagnostic plot of bootstrap iterations (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])

    if bootstrap_raw_iter is not None:
        # Calculate global delta for each bootstrap iteration
        global_deltas = np.mean(bootstrap_raw_iter, axis=0)
        n_iters = len(global_deltas)
       
        # Create a twin axis for convergence metrics
        ax3b = ax3.twinx()

        # PLOT 3.1: Running average with convergence envelope
        running_avg = np.cumsum(global_deltas) / np.arange(1, n_iters + 1)

        # Calculate running standard error of the mean
        running_std = np.array([np.std(global_deltas[:i+1]) for i in range(n_iters)])
        running_sem = running_std / np.sqrt(np.arange(1, n_iters + 1))

        # Plot running average with envelope
        ax3.plot(running_avg, 'b-', linewidth=2, label='Running Average')
        ax3.fill_between(range(n_iters), 
                         running_avg - 1.96 * running_sem,
                         running_avg + 1.96 * running_sem,
                         color='blue', alpha=0.2, label='95% CI (running)')

        # Add individual iterations as scatter with transparency
        ax3.plot(global_deltas, 'o', alpha=0.1, markersize=2, color='gray')

        # PLOT 3.2: Final estimate and CI as reference
        final_mean = np.mean(global_deltas)
        ci_lower = np.percentile(global_deltas, 2.5)
        ci_upper = np.percentile(global_deltas, 97.5)

        ax3.axhline(y=final_mean, color='red', linestyle='-', label='Mean')
        ax3.axhspan(ci_lower, ci_upper, color='red', alpha=0.1, label='95% CI')

        # PLOT 3.3: Convergence metric (percent change)
        pct_change = np.abs(np.diff(running_avg) / running_avg[:-1] * 100)
        ax3b.plot(range(1, n_iters), pct_change, 'g-', alpha=0.5, label='% Change')
        # Add threshold line for convergence
        convergence_threshold = 0.1  # Consider converged when change < 0.1%
        ax3b.axhline(y=convergence_threshold, color='green', linestyle='--', alpha=0.7)
        
        # Find iteration where convergence is reached
        try:
            converged_iter = np.where(pct_change < convergence_threshold)[0][0]
            ax3.axvline(x=converged_iter, color='green', linestyle=':', alpha=0.7)
            ax3.text(converged_iter + 10, min(running_avg), 
                     f"Converged at iteration {converged_iter}", 
                     fontsize=fontsize-2, color='green')
        except IndexError:
            # Convergence not reached
            ax3.text(0.05, 0.05, "Convergence threshold not reached", 
                     transform=ax3.transAxes, fontsize=fontsize-2, color='red')
        
        # Set labels and legends
        ax3.set_xlabel('Bootstrap Iteration', fontsize=fontsize)
        ax3.set_ylabel('Global Delta Score', fontsize=fontsize)
        ax3b.set_ylabel('Percent Change in Estimate (%)', fontsize=fontsize-1, color='green')
        ax3b.tick_params(axis='y', labelcolor='green')

        # Combine legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, 
                  loc='upper right', fontsize=fontsize-2)
        
        ax3.set_title('Bootstrap Convergence Diagnostic', fontsize=fontsize+2)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Bootstrap iteration data not available", 
                ha='center', va='center', fontsize=fontsize)
        
    # 4. Actual vs. resampled delta comparison (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])

    # Merge bootstrap results with original deltas
    comparison_data = bootstrap_results.copy()

    # Prepare data for actual vs. resampled comparison
    merged_data = pd.merge(
        comparison_data,
        original_deltas[(original_deltas['dose_E'] > 0) & (original_deltas['dose_X'] > 0)].drop_duplicates(subset=['dose_E', 'dose_X']),
        on=['dose_E', 'dose_X'],
        suffixes=('_bootstrap', '')
    )

    # Plot actual vs. resampled deltas
    ax4.scatter(
        merged_data['delta_score'], 
        merged_data['delta_score_mean'],
        alpha=0.7,
        s=50,
        c=np.abs(merged_data['delta_score'] - merged_data['delta_score_mean']),
        cmap='viridis'
    )

    # Add error bars for bootstrap uncertainty
    ax4.errorbar(
        merged_data['delta_score'],
        merged_data['delta_score_mean'],
        yerr=[
            merged_data['delta_score_mean'] - merged_data['delta_score_lower'],
            merged_data['delta_score_upper'] - merged_data['delta_score_mean']
        ],
        fmt='none',
        ecolor='gray',
        alpha=0.3
    )

    # Add perfect correlation line
    min_val = min(merged_data['delta_score'].min(), merged_data['delta_score_mean'].min())
    max_val = max(merged_data['delta_score'].max(), merged_data['delta_score_mean'].max())
    margin = (max_val - min_val) * 0.1
    ax4.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 'k--', alpha=0.7)

    ax4.set_xlabel('Actual Delta Score', fontsize=fontsize)
    ax4.set_ylabel('Bootstrap Mean Delta Score', fontsize=fontsize)
    ax4.set_title('Actual vs. Bootstrap Delta Scores', fontsize=fontsize+2)
    ax4.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(merged_data['delta_score'], merged_data['delta_score_mean'])[0, 1]
    ax4.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=ax4.transAxes,
            bbox=dict(facecolor='white', alpha=0.7))
    
    # Set overall title
    fig.suptitle(f'{cell_line or ""} Bootstrap Analysis (95% CI)', fontsize=fontsize+4)
    plt.tight_layout()
    
    return fig


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
    
    # Create logarithmically spaced grid if needed
    if dose_E_max / dose_E_min > 10:
        fine_doses_E = np.logspace(np.log10(dose_E_min), np.log10(dose_E_max + 1), grid_density)
    else:
        fine_doses_E = np.linspace(dose_E_min, dose_E_max + 1, grid_density)
        
    if dose_X_max / dose_X_min > 10:
        fine_doses_X = np.logspace(np.log10(dose_X_min), np.log10(dose_X_max + 1), grid_density)
    else:
        fine_doses_X = np.linspace(dose_X_min, dose_X_max + 1, grid_density)

    # Add zeros to ensure we include monotherapy points
    fine_doses_E = np.sort(np.concatenate([[0], fine_doses_E]))
    fine_doses_X = np.sort(np.concatenate([[0], fine_doses_X]))
    
    # Create mesh grid for interpolation
    X_fine, Y_fine = np.meshgrid(fine_doses_X, fine_doses_E)
    
    # Calculate predicted effects for all dose pairs
    Z_zip = np.zeros_like(X_fine)
    Z_obs = np.zeros_like(X_fine)
    Z_delta = np.zeros_like(X_fine)
    
    for i, dose_e in enumerate(fine_doses_E):
        for j, dose_x in enumerate(fine_doses_X):
            # Skip (0,0) point
            if dose_e == 0 and dose_x == 0:
                Z_zip[i, j] = Z_obs[i, j] = Z_delta[i, j] = 0
                continue
                
            # Calculate monotherapy effects
            y1 = 0 if dose_e == 0 else logistic_4PL(dose_e, params_drug1['EC50'], params_drug1['Hill'])
            y2 = 0 if dose_x == 0 else logistic_4PL(dose_x, params_drug2['EC50'], params_drug2['Hill'])
            
            # Calculate ZIP effect
            Z_zip[i, j] = y1 + y2 - y1 * y2
            
            # For combination points, calculate observed effect
            if dose_e > 0 and dose_x > 0:
                # Calculate from perspective 1: E affects X
                y_c_1to2 = potency_shift_model(
                    dose_e, 
                    params_shifts['E_to_X']['EC50'], 
                    params_shifts['E_to_X']['Hill'], 
                    y2
                )
                
                # Calculate from perspective 2: X affects E
                y_c_2to1 = potency_shift_model(
                    dose_x, 
                    params_shifts['X_to_E']['EC50'], 
                    params_shifts['X_to_E']['Hill'], 
                    y1
                )
                
                # Average the two perspectives
                Z_obs[i, j] = (y_c_1to2 + y_c_2to1) / 2
                
                # Calculate delta score
                Z_delta[i, j] = Z_obs[i, j] - Z_zip[i, j]
            else:
                # For monotherapy points
                Z_obs[i, j] = max(y1, y2)
                Z_delta[i, j] = 0
    
    # PANEL 1: Monotherapy dose-response curves with dual x-axis
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Extract monotherapy data
    mono_E = results[results['dose_X'] == 0]
    mono_X = results[results['dose_E'] == 0]
    
    # Plot experimental points for EcAII (orange) on primary axis
    sns.lineplot(
        data=mono_E, x='dose_E', y='inhibition', 
        marker='o', linestyle='none', 
        err_style="bars", errorbar=("se", 1), 
        label='EcAII', color='tab:orange', ax=ax1
    )
    # Plot fitted curves for EcAII
    dose_e_range = np.logspace(np.log10(max(mono_E['dose_E'].min(), 1e-2)), np.log10(mono_E['dose_E'].max()), 100)

    y_e_fit = [logistic_4PL(x, params_drug1['EC50'], params_drug1['Hill']) for x in dose_e_range]
    ax1.plot(dose_e_range, y_e_fit, '-', color='tab:orange', linewidth=2)
    # Configure primary axis (EcAII)
    ax1.set_xscale('log')
    ax1.set_xlabel('EcAII concentration (U/ml)', color='tab:orange', fontsize=fontsize)
    ax1.tick_params(axis='x', labelcolor='tab:orange')
    ax1.set_ylabel('Inhibition Fraction', fontsize=fontsize)

    # Create twin axis for X-ray
    ax1b = ax1.twiny()
    
    # Plot experimental points for X-ray (purple)
    sns.lineplot(
        data=mono_X, x='dose_X', y='inhibition', 
        marker='s', linestyle='none', 
        err_style="bars", errorbar=("se", 1), 
        label='X-ray', color='tab:purple', ax=ax1b,
        legend=False
    )
    # Plot fitted curves for X-ray
    dose_x_range = np.linspace(mono_X['dose_X'].min(), mono_X['dose_X'].max(), 100)
    y_x_fit = [logistic_4PL(x, params_drug2['EC50'], params_drug2['Hill']) for x in dose_x_range]
    ax1b.plot(dose_x_range, y_x_fit, '-', color='tab:purple', linewidth=2)

    # Configure secondary axis (X-ray)
    ax1b.set_xlabel('X-ray dose (Gy)', color='tab:purple', fontsize=fontsize)
    ax1b.tick_params(axis='x', labelcolor='tab:purple')

    # Add parameter annotations
    param_text = f"EcAII: EC50 = {params_drug1['EC50']:.2f}, Hill = {params_drug1['Hill']:.2f}\n"
    param_text += f"X-ray: EC50 = {params_drug2['EC50']:.2f}, Hill = {params_drug2['Hill']:.2f}"
    
    ax1.text(0.05, 0.95, param_text, transform=ax1.transAxes, 
            verticalalignment='top', fontsize=fontsize-1,
            bbox=dict(boxstyle='round', facecolor='white'))

    ax1.set_title('Monotherapy Dose-Response Curves', fontsize=fontsize+2)
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=fontsize-1)

    ax1.grid(True, alpha=0.3)
    
    # PANEL 2: 2D landscape of delta scores with contours
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create masked array for delta scores
    masked_Z_delta = np.ma.masked_invalid(Z_delta)
    
    # Plot smooth contour-filled surface
    levels = np.linspace(vmin/100, vmax/100, 21) # Convert back to original scale for calculation
    cs = ax2.contourf(X_fine, Y_fine, masked_Z_delta * 100, levels=levels * 100, cmap=cmap, alpha=0.9)
    
    # Add contour lines with labels
    contour_levels = np.linspace(vmin/100, vmax/100, 9) #  Convert back to original scale
    contour = ax2.contour(X_fine, Y_fine, masked_Z_delta * 100, levels=contour_levels * 100, 
                        colors='black', alpha=0.5, linewidths=0.7)
    # Format contour labels as percentages
    ax2.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    
    # Add experimental points
    combo_points = results[(results['dose_E'] > 0) & (results['dose_X'] > 0)]
    ax2.scatter(combo_points['dose_X'], combo_points['dose_E'], color='black', s=20, marker='o')
    
    # Set axis scales
    if dose_E_max / dose_E_min > 10:
        ax2.set_yscale('log')
    if dose_X_max / dose_X_min > 10:
        ax2.set_xscale('log')
    
    # Add colorbar
    cbar = fig.colorbar(cs, ax=ax2, label='Delta Score %')
    
    ax2.set_title('Synergy Landscape with Contours', fontsize=fontsize+2)
    ax2.set_xlabel('X-ray dose (Gy)', fontsize=fontsize)
    ax2.set_ylabel('EcAII concentration (U/ml)', fontsize=fontsize)
    ax2.grid(True, alpha=0.3)
    
    # PANEL 3: Synergy heatmap with significance annotations
    ax3 = fig.add_subplot(gs[1, 0])
    
    if bootstrap_results is not None:
        # Create base heatmap from bootstrap results
        _, ax3, _ = create_synergy_heatmap_base(
            bootstrap_results,
            value_to_plot='delta_score_mean',
            cell_line=cell_line,
            cmap=cmap, 
            vmin=vmin, 
            vmax=vmax, 
            ax=ax3,
            annot=True
        )
        
        # Add confidence interval annotations
        ax3 = annotate_heatmap_with_ci(ax3, bootstrap_results)
        ax3.set_title('Synergy with Statistical Significance', fontsize=fontsize+2)
    else:
        # Use regular delta scores without CIs
        _, ax3, _ = create_synergy_heatmap_base(
            results,
            value_to_plot='delta_score_mean',
            cell_line=cell_line,
            cmap=cmap, 
            vmin=vmin, 
            vmax=vmax, 
            ax=ax3
        )
        ax3.set_title('Synergy Delta Scores', fontsize=fontsize+2)
    
    # PANEL 4: Global delta vs. dose
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Get combination data
    combo_data = results[(results['dose_E'] > 0) & (results['dose_X'] > 0)].copy()
    
    # Calculate global delta for each dose of EcAII
    ecaii_delta = combo_data.groupby('dose_E')['delta_score'].agg(['mean', 'sem']).reset_index()
    
    # Plot EcAII dose vs delta
    ax4.errorbar(
        ecaii_delta['dose_E'], 
        ecaii_delta['mean'],
        yerr=ecaii_delta['sem'],
        fmt='o-', 
        color='blue',
        label='By EcAII dose',
        capsize=5
    )
    # Set axis scales and labels
    if dose_E_max / dose_E_min > 10:
        ax4.set_xscale('log')

    ax4.set_xlabel('Dose', fontsize=fontsize)
    ax4.set_ylabel('Mean Delta Score by EcAII dose', color='blue', fontsize=fontsize)
    ax4.tick_params(axis='y', labelcolor='blue')

    # Calculate global delta for each dose of X-ray
    xray_delta = combo_data.groupby('dose_X')['delta_score'].agg(['mean', 'sem']).reset_index()
    
    # Create twin axis for X-ray
    ax4b = ax4.twinx()
    
    # Plot X-ray dose vs delta
    ax4b.errorbar(
        xray_delta['dose_X'], 
        xray_delta['mean'],
        yerr=xray_delta['sem'],
        fmt='s--', 
        color='red',
        label='By X-ray dose',
        capsize=5
    )
    
    ax4b.set_ylabel('Mean Delta Score by X-ray dose', color='red', fontsize=fontsize)
    ax4b.tick_params(axis='y', labelcolor='red')
    
    # Add zero line
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add overall mean line
    global_mean = combo_data['delta_score'].mean()
    global_sem = combo_data['delta_score'].sem()
    
    ax4.axhline(y=global_mean, color='purple', linestyle='-', alpha=0.7)
    ax4.axhspan(global_mean-global_sem, global_mean+global_sem, color='purple', alpha=0.1)
    
    # Add text with global stats
    global_text = f"Global δ = {global_mean:.3f} ± {global_sem:.3f}"
    ax4.text(0.05, 0.05, global_text, transform=ax4.transAxes,
            bbox=dict(facecolor='white', alpha=0.7), fontsize=fontsize-1)
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax4.set_title('Global Delta Score by Dose', fontsize=fontsize+2)
    ax4.grid(True, alpha=0.3)
    
    # Set overall title
    if cell_line:
        fig.suptitle(f'{cell_line} Synergy Analysis (ZIP Model)', fontsize=fontsize+2, y=0.98)
    else:
        fig.suptitle('Synergy Analysis (ZIP Model)', fontsize=fontsize+2, y=0.98)

    plt.tight_layout()
    
    return fig


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
        from scipy.stats import gaussian_kde
        
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
            
        results = results_dict[cell_line]
        
        # Get only combination data (where both drugs are present)
        combo_data = results[(results['dose_E'] > 0) & (results['dose_X'] > 0)]
        
        if not combo_data.empty:
            # Extract delta scores and cell line info
            for _, row in combo_data.iterrows():
                plot_data.append({
                    'Cell Line': cell_line,
                    'Delta Score': row['delta_score'] * 100  # Convert to percentage
                })
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    if plot_df.empty:
        return plt.figure(figsize=figsize)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create violin plot
    sns.violinplot(data=plot_df, x='Cell Line', y='Delta Score', ax=ax, inner='box', cut=0)
    
    # Add swarm plot for individual points
    sns.swarmplot(data=plot_df, x='Cell Line', y='Delta Score', ax=ax, 
                 color='white', edgecolor='auto', size=2, alpha=0.7)
    
    # Add a horizontal line at y=0 (no synergy)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    
    # Calculate and display the mean delta score for each cell line
    for i, cell_line in enumerate([cl for cl in cell_lines if cl in results_dict]):
        cell_data = plot_df[plot_df['Cell Line'] == cell_line]
        mean_delta = cell_data['Delta Score'].mean()
        ax.text(i, mean_delta, f"Mean: {mean_delta:.2f}", 
                ha='center', va='bottom', fontsize=fontsize-2, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add statistical test results
    if len(cell_lines) > 1:
        # Perform ANOVA if more than 2 groups
        cell_lines_in_data = plot_df['Cell Line'].unique()
        if len(cell_lines_in_data) > 2:
            groups = [plot_df[plot_df['Cell Line'] == cl]['Delta Score'] for cl in cell_lines_in_data]
            f_val, p_val = stats.f_oneway(*groups)
            ax.text(0.5, 0.01, f"ANOVA: p={p_val:.4f}", 
                    ha='center', transform=ax.transAxes, fontsize=fontsize-2,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        # Perform t-test if exactly 2 groups
        elif len(cell_lines_in_data) == 2:
            group1 = plot_df[plot_df['Cell Line'] == cell_lines_in_data[0]]['Delta Score']
            group2 = plot_df[plot_df['Cell Line'] == cell_lines_in_data[1]]['Delta Score']
            t_val, p_val = stats.ttest_ind(group1, group2)
            ax.text(0.5, 0.01, f"t-test: p={p_val:.4f}", 
                    ha='center', transform=ax.transAxes, fontsize=fontsize-2,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Formatting
    ax.set_title('Distribution of Synergy Scores Across Cell Lines', fontsize=fontsize+2)
    ax.set_xlabel('Cell Line', fontsize=fontsize)
    ax.set_ylabel('Delta Score (%)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    
    # Add horizontal lines for standard synergy thresholds
    ax.axhline(y=10, color='green', linestyle=':', alpha=0.7)
    ax.axhline(y=-10, color='red', linestyle=':', alpha=0.7)
    ax.text(ax.get_xlim()[1], 10, 'Synergy', ha='right', va='bottom', fontsize=fontsize-2)
    ax.text(ax.get_xlim()[1], -10, 'Antagonism', ha='right', va='top', fontsize=fontsize-2)
    
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
    e_to_x_ec50 = []
    e_to_x_hill = []
    x_to_e_ec50 = []
    x_to_e_hill = []
    cell_line_labels = []
    
    for cell_line in cell_lines:
        if cell_line not in params_dict:
            continue
            
        if 'potency_shifts' not in params_dict[cell_line]:
            # Extract from main parameters if possible
            if 'params_ecaii' in params_dict[cell_line] and 'params_xray' in params_dict[cell_line]:
                # For demonstration, just use placeholder data if potency shifts not stored
                e_to_x_ec50.append(0)
                e_to_x_hill.append(0)
                x_to_e_ec50.append(0)
                x_to_e_hill.append(0)
                cell_line_labels.append(cell_line)
            continue
        
        shifts = params_dict[cell_line]['potency_shifts']
        
        if 'E_to_X' in shifts and 'X_to_E' in shifts:
            e_to_x_ec50.append(shifts['E_to_X']['EC50'])
            e_to_x_hill.append(shifts['E_to_X']['Hill'])
            x_to_e_ec50.append(shifts['X_to_E']['EC50'])
            x_to_e_hill.append(shifts['X_to_E']['Hill'])
            cell_line_labels.append(cell_line)
    
    if not cell_line_labels:
        # If no potency shift data available, create placeholder figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No potency shift data available", 
                ha='center', va='center', transform=ax.transAxes, fontsize=fontsize)
        ax.set_title("Potency Shift Parameters", fontsize=fontsize+2)
        return fig
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Set bar width and positions
    bar_width = 0.35
    indices = np.arange(len(cell_line_labels))
    
    # Plot EC50 values
    ax1.bar(indices - bar_width/2, e_to_x_ec50, bar_width, label='EcAII to X-ray', color='skyblue')
    ax1.bar(indices + bar_width/2, x_to_e_ec50, bar_width, label='X-ray to EcAII', color='salmon')
    
    # Plot Hill coefficients
    ax2.bar(indices - bar_width/2, e_to_x_hill, bar_width, label='EcAII to X-ray', color='skyblue')
    ax2.bar(indices + bar_width/2, x_to_e_hill, bar_width, label='X-ray to EcAII', color='salmon')
    
    # Set titles and labels
    ax1.set_title('EC50 Potency Shift Parameters', fontsize=fontsize+1)
    ax2.set_title('Hill Coefficient Potency Shift Parameters', fontsize=fontsize+1)
    
    for ax in [ax1, ax2]:
        ax.set_xticks(indices)
        ax.set_xticklabels(cell_line_labels, fontsize=fontsize-1)
        ax.legend(fontsize=fontsize-2)
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
        ax.grid(True, alpha=0.3)
    
    ax1.set_ylabel('EC50', fontsize=fontsize)
    ax2.set_ylabel('Hill Coefficient', fontsize=fontsize)
    
    # Add main title
    fig.suptitle('Comparison of Potency Shift Parameters Across Cell Lines', 
                fontsize=fontsize+4, y=0.98, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    return fig
