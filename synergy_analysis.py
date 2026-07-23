import numpy as np
from typing import cast
from src.monotherapy import fit_monotherapy, logistic_4PL
from src.synergy import (calculate_delta_scores, get_potency_shifts, bootstrap_delta_scores,
                         bootstrap_model_scores, calculate_model_scores, classify_model_consensus)
from src.power_analysis import (
    convergence_diagnostics,
    empirical_ci_coverage,
    minimum_detectable_effect,
    plot_convergence_diagnostics,
    plot_mde_curve,
    plot_ci_coverage,
)
import src.visualization as viz
import pandas as pd
import matplotlib.pyplot as plt
import os
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


def load_cell_line_data(data_path):
    """Load a cell-line CSV and normalize the response column to `inhibition`."""
    data = pd.read_csv(data_path, comment='#')
    if 'live_normalized' in data.columns and 'inhibition' not in data.columns:
        data['inhibition'] = 1 - data['live_normalized']
    elif 'live' in data.columns and 'inhibition' not in data.columns:
        data['inhibition'] = 1 - data['live']
    return data


def analyze_cell_line(data_path, cell_line, n_bootstrap=1000, confidence_level=0.95, output_dir='results'):
    """Run full analysis for one cell line."""
    logger.info(f"Analyzing {cell_line}...")

    # Load data
    data = load_cell_line_data(data_path)
        
    # Check available columns and calculate inhibition appropriately
    logger.info(f"Available columns in {cell_line} data: {data.columns.tolist()}")
    
    if 'live_normalized' in data.columns and 'inhibition' not in data.columns:
        logger.info(f"Converting proliferation to inhibition for {cell_line}.")
        data['inhibition'] = 1 - data['live_normalized']
    elif 'inhibition' in data.columns:
        # Inhibition is already calculated
        pass
    elif 'live' in data.columns:
        # Try using 'live' column if available
        logger.info(f"Calculating inhibition from 'live' for {cell_line}.")
        data['inhibition'] = 1 - data['live']
    else:
        logger.error(f"No valid inhibition data found for {cell_line}. Available columns: {data.columns.tolist()}")
        raise ValueError(f"Data for {cell_line} must contain 'live_normalized', 'inhibition', or 'live' columns.")
    

    # Extract monotherapy data
    ecaii_mono = data[data['dose_X'] == 0]
    xray_mono = data[data['dose_E'] == 0]
    
    # Fit monotherapy curves
    params_ecaii = fit_monotherapy(ecaii_mono, 'dose_E', 'inhibition')
    params_xray = fit_monotherapy(xray_mono, 'dose_X', 'inhibition')
    
    if 'covariance_matrix' not in params_ecaii:
        logger.error(f"Failed to fit EcAII monotherapy for {cell_line}. Check data integrity.")
        raise ValueError(f"Monotherapy fitting failed for {cell_line} with EcAII data.")
    if 'covariance_matrix' not in params_xray:
        logger.error(f"Failed to fit X-ray monotherapy for {cell_line}. Check data integrity.")
        raise ValueError(f"Monotherapy fitting failed for {cell_line} with X-ray data.")

    # Generate plots
    os.makedirs(f'{output_dir}/figures/{cell_line}', exist_ok=True)
    fig_ecaii = viz.plot_monotherapy_curve(
        ecaii_mono, 'dose_E', 'inhibition', params_ecaii,
        style='bar',
        title=f'{cell_line}: EcAII Monotherapy'
    )
    fig_ecaii.savefig(f'{output_dir}/figures/{cell_line}/ecaii_monotherapy.png', dpi=300, bbox_inches='tight')
    plt.close(fig_ecaii)

    fig_xray = viz.plot_monotherapy_curve(
        xray_mono, 'dose_X', 'inhibition', params_xray, style='bar',
        title = f'{cell_line}: X-ray Monotherapy', x_logscale=False
    )
    fig_xray.savefig(f'{output_dir}/figures/{cell_line}/xray_monotherapy.png', dpi=300, bbox_inches='tight')
    plt.close(fig_xray)

    # Fit potency shift parameters using combination data
    logger.info(f"Fitting potency shift parameters for {cell_line}...")
    params_shifts = get_potency_shifts(data, params_ecaii, params_xray, drug_col1='dose_E', drug_col2='dose_X')   
    # Log the fitted parameters
    logger.info(f"EcAII to X-ray potency shift: EC50={params_shifts['E_to_X']['EC50']:.3f}±{params_shifts['E_to_X']['errors'][0]:.3f}, Hill={params_shifts['E_to_X']['Hill']:.3f}±{params_shifts['E_to_X']['errors'][1]:.3f}, R²={params_shifts['E_to_X']['r_squared']:.3f}")
    logger.info(f"X-ray to EcAII potency shift: EC50={params_shifts['X_to_E']['EC50']:.3f}±{params_shifts['X_to_E']['errors'][0]:.3f}, Hill={params_shifts['X_to_E']['Hill']:.3f}±{params_shifts['X_to_E']['errors'][1]:.3f}, R²={params_shifts['X_to_E']['r_squared']:.3f}")

    # Calculate delta scores
    results = calculate_delta_scores(data, params_ecaii, params_xray, params_shifts)
    # Save delta scores results
    os.makedirs(f'{output_dir}/parameters/{cell_line}', exist_ok=True)
    results.to_csv(f'{output_dir}/parameters/{cell_line}/{cell_line}_delta_scores.csv', index=False)

    # Perform bootstrap analysis
    logger.info(f"Performing bootstrap analysis with {n_bootstrap} iterations...")
    bootstrap_results, bootstrap_raw_iter = bootstrap_delta_scores(data=data,
        params_drug1=params_ecaii, params_drug2=params_xray,
        n_bootstrap=n_bootstrap, confidence_level=confidence_level,
        drug_col1='dose_E', drug_col2='dose_X'
    )
    bootstrap_results = cast(pd.DataFrame, bootstrap_results)
    # Save bootstrap results
    bootstrap_results.to_csv(f'{output_dir}/parameters/{cell_line}/{cell_line}_bootstrap_results.csv', index=False)
    logger.info(f"Bootstrap analysis completed for {cell_line}. Results saved in {output_dir}/parameters/{cell_line}/.")

    # Create bootstrap visualizations
    # Create detailed bootstrap results visualization
    fig_detailed = viz.plot_detailed_bootstrap_results(
        bootstrap_results=bootstrap_results, 
        original_deltas=results,  # Pass original delta scores
        cell_line=cell_line,
        bootstrap_raw_iter=bootstrap_raw_iter,  # Include bootstrap data for detailed results
        fontsize=15
    )
    fig_detailed.savefig(f'{output_dir}/figures/{cell_line}/detailed_bootstrap_results.png', dpi=300, bbox_inches='tight')
    plt.close(fig_detailed)

    # Count significant synergistic and antagonistic combinations
    fig_publication = viz.panel_synergy_heatmap(
        results, 
        params_ecaii, 
        params_xray, 
        params_shifts,
        bootstrap_results=bootstrap_results,  # Include bootstrap results
        cell_line=cell_line
    )
    fig_publication.savefig(f'{output_dir}/figures/{cell_line}/publication_quality_synergy.png', dpi=300, bbox_inches='tight')
    plt.close(fig_publication)

    fig_contour = viz.plot_contour_landscape(
        results=results,
        params_drug1=params_ecaii,
        params_drug2=params_xray,
        params_shifts=params_shifts,
        cell_line=cell_line,
        grid_density=100,
        cmap='RdBu',
        vmin=-25,
        vmax=25,
        fontsize=12
    )
    fig_contour.savefig(f'{output_dir}/figures/{cell_line}/contour_landscape.png', dpi=300, bbox_inches='tight')
    plt.close(fig_contour)
    logger.info(f"Contour landscape saved for {cell_line}.")

    # Return the results for further analysis
    logger.info(f"Analysis completed for {cell_line}\n")
    return results, bootstrap_results, params_ecaii, params_xray, params_shifts



def analyze_cell_line_multimodel(data_path, cell_line, models=('zip', 'bliss', 'hsa', 'loewe'),
                                  n_bootstrap=1000, confidence_level=0.95, output_dir='results'):
    """Run ZIP analysis + multi-model bootstrap for one cell line.

    Extends ``analyze_cell_line`` by also running ``bootstrap_model_scores``
    for the specified models and saving the per-model bootstrap summary CSV.

    Parameters
    ----------
    data_path : str
        Path to the normalised CSV for this cell line.
    cell_line : str
        Cell-line label used for directory and file names.
    models : tuple of str
        Synergy models to compare ('zip', 'bliss', 'hsa', 'loewe').
    n_bootstrap : int
        Bootstrap iterations.
    confidence_level : float
        Percentile CI level.
    output_dir : str
        Root directory for result files.

    Returns
    -------
    results : DataFrame   — ZIP delta scores
    bootstrap_results : DataFrame  — ZIP bootstrap summary
    multimodel_bootstrap : DataFrame — multi-model bootstrap summary
    params_ecaii, params_xray, params_shifts : dicts
    """
    # Run standard ZIP analysis (figures, ZIP bootstrap, delta scores)
    results, bootstrap_results, params_ecaii, params_xray, params_shifts = analyze_cell_line(
        data_path, cell_line, n_bootstrap=n_bootstrap,
        confidence_level=confidence_level, output_dir=output_dir
    )

    # Load data again for multi-model bootstrap
    data = load_cell_line_data(data_path)

    logger.info(f"Running multi-model bootstrap ({models}) for {cell_line}...")
    multimodel_bootstrap, _ = bootstrap_model_scores(
        data=data,
        params_drug1=params_ecaii,
        params_drug2=params_xray,
        models=models,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        drug_col1='dose_E',
        drug_col2='dose_X'
    )

    # Add point-estimate model scores and consensus
    model_scores_df = results.copy()
    for m in models:
        model_scores_df = calculate_model_scores(
            model_scores_df, params_ecaii, params_xray, model=m,
            drug_col1='dose_E', drug_col2='dose_X'
        )
    model_scores_df = classify_model_consensus(model_scores_df, models=models)

    # Save outputs
    out_params = f'{output_dir}/parameters/{cell_line}'
    os.makedirs(out_params, exist_ok=True)
    multimodel_bootstrap.to_csv(f'{out_params}/{cell_line}_multimodel_bootstrap.csv', index=False)
    model_scores_df.to_csv(f'{out_params}/{cell_line}_model_scores.csv', index=False)
    logger.info(f"Multi-model results saved for {cell_line}.")

    return results, bootstrap_results, multimodel_bootstrap, params_ecaii, params_xray, params_shifts


def build_multimodel_summary_table(cell_lines, output_dir='results',
                                   models=('zip', 'bliss', 'hsa', 'loewe')):
    """Aggregate multi-model bootstrap CSVs into a single manuscript-ready table.

    Loads ``{output_dir}/parameters/{cl}/{cl}_multimodel_bootstrap.csv`` for
    each cell line and produces a summary DataFrame with one row per
    (cell_line, dose_E, dose_X, model) containing mean deviation, 95% CI,
    adjusted p-value, and effect classification.

    Parameters
    ----------
    cell_lines : list of str
    output_dir : str
    models : tuple of str

    Returns
    -------
    summary : DataFrame
    """
    rows = []
    for cl in cell_lines:
        path = f'{output_dir}/parameters/{cl}/{cl}_multimodel_bootstrap.csv'
        if not os.path.exists(path):
            logger.warning(f"Multi-model bootstrap not found for {cl}: {path}")
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            for m in models:
                if f'{m}_mean' not in df.columns:
                    continue
                rows.append({
                    'cell_line': cl,
                    'dose_E': row['dose_E'],
                    'dose_X': row['dose_X'],
                    'model': m,
                    'deviation_mean': row[f'{m}_mean'],
                    'deviation_lower': row[f'{m}_lower'],
                    'deviation_upper': row[f'{m}_upper'],
                    'p_value': row[f'{m}_p_value'],
                    'p_adjusted': row[f'{m}_p_adjusted'],
                    'significant': row[f'{m}_significant'],
                    'effect_type': row[f'{m}_effect_type'],
                })
    summary = pd.DataFrame(rows)
    return summary


def run_power_diagnostics(bootstrap_iterations, dose_pairs_df, output_dir, cell_line,
                          confidence_level=0.95, step_size=50,
                          run_coverage=False, coverage_simulations=50,
                          coverage_bootstrap=200, run_mde=False,
                          mde_simulations=50, mde_bootstrap=200,
                          mde_effect_sizes=None):
    """Run and save power diagnostics for one cell line.

    Always computes convergence diagnostics from raw bootstrap iterations.
    Coverage and MDE analyses are optional because they are simulation-heavy.

    Returns
    -------
    dict with keys that may include:
      convergence_df, coverage_df, coverage_summary, mde_df
    """
    out_params = f'{output_dir}/parameters/{cell_line}'
    out_figs = f'{output_dir}/figures/{cell_line}'
    os.makedirs(out_params, exist_ok=True)
    os.makedirs(out_figs, exist_ok=True)

    outputs = {}

    convergence_df = convergence_diagnostics(
        bootstrap_iterations,
        dose_pairs_df,
        confidence_level=confidence_level,
        step_size=step_size,
    )
    convergence_df.to_csv(f'{out_params}/{cell_line}_convergence_diagnostics.csv', index=False)
    fig_conv = plot_convergence_diagnostics(
        convergence_df,
        confidence_level=confidence_level,
    )
    fig_conv.savefig(f'{out_figs}/{cell_line}_convergence_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close(fig_conv)
    outputs['convergence_df'] = convergence_df

    if run_coverage:
        coverage_df, coverage_summary = empirical_ci_coverage(
            true_params_E={'EC50': 0.2, 'Hill': 1.5},
            true_params_X={'EC50': 3.0, 'Hill': 1.2},
            n_simulations=coverage_simulations,
            n_bootstrap=coverage_bootstrap,
            confidence_level=confidence_level,
        )
        coverage_df.to_csv(f'{out_params}/{cell_line}_ci_coverage.csv', index=False)
        pd.DataFrame([coverage_summary]).to_csv(
            f'{out_params}/{cell_line}_ci_coverage_summary.csv', index=False
        )
        fig_cov = plot_ci_coverage(coverage_df, coverage_summary)
        fig_cov.savefig(f'{out_figs}/{cell_line}_ci_coverage.png', dpi=300, bbox_inches='tight')
        plt.close(fig_cov)
        outputs['coverage_df'] = coverage_df
        outputs['coverage_summary'] = coverage_summary

    if run_mde:
        mde_df = minimum_detectable_effect(
            n_bootstrap=mde_bootstrap,
            n_simulations=mde_simulations,
            effect_sizes=mde_effect_sizes,
        )
        mde_df.to_csv(f'{out_params}/{cell_line}_mde_table.csv', index=False)
        fig_mde = plot_mde_curve(mde_df)
        fig_mde.savefig(f'{out_figs}/{cell_line}_mde_curve.png', dpi=300, bbox_inches='tight')
        plt.close(fig_mde)
        outputs['mde_df'] = mde_df

    logger.info(f"Power diagnostics saved for {cell_line}.")
    return outputs


def compare_cell_lines(results_dict, params_dict, bootstrap_dict=None, 
                       cell_lines=None, fontsize=12, linewidth=2, markersize=6, figsize=(18, 8)):
    """
    Compare monotherapy dose-response and synergy across multiple cell lines.
    
    Parameters:
    results_dict: dict, mapping cell lines to their results DataFrames
    params_dict: dict, mapping cell lines to their parameter dictionaries containing:
                 {'params_ecaii': {...}, 'params_xray': {...}, 'params_shifts': {...}}
    bootstrap_dict: dict, mapping cell lines to their bootstrap results (optional)
    cell_lines: list, cell lines to include (if None, use all in results_dict)
    fontsize: int, font size for plot text
    linewidth: int, line width for plots
    markersize: int, marker size for plots
    figsize: tuple, size of the figure
    
    Returns:
    figs: list of matplotlib figures
    """
    if cell_lines is None:
        cell_lines = list(results_dict.keys())

    # Create list to store figures
    figures = []

    # Figure 1: Monotherapy response comparison
    fig_mono = viz.plot_cell_line_monotherapy_comparison(
        results_dict=results_dict,
        params_dict=params_dict,
        cell_lines=cell_lines,
        fontsize=fontsize,
        linewidth=linewidth,
        markersize=markersize,
        figsize=figsize
    )
    figures.append(fig_mono)

    # Figure 2: Delta Score Comparison across cell lines
    fig_delta = viz.plot_synergy_heatmap_comparison(
        results_dict=results_dict,
        bootstrap_dict=bootstrap_dict,
        cell_lines=cell_lines,
        fontsize=fontsize,
        figsize=figsize
    )
    figures.append(fig_delta)

    # Figure 3: Delta score distribution comparison
    fig_distribution = viz.plot_delta_score_distributions(
        results_dict,
        cell_lines=cell_lines,
        fontsize=fontsize,
        figsize=figsize
    )
    figures.append(fig_distribution)
    
    # Figure 4: Potency shift parameter comparison
    # First, ensure potency shifts are included in params_dict
    for cell_line in cell_lines:
        if cell_line in params_dict and 'params_shifts' not in params_dict[cell_line]:
            # Try to extract from results if available
            if cell_line in results_dict:
                try:
                    results = results_dict[cell_line]
                    params_shifts = get_potency_shifts(
                        results, 
                        params_dict[cell_line]['params_ecaii'], 
                        params_dict[cell_line]['params_xray']
                    )
                    params_dict[cell_line]['params_shifts'] = params_shifts
                except Exception as e:
                    logger.warning(f"Could not calculate potency shifts for {cell_line}: {e}")

    fig_potency = viz.plot_potency_shift_comparison(
        params_dict=params_dict,
        cell_lines=cell_lines,
        fontsize=fontsize,
        figsize=figsize
    )
    figures.append(fig_potency)

    # Figure 5: Contour landscape comparison across cell lines
    fig_contour_comp = viz.plot_contour_landscape_comparison(
        results_dict=results_dict,
        params_dict=params_dict,
        cell_lines=cell_lines,
        grid_density=100,
        cmap='RdBu',
        vmin=-25,
        vmax=25,
        fontsize=fontsize
    )
    figures.append(fig_contour_comp)

    return figures


def load_existing_results(cell_lines, output_dir='results'):
    """
    Load previously computed results for specified cell lines.
    
    Parameters:
    cell_lines: list of cell lines to load
    output_dir: str, directory where results are stored. (default: 'results')

    Returns:
    Tuple of (all_results, all_params, all_bootstrap_results, loaded_cell_lines)
    """
    all_results = {}
    all_params = {}
    all_bootstrap_results = {}
    loaded_cell_lines = []
    
    for cell_line in cell_lines:
        # Check if parameter files exist
        delta_scores_path = f'{output_dir}/parameters/{cell_line}/{cell_line}_delta_scores.csv'
        bootstrap_path = f'{output_dir}/parameters/{cell_line}/{cell_line}_bootstrap_results.csv'

        if not os.path.exists(delta_scores_path):
            logging.warning(f"Delta scores file not found for {cell_line}: {delta_scores_path}")
            continue
            
        # Load delta scores results
        try:
            results = pd.read_csv(delta_scores_path)
            all_results[cell_line] = results
            
            # Extract parameters from the monotherapy data
            mono_E = results[results['dose_X'] == 0]
            mono_X = results[results['dose_E'] == 0]
            
            params_ecaii = fit_monotherapy(mono_E, 'dose_E', 'inhibition')
            params_xray = fit_monotherapy(mono_X, 'dose_X', 'inhibition')
            try:
                params_shifts = get_potency_shifts(results, params_ecaii, params_xray, 
                                                   drug_col1='dose_E', drug_col2='dose_X')
                logging.info(f"Successfully calculated potency shifts for {cell_line}")
            except Exception as e:
                logging.warning(f"Could not calculate potency shifts for {cell_line}: {e}")
                params_shifts = None
            
            
            all_params[cell_line] = {
                'params_ecaii': params_ecaii,
                'params_xray': params_xray,
                'params_shifts': params_shifts
            }
            
            loaded_cell_lines.append(cell_line)
            logging.info(f"Successfully loaded results for {cell_line}")

            # Load bootstrap results if available
            if os.path.exists(bootstrap_path):
                bootstrap_results = pd.read_csv(bootstrap_path)
                all_bootstrap_results[cell_line] = bootstrap_results
                logging.info(f"Successfully loaded bootstrap results for {cell_line}")
            
        except Exception as e:
            logging.error(f"Error loading results for {cell_line}: {e}")
    
    return all_results, all_params, all_bootstrap_results, loaded_cell_lines


def analyze_drug_synergy(data_dir='data/processed', output_dir='results', cell_lines=None,
                         with_bootstrap=True, nbootstrap=1000, use_existing=False,
                         figsize=(18, 12), fontsize=18, linewidth=4, markersize=8,
                         with_power_diagnostics=False, convergence_step_size=50,
                         with_coverage=False, coverage_simulations=50,
                         coverage_bootstrap=200, with_mde=False,
                         mde_simulations=50, mde_bootstrap=200,
                         mde_effect_sizes=None, with_multimodel=False,
                         multimodel_models=('zip', 'bliss', 'hsa', 'loewe')):
    """Run analysis for specified cell lines or load existing results.
    
    Parameters:
    data_dir: str, directory containing processed data files
    output_dir: str, directory for saving results
    cell_lines: list, cell lines to analyze (if None, uses default list)
    with_bootstrap: bool, whether to perform bootstrap analysis
    nbootstrap: int, number of bootstrap iterations
    use_existing: bool, whether to use existing results if available
    figsize: tuple, figure size
    fontsize: int, font size for plots
    linewidth: int, line width for plots
    markersize: int, marker size for plots
    with_power_diagnostics: bool, whether to export convergence diagnostics
    convergence_step_size: int, checkpoint interval for convergence diagnostics
    with_coverage: bool, whether to run empirical CI coverage simulations
    coverage_simulations: int, number of coverage simulation datasets
    coverage_bootstrap: int, bootstrap iterations per coverage simulation
    with_mde: bool, whether to run minimum detectable effect analysis
    mde_simulations: int, number of simulation datasets per effect size
    mde_bootstrap: int, bootstrap iterations per MDE simulation
    mde_effect_sizes: sequence, optional effect sizes for MDE sweep
    with_multimodel: bool, whether to export multimodel bootstrap and score tables
    multimodel_models: tuple of str, model set for multimodel analysis

    Returns:
    tuple: (all_results, all_params) dictionaries with analysis results
"""
    # Create output directories
    os.makedirs(f'{output_dir}/parameters', exist_ok=True)
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    # Default cell lines if none provided
    if cell_lines is None:
        # Modify this list to include the cell lines you want to analyze
        # Ensure these correspond to your actual data files in the data/processed directory
        cell_lines = ['A549', 'BT549', '786O', 'MDAMB231']  # Add your cell lines here

    # Set up dictionary to store results for each cell line
    all_results = {}
    all_params = {}
    all_bootstrap_results = {}

    if use_existing:
        # Try to load existing results
        logger.info("Loading existing results instead of re-analyzing...")
        all_results, all_params, all_bootstrap_results, loaded_cell_lines = load_existing_results(cell_lines, output_dir=output_dir)
        
        if not loaded_cell_lines:
            logger.info("No existing results found. Performing full analysis instead.")
            use_existing = False
        else:
            logger.info(f"Successfully loaded results for: {', '.join(loaded_cell_lines)}")
            cell_lines = loaded_cell_lines
    
    if not use_existing:
        # Run full analysis for each cell line
        for cell_line in cell_lines:
            try:
                data_path = os.path.join(data_dir, f'{cell_line}_synergy_normalized.csv')
                if not os.path.exists(data_path):
                    logger.warning(f"Warning: Data file not found: {data_path}")
                    continue

                if with_bootstrap:
                    if with_multimodel:
                        logger.info(f"Running ZIP + multimodel analysis with bootstrap for {cell_line}...")
                        results, bootstrap_results, _, params_ecaii, params_xray, params_shifts = analyze_cell_line_multimodel(
                            data_path,
                            cell_line,
                            models=multimodel_models,
                            n_bootstrap=nbootstrap,
                            output_dir=output_dir
                        )
                    else:
                        logger.info(f"Running ZIP analysis with bootstrap for {cell_line}...")
                        results, bootstrap_results, params_ecaii, params_xray, params_shifts = analyze_cell_line(
                            data_path, cell_line, n_bootstrap=nbootstrap, output_dir=output_dir
                        )

                    bootstrap_results = cast(pd.DataFrame, bootstrap_results)
                    all_bootstrap_results[cell_line] = bootstrap_results

                    if with_power_diagnostics:
                        combo_dose_pairs = bootstrap_results[['dose_E', 'dose_X']].copy()
                        power_data = load_cell_line_data(data_path)
                        _, bootstrap_raw_iter = bootstrap_delta_scores(
                            data=power_data,
                            params_drug1=params_ecaii,
                            params_drug2=params_xray,
                            n_bootstrap=nbootstrap,
                            confidence_level=0.95,
                            drug_col1='dose_E',
                            drug_col2='dose_X'
                        )
                        run_power_diagnostics(
                            bootstrap_raw_iter,
                            combo_dose_pairs,
                            output_dir=output_dir,
                            cell_line=cell_line,
                            confidence_level=0.95,
                            step_size=convergence_step_size,
                            run_coverage=with_coverage,
                            coverage_simulations=coverage_simulations,
                            coverage_bootstrap=coverage_bootstrap,
                            run_mde=with_mde,
                            mde_simulations=mde_simulations,
                            mde_bootstrap=mde_bootstrap,
                            mde_effect_sizes=mde_effect_sizes,
                        )
                else:
                    results, _, params_ecaii, params_xray, params_shifts = analyze_cell_line(
                        data_path, cell_line, output_dir=output_dir
                    )

                # Store results and parameters for this cell line
                all_results[cell_line] = results
                all_params[cell_line] = {
                    'params_ecaii': params_ecaii,
                    'params_xray': params_xray,
                    'params_shifts': params_shifts
                }
            except Exception as e:
                logging.error(f"Error processing {cell_line}: {e}")

    # Compare cell lines after all processing is done
    has_bootstrap = with_bootstrap or (use_existing and len(all_bootstrap_results) > 0)

    if has_bootstrap:
        figs_comparison = compare_cell_lines(all_results, all_params, all_bootstrap_results, cell_lines=cell_lines, 
                                             figsize=figsize, fontsize=fontsize, linewidth=linewidth, markersize=markersize)
    else:
        logger.info("No bootstrap results available for comparison.")
        figs_comparison = compare_cell_lines(all_results, all_params,
                                             figsize=figsize, fontsize=fontsize, linewidth=linewidth, markersize=markersize)

    # Save comparison figures
    if figs_comparison:
        for i, fig in enumerate(figs_comparison):
            fig.savefig(f'{output_dir}/figures/comparison_synergy_{i}.png', dpi=400)
            plt.close(fig)
        logger.info("Comparison figures saved.")

    logger.info("Analysis completed.")

    return all_results, all_params

def parse_figsize(s):
    """Convert string 'width,height' to tuple (width, height)

        Example:
            Input: '18,12'
            Output: (18.0, 12.0)
    """
    try:
        width, height = map(float, s.split(','))
        return (width, height)
    except:
        raise argparse.ArgumentTypeError("Figsize must be width,height")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run synergy analysis with or without bootstrap')
    parser.add_argument('--data-dir', type=str, default='data/processed', 
                        help='Directory containing processed data files')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results and figures')
    parser.add_argument('--cell-lines', type=str, nargs='+',
                        help='Cell lines to analyze (default: A549, BT549, 786O, MDAMB231)')
    parser.add_argument('--bootstrap', action='store_true', help='Run with bootstrap analysis')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of bootstrap iterations')
    parser.add_argument('--use-existing', action='store_true', help='Use existing analysis results instead of recomputing')
    parser.add_argument('--power-diagnostics', action='store_true',
                        help='Export bootstrap convergence diagnostics per cell line')
    parser.add_argument('--multimodel', action='store_true',
                        help='Run multimodel analysis (ZIP, Bliss, HSA, Loewe) in addition to ZIP outputs')
    parser.add_argument('--convergence-step-size', type=int, default=50,
                        help='Checkpoint spacing for convergence diagnostics')
    parser.add_argument('--coverage', action='store_true',
                        help='Run empirical CI coverage simulations (slow)')
    parser.add_argument('--coverage-simulations', type=int, default=50,
                        help='Number of synthetic datasets for coverage analysis')
    parser.add_argument('--coverage-bootstrap', type=int, default=200,
                        help='Bootstrap iterations per coverage simulation')
    parser.add_argument('--mde', action='store_true',
                        help='Run minimum detectable effect analysis (slow)')
    parser.add_argument('--mde-simulations', type=int, default=50,
                        help='Simulation datasets per effect-size point for MDE')
    parser.add_argument('--mde-bootstrap', type=int, default=200,
                        help='Bootstrap iterations per MDE simulation')
    parser.add_argument('--mde-effect-sizes', type=float, nargs='+',
                        help='Optional effect sizes for MDE sweep')
    parser.add_argument('--fontsize', type=int, default=18, help='Font size for figures')
    parser.add_argument('--linewidth', type=int, default=4, help='Line width for plots')
    parser.add_argument('--figsize', type=parse_figsize, default=(18, 12), help='Figure size as width,height (e.g., 18,12)')
    parser.add_argument('--markersize', type=int, default=8, help='Marker size for plots')
    
    args = parser.parse_args()

    analyze_drug_synergy(data_dir=args.data_dir,
                         output_dir=args.output_dir,
                         cell_lines=args.cell_lines,
                         with_bootstrap=args.bootstrap, 
                         nbootstrap=args.iterations, 
                         use_existing=args.use_existing,
                         with_power_diagnostics=args.power_diagnostics,
                         convergence_step_size=args.convergence_step_size,
                         with_coverage=args.coverage,
                         coverage_simulations=args.coverage_simulations,
                         coverage_bootstrap=args.coverage_bootstrap,
                         with_mde=args.mde,
                         mde_simulations=args.mde_simulations,
                         mde_bootstrap=args.mde_bootstrap,
                         mde_effect_sizes=args.mde_effect_sizes,
                         with_multimodel=args.multimodel,
                         fontsize=args.fontsize,
                         linewidth=args.linewidth,
                         figsize=args.figsize,
                         markersize=args.markersize
    )