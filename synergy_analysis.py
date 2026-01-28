from src.monotherapy import fit_monotherapy, logistic_4PL
from src.synergy import calculate_delta_scores, get_potency_shifts, bootstrap_delta_scores
import src.visualization as viz
import pandas as pd
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


def analyze_cell_line(data_path, cell_line, n_bootstrap=1000, confidence_level=0.95, output_dir='results'):
    """Run full analysis for one cell line."""
    logger.info(f"Analyzing {cell_line}...")

    # Load data
    data = pd.read_csv(data_path)
        
    # Check available columns and calculate inhibition appropriately
    logger.info(f"Available columns in {cell_line} data: {data.columns.tolist()}")
    
    if 'live_normalized' in data.columns:
        # Convert proliferation to inhibition
        logger.info(f"Normalizing 'live_normalized' for {cell_line}.")
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
    fig_ecaii.savefig(f'{output_dir}/figures/{cell_line}/ecaii_monotherapy.png')

    fig_xray = viz.plot_monotherapy_curve(
        xray_mono, 'dose_X', 'inhibition', params_xray, style='bar',
        title = f'{cell_line}: X-ray Monotherapy', x_logscale=False
    )
    fig_xray.savefig(f'{output_dir}/figures/{cell_line}/xray_monotherapy.png')

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
    
    # Return the results for further analysis
    logger.info(f"Analysis completed for {cell_line}\n")
    return results, bootstrap_results


def compare_cell_lines(results_dict, params_dict, bootstrap_dict=None, 
                       cell_lines=None, fontsize=12, linewidth=2, markersize=6, figsize=(18, 8)):
    """
    Compare monotherapy dose-response and synergy across multiple cell lines.
    
    Parameters:
    results_dict: dict, mapping cell lines to their results DataFrames
    params_dict: dict, mapping cell lines to their parameter dictionaries containing:
                 {'params_ecaii': {...}, 'params_xray': {...}}
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
        if cell_line in params_dict and 'potency_shifts' not in params_dict[cell_line]:
            # Try to extract from results if available
            if cell_line in results_dict:
                try:
                    results = results_dict[cell_line]
                    params_shifts = get_potency_shifts(
                        results, 
                        params_dict[cell_line]['params_ecaii'], 
                        params_dict[cell_line]['params_xray']
                    )
                    params_dict[cell_line]['potency_shifts'] = params_shifts
                except Exception as e:
                    logger.warning(f"Could not calculate potency shifts for {cell_line}: {e}")

    fig_potency = viz.plot_potency_shift_comparison(
        params_dict=params_dict,
        cell_lines=cell_lines,
        fontsize=fontsize,
        figsize=figsize
    )
    figures.append(fig_potency)

    return figures


def load_existing_results(cell_lines, output_dir='results'):
    """
    Load existing results for specified cell lines.
    
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
            
            all_params[cell_line] = {
                'params_ecaii': params_ecaii,
                'params_xray': params_xray
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
                         figsize=(18, 12), fontsize=18, linewidth=4, markersize=8):
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
                    logger.info(f"Running analysis with bootstrap for {cell_line}...")
                    results, bootstrap_results = analyze_cell_line(data_path, cell_line, n_bootstrap=nbootstrap, output_dir=output_dir)
                    all_bootstrap_results[cell_line] = bootstrap_results
                else:
                    results, _ = analyze_cell_line(data_path, cell_line, output_dir=output_dir)

                # Extract the parameters from your analysis
                params_ecaii = fit_monotherapy(results[results['dose_X'] == 0], 'dose_E', 'inhibition')
                params_xray = fit_monotherapy(results[results['dose_E'] == 0], 'dose_X', 'inhibition')
                # Store results and parameters for this cell line
                all_results[cell_line] = results
                all_params[cell_line] = {
                    'params_ecaii': params_ecaii,
                    'params_xray': params_xray
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
                         fontsize=args.fontsize,
                         linewidth=args.linewidth,
                         figsize=args.figsize,
                         markersize=args.markersize
    )