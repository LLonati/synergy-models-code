import numpy as np
import pandas as pd
from src.monotherapy import logistic_4PL, calculate_r_squared, fit_monotherapy
from scipy.optimize import curve_fit
from scipy import stats
import logging
from statsmodels.stats.multitest import multipletests


def calculate_zip_effect(dose1, dose2, params1, params2):
    """Calculate expected ZIP effect based on Bliss independence.
        
    Parameters:
    dose1: float, dose of drug 1
    dose2: float, dose of drug 2
    params1: dict, fitted parameters for drug 1
    params2: dict, fitted parameters for drug 2
    
    Returns:
    float: Expected ZIP effect (y_ZIP = y1 + y2 - y1*y2)
    """
    # Ensure parameters are valid
    if 'EC50' not in params1 or 'Hill' not in params1:
        raise ValueError("params1 must contain 'EC50' and 'Hill' keys")
    if not isinstance(params1['EC50'], (int, float)) or not isinstance(params1['Hill'], (int, float)):
        raise ValueError("params1 values for 'EC50' and 'Hill' must be numeric")
    if 'EC50' not in params2 or 'Hill' not in params2:
        raise ValueError("params2 must contain 'EC50' and 'Hill' keys")
    if not isinstance(params2['EC50'], (int, float)) or not isinstance(params2['Hill'], (int, float)):
        raise ValueError("params2 values for 'EC50' and 'Hill' must be numeric")


    # Calculate individual effects
    y1 = 0 if dose1 == 0 else logistic_4PL(dose1, params1['EC50'], params1['Hill'])
    y2 = 0 if dose2 == 0 else logistic_4PL(dose2, params2['EC50'], params2['Hill'])
    
    # Calculate Bliss independence effect
    return y1 + y2 - y1 * y2


def potency_shift_model(dose1, EC50, Hill, mono_effect2):
    """
    Model for how the response of drug 2 shifts based on the dose of drug 1.
    
    Parameters:
    dose1: float, dose of drug 1
    dose2: float, dose of drug 2
    EC50: float, dose of drug 1 at which the effect is half-maximal
    Hill: float, Hill coefficient
    mono_effect2: float, monotherapy effect of dose2 alone
    
    Returns:
    float: Expected effect of drug 2 given the dose of drug 1
    """
    # This implements: y_C = (y2 + (dose1/EC50)^Hill) / (1 + (dose1/EC50)^Hill)
    # this is equivalent to the shifted effect of drug 2 at the given dose1
    # y_C = logistic_4PL(dose1, EC50, Hill, bottom=mono_effect2, top=1)
    # The current implementation is preferred because it directly calculates the shifted effect
    # using the dose ratio power formula, which avoids dependency on additional parameters 
    # like 'bottom' and 'top' in logistic_4PL, ensuring simplicity and consistency.
    
    # Calculate the shifted effect of drug 2 at the given dose
    if EC50 == 0:
        raise ValueError("EC50 must be non-zero to avoid division by zero.")
    
    dose_ratio_power = (dose1 / EC50) ** Hill
    return (mono_effect2 + dose_ratio_power) / (1 + dose_ratio_power)


def calculate_observed_combination_effect(dose1, dose2, params1, params2, params_1to2, params_2to1):
    """Calculate observed combination effect as average of both directions.
    
    Parameters:
    dose1: float, dose of drug 1
    dose2: float, dose of drug 2
    params1: dict, monotherapy fitted parameters for drug 1
    params2: dict, monotherapy fitted parameters for drug 2
    params_1to2: dict, fitted parameters for potency shift from drug 1 to drug 2
    params_2to1: dict, fitted parameters for potency shift from drug 2 to drug 1

    Returns:
    float: Expected observed effect of the combination averaged from both directions
    """
    # Calculate monotherapy effects
    y1 = logistic_4PL(dose1, params1['EC50'], params1['Hill'])
    y2 = logistic_4PL(dose2, params2['EC50'], params2['Hill'])

    # Calculate effect from first drug's perspective (1←2)
    y_c_1to2 = potency_shift_model(dose1, params_1to2['EC50'], params_1to2['Hill'], y2)

    # Calculate effect from second drug's perspective (2←1)
    y_c_2to1 = potency_shift_model(dose2, params_2to1['EC50'], params_2to1['Hill'], y1)
    # Average the two effects
    return (y_c_1to2 + y_c_2to1) / 2


def fit_potency_shift_parameters(data, drug_col1, drug_col2, params1=None, params2=None, effect_col='inhibition'):
    """
    Fit potency shift parameters from combination data.
    Model how one drug's dose response shifts as a function of the other drug's dose.
    
    Parameters:
    data: DataFrame with dose and effect data
    drug_col1: str, column name for the shifting drug (e.g., 'dose_E')
    drug_col2: str, column name for the drug which shifts the response of drug_col1 (e.g., 'dose_X')
    effect_col: str, column name for effect (Default: 'inhibition')
    
    Returns:
    dict: Dictionary with fitted parameters for shifts
    """
    # Get only combination data (both drugs > 0)
    combo_data = data[(data[drug_col1] > 0) & (data[drug_col2] > 0)].copy()
    
    # Early exit if no combination data
    if len(combo_data) == 0:
        logging.warning(f"No combination data for {drug_col1} and {drug_col2}.")
        return {
            'EC50': np.nan,
            'Hill': np.nan,
            'errors': np.array([np.nan, np.nan]),
            'r_squared': np.nan,
            'covariance_matrix': np.array([[np.nan, np.nan], [np.nan, np.nan]])
        }
    
    # Get monotherapy data for each drug
    mono1_data = data[(data[drug_col1] > 0) & (data[drug_col2] == 0)].copy()
    mono2_data = data[(data[drug_col1] == 0) & (data[drug_col2] > 0)].copy()
    
    if params1 is None:
        # Fit monotherapy for drug_col1 if not provided
        params1 = fit_monotherapy(mono1_data, drug_col1, effect_col)
    if params2 is None:
        # Fit monotherapy for drug_col2 if not provided
        params2 = fit_monotherapy(mono2_data, drug_col2, effect_col)

    # Get dose values
    dose1_array = combo_data[drug_col1].values
    dose2_array = combo_data[drug_col2].values
    observed_effects = combo_data[effect_col].values

    # Define a vectorized fitting function that uses the potency shift model
    def fit_function(doses, EC50, Hill):
        """Wrapper for potency_shift_model to use with curve_fit"""
        # Ensure parameters are valid
        EC50 = max(EC50, 1e-6)  # Prevent division by zero

        dose_1_values = dose1_array
        dose_2_values = dose2_array

        # Calculate monotherapy effect for the shifted drug for each dose2
        mono_effect = np.array([
            logistic_4PL(d2, params2['EC50'], params2['Hill'])
            for d2 in dose_2_values
        ])

        # Apply potency shift model for each dose pair
        results = np.array([
            potency_shift_model(d1, EC50, Hill, m_effect)
            for d1, m_effect in zip(dose_1_values, mono_effect)
        ])  
        return results
    
    # Initial parameter guesses
    initial_ec50_guess = np.median(data[drug_col1][data[drug_col1] > 0]) # Calculate initial EC50 guess    
    if not np.isfinite(initial_ec50_guess) or initial_ec50_guess <= 0:
        initial_ec50_guess = 1.0  # Fallback value   
    p0 = [initial_ec50_guess, 1.0]  # Initial guess for EC50 and Hill
    bounds = ([1e-6, 0.1], [1e5, 10])  # Bounds for parameters

    try:
        # Fit the model to the data - use dummy xdata since we already have our arrays
        params_fit, covariance = curve_fit(
            fit_function, np.ones(len(observed_effects)),  # Dummy xdata
            observed_effects,  # ydata is the observed effects
            p0=p0, bounds=bounds
        )

        # Calculate errors
        errors = np.sqrt(np.diag(covariance))

        # Calculate predicted values and R-squared
        y_pred = fit_function(None, *params_fit)
        r2 = calculate_r_squared(observed_effects, y_pred)

        # Return fitted parameters and metrics
        return {
            'EC50': params_fit[0],
            'Hill': params_fit[1],
            'errors': errors,
            'r_squared': r2,
            'covariance_matrix': covariance
            }
    
    except Exception as e:
        logging.error(
            f"Error fitting potency shift model for {drug_col1} to {drug_col2}: {e}. "
            f"Input parameters: drug_col1={drug_col1}, drug_col2={drug_col2}, effect_col={effect_col}. "
            f"Data shape: {data.shape}, combination data shape: {combo_data.shape}.",
            exc_info=True
        )
        # Return None or some default parameters if fitting fails
        return {
            'EC50': np.nan,
            'Hill': np.nan,
            'errors': np.array([np.nan, np.nan]),
            'r_squared': np.nan,
            'covariance_matrix':  np.array([[np.nan, np.nan], [np.nan, np.nan]])
        }


def get_potency_shifts(data, params_drug1, params_drug2, drug_col1='dose_E', drug_col2='dose_X'):
    """
    Calculate potency shifts in both directions.

    This function calculates potency shifts for two drugs or experimental conditions
    being studied for their combined effects in a synergy model.
   
    Parameters:
    data: DataFrame with experimental combination data
    params_drug1: dict, fitted monotherapy parameters for drug 1
    params_drug2: dict, fitted monotherapy parameters for drug 2
    drug_col1: str, column name for drug 1 (Default: 'dose_E')
    drug_col2: str, column name for drug 2 (Default: 'dose_X')

    Returns:
    dict: Dictionary with shift parameters for both directions
    """
    # Fit shifts for drug 1 affecting drug 2's response
    params_1to2 = fit_potency_shift_parameters(
        data, drug_col1, drug_col2, params_drug1, params_drug2, effect_col='inhibition'
    )
    # Fit shifts for drug 2 affecting drug 1's response
    params_2to1 = fit_potency_shift_parameters(
        data, drug_col2, drug_col1, params_drug2, params_drug1, effect_col='inhibition'
    )
    
    return {
        'X_to_E': params_1to2,  # X-ray effect on EcAII
        'E_to_X': params_2to1   # EcAII effect on X-ray
    }


def calculate_delta_scores(data, params_drug1, params_drug2, params_shifts=None):
    """
    Calculate delta scores for all dose combinations.
    
    Delta score represents the difference between the observed combination effect 
    (`y_observed`) and the expected ZIP effect (`y_ZIP`). It quantifies the deviation 
    from the Bliss independence model, indicating synergy (positive delta score) 
    or antagonism (negative delta score) between the drugs.
        
    Parameters:
    data: DataFrame with experimental data
    params_drug1: dict, fitted parameters for drug 1
    params_drug2: dict, fitted parameters for drug 2
    params_shifts: dict, containing shift parameters for both directions
    
    Returns:
    DataFrame with model results
    """
    results = data.copy()
    # Calculate expected effects for monotherapies
    results['y1'] = results['dose_E'].apply(
        lambda x: logistic_4PL(x, params_drug1['EC50'], params_drug1['Hill'])
    )
    results['y2'] = results['dose_X'].apply(
        lambda x: logistic_4PL(x, params_drug2['EC50'], params_drug2['Hill'])
    )
    # Calculate expected ZIP effect
    results['y_ZIP'] = results.apply(
        lambda row: calculate_zip_effect(row['dose_E'], row['dose_X'], params_drug1, params_drug2), 
        axis=1
    )

    # Calculate potency shifts if not provided
    if params_shifts is None:
        params_shifts = get_potency_shifts(results, params_drug1, params_drug2)

    # Validate params_shifts contains required keys
    required_keys = ['E_to_X', 'X_to_E']            
    for direction in required_keys:
        if direction not in params_shifts:
            raise KeyError(f"Missing required key '{direction}' in params_shifts")
        for param in ['EC50', 'Hill']:
            if param not in params_shifts[direction]:
                raise ValueError(f"Missing required key '{param}' in params_shifts['{direction}']")
            if not np.isfinite(params_shifts[direction][param]):
                raise ValueError(f"Invalid value for '{param}' in params_shifts['{direction}']")      

    # Only calculate potency shifts for combination points
    combination_mask = (results['dose_E'] > 0) & (results['dose_X'] > 0)

    # For combination points, calculate observed effects from both perspectives
    if combination_mask.any():
        dose_E_array = results.loc[combination_mask, 'dose_E'].values
        dose_X_array = results.loc[combination_mask, 'dose_X'].values

        # Calculate monotherapy effects
        y1_array = logistic_4PL(dose_E_array, params_drug1['EC50'], params_drug1['Hill'])
        y2_array = logistic_4PL(dose_X_array, params_drug2['EC50'], params_drug2['Hill'])

        # Calculate effects from both perspectives
        # 1. E affects X response: y_C^(1←2)
        y_c_1to2_array = potency_shift_model(
            dose_E_array,
            params_shifts['E_to_X']['EC50'],
            params_shifts['E_to_X']['Hill'], 
            y2_array
        )
        # 2. X affects E response: y_C^(2←1)
        y_c_2to1_array = potency_shift_model(
            dose_X_array,
            params_shifts['X_to_E']['EC50'],
            params_shifts['X_to_E']['Hill'],
            y1_array
        )

        # Store individual perspective effects for analysis
        results.loc[combination_mask, 'y_c_1to2'] = y_c_1to2_array
        results.loc[combination_mask, 'y_c_2to1'] = y_c_2to1_array

        # Average the effects to get overall observed effect
        y_observed_combo = (y_c_1to2_array + y_c_2to1_array) / 2

        results['y_observed'] = results['inhibition']        
        results.loc[combination_mask, 'y_observed_model'] = y_observed_combo

        # Calculate delta scores accordingly to
        # δ(θ) = (y_C^(1←2) - y_ZIP)/2 + (y_C^(2←1) - y_ZIP)/2
        # Since y_ZIP is the same from both perspectives, this simplifies to:
        # δ(θ) = (y_C^(1←2) + y_C^(2←1))/2 - y_ZIP = y_observed_model - y_ZIP
        results.loc[combination_mask, 'delta_score'] = (
            results.loc[combination_mask, 'y_observed_model'] - 
            results.loc[combination_mask, 'y_ZIP']
        )

        # Calculate experimental delta scores
        results.loc[combination_mask, 'delta_score_exp'] = (
            results.loc[combination_mask, 'inhibition'] -
            results.loc[combination_mask, 'y_ZIP']
        )
    else:
        # If no combination data, set delta scores to NaN
        results['y_c_1to2'] = np.nan
        results['y_c_2to1'] = np.nan
        results['y_observed_model'] = np.nan
        results['delta_score'] = np.nan
        results['delta_score_exp'] = np.nan

    return results


def bootstrap_delta_scores(data, params_drug1, params_drug2, n_bootstrap=1000,
                           confidence_level=0.95, drug_col1='dose_E', drug_col2='dose_X'):
    """
    Bootstrap delta scores by sampling parameters from their estimated distributions
    and recalculating potency shifts for each bootstrap iteration.

    Parameters:
    data: DataFrame with experimental data
    params_drug1: dict, fitted parameters for drug 1 (must include 'EC50', 'Hill', 'covariance_matrix')
    params_drug2: dict, fitted parameters for drug 2 (must include 'EC50', 'Hill', 'covariance_matrix')
    n_bootstrap: int, number of bootstrap iterations
    confidence_level: float, confidence level for intervals (0.95 = 95% CI)
    drug_col1: str, column name for drug 1 (Default: 'dose_E')
    drug_col2: str, column name for drug 2 (Default: 'dose_X')

    Returns:
    DataFrame with bootstrap results including confidence intervals
    """
    # Add progress reporting
    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_bootstrap), desc="Bootstrap iterations")
    except ImportError:
        # Fallback to simple progress reporting if tqdm not installed
        print(f"Starting {n_bootstrap} bootstrap iterations...")
        iterator = range(n_bootstrap)

    # Validate inputs
    for params, name in [(params_drug1, "params_drug1"), (params_drug2, "params_drug2")]:
        if 'covariance_matrix' not in params:
            raise ValueError(f"Missing covariance matrix in {name}")
    
    # Extract all combination data (including replicates) for fitting potency shifts
    combo_mask = (data[drug_col1] > 0) & (data[drug_col2] > 0)
    combo_data = data[combo_mask].copy()
    
    if len(combo_data) == 0:
        logging.warning("No combination data provided for bootstrap analysis. Returning empty DataFrame.")
        return pd.DataFrame(columns=[drug_col1, drug_col2, 'delta_score_mean', 'delta_score_lower', 'delta_score_upper', 'significant'])

    # Get unique dose combinations for calculating delta scores
    unique_combo_data = combo_data.drop_duplicates([drug_col1, drug_col2])

    # Extract parameters and covariance matrices
    params1_mean = np.array([params_drug1['EC50'], params_drug1['Hill']])
    params2_mean = np.array([params_drug2['EC50'], params_drug2['Hill']])    
    cov1 = params_drug1['covariance_matrix']
    cov2 = params_drug2['covariance_matrix']
    
    # Initialize array to store bootstrap results
    # For each unique dose combination, we'll store delta scores from all bootstrap iterations
    bootstrap_iterations = np.zeros((len(unique_combo_data), n_bootstrap))

    # Run bootstrap iterations
    for i in iterator:
        # Sample parameters from multivariate normal distributions
        params1_sample = np.random.multivariate_normal(params1_mean, cov1)
        params2_sample = np.random.multivariate_normal(params2_mean, cov2)
        
        # Log sampled parameters for debugging
        logging.debug(f"Bootstrap iteration {i}: params1_sample={params1_sample}, params2_sample={params2_sample}")
       
        # Create parameter dictionaries for this bootstrap iteration
        params1_dict = {'EC50': params1_sample[0], 'Hill': params1_sample[1], 'covariance_matrix': cov1}
        params2_dict = {'EC50': params2_sample[0], 'Hill': params2_sample[1], 'covariance_matrix': cov2}
        
        # Calculate potency shifts with these sampled parameters
        # Use FULL dataset (with replicates) for robust fitting
        params_shifts_dict = get_potency_shifts(data, params1_dict, params2_dict, drug_col1, drug_col2)

        # Calculate delta scores for all dose combinations
        delta_results = calculate_delta_scores(data, params1_dict, params2_dict, params_shifts_dict)

        # Extract just the delta scores for each unique dose combination and store in bootstrap_results
        for j, (_, row) in enumerate(unique_combo_data.iterrows()):
            # Find the delta score for this dose combination
            dose1 = row[drug_col1]
            dose2 = row[drug_col2]
            # Get first matching row from delta_results
            match_idx = delta_results[(delta_results[drug_col1] == dose1) & 
                                      (delta_results[drug_col2] == dose2)].index[0]
            bootstrap_iterations[j, i] = delta_results.loc[match_idx, 'delta_score']

    # Calculate statistics for each dose combination
    delta_means = np.mean(bootstrap_iterations, axis=1)
    delta_vars = np.var(bootstrap_iterations, axis=1)

    # Calculate alpha for the desired confidence level
    alpha = 1 - confidence_level
    
    # Use percentile method for CI
    ci_lower = np.percentile(bootstrap_iterations, 100 * alpha/2, axis=1)
    ci_upper = np.percentile(bootstrap_iterations, 100 * (1 - alpha/2), axis=1)

    # Calculate p-values based on how many bootstrap samples cross zero
    p_values = np.zeros(len(unique_combo_data))
    for j in range(len(unique_combo_data)):
        if delta_means[j] > 0:
            # For positive delta scores, count how many bootstrap samples are greater than the mean
            p_values[j] = np.sum(bootstrap_iterations[j, :] <=0) / n_bootstrap
        else:
            # For negative delta scores, count how many bootstrap samples are less than the mean
            p_values[j] = np.sum(bootstrap_iterations[j, :] >= 0) / n_bootstrap

        # Double the p-value for two-tailed test
        p_values[j] = min(p_values[j] * 2, 1.0)  # Ensure p-values are capped at 1.0
    
        # Set minimum p-value based on bootstrap sample size
        min_p_value = 1.0 / n_bootstrap
        p_values[j] = max(p_values[j], min_p_value)

    # Combine results with unique dose combinations
    result_df = unique_combo_data[[drug_col1, drug_col2]].copy()
    result_df['delta_score_mean'] = delta_means
    result_df['delta_score_var'] = delta_vars
    result_df['delta_score_lower'] = ci_lower
    result_df['delta_score_upper'] = ci_upper
    result_df['p_value'] = p_values

    result_df = test_delta_scores_significance(result_df, alpha=alpha, method='fdr_bh')

    return result_df, bootstrap_iterations


def test_delta_scores_significance(bootstrap_results, alpha=0.05, method='fdr_bh'):
    """
    Test if delta scores differ significantly from 0 and adjust for multiple testing.
    
    Parameters:
    bootstrap_results: DataFrame, output from bootstrap_delta_scores
    alpha: float, significance level
    method: str, multiple testing correction method (Default: 'fdr_bh' - Benjamini-Hochberg (non-negative) FDR)

    Returns:
    DataFrame with significance test results
    """
    # Extract p-values from bootstrap results
    p_values = bootstrap_results['p_value'].values

    # Adjust p-values for multiple testing
    rejected, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method=method)

    # Add adjusted p-values and significance to the results DataFrame
    results = bootstrap_results.copy()
    results['p_adjusted'] = p_adjusted
    results['significant_adjusted'] = rejected

    # Classify each combination as synergistic, antagonistic, or neutral
    conditions = [
        (results['significant_adjusted'] & (results['delta_score_mean'] > 0)),
        (results['significant_adjusted'] & (results['delta_score_mean'] < 0)),
        (~results['significant_adjusted'])
    ]
    choices = ['synergistic', 'antagonistic', 'neutral']
    results['effect_type'] = np.select(conditions, choices, default='neutral')
    
    return results
