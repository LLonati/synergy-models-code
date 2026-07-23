import numpy as np
import pandas as pd
from src.monotherapy import logistic_4PL, calculate_r_squared, fit_monotherapy
from scipy.optimize import curve_fit
from scipy.optimize import brentq
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

def calculate_bliss_effect(dose1, dose2, params1, params2):
    """Calculate expected effect based on Bliss independence.
    This is equivalent to the ZIP baseline term before potency-shift corrections.
    """
    return calculate_zip_effect(dose1, dose2, params1, params2)


def calculate_hsa_effect(dose1, dose2, params1, params2):
    """Calculate expected effect based on HSA (Highest Single Agent)."""
    y1 = 0 if dose1 == 0 else logistic_4PL(dose1, params1['EC50'], params1['Hill'])
    y2 = 0 if dose2 == 0 else logistic_4PL(dose2, params2['EC50'], params2['Hill'])
    return max(y1, y2)


def logistic_inverse_effect(effect, params, eps=1e-9):
    """Return dose that gives a target effect for 4PL modelwith bottom=0, top=1."""
    ec50 = params.get('EC50')
    hill = params.get('Hill')
    if ec50 is None or hill is None:
        raise ValueError("params must contain 'EC50' and 'Hill'")
    if not np.isfinite(ec50) or not np.isfinite(hill) or ec50 <= 0 or hill <= 0:
        raise ValueError("EC50 and Hill must be finite positive values")
    
    clipped = np.clip(effect, eps, 1 - eps)
    return ec50 * (clipped / (1 - clipped)) ** (1.0 / hill)


def calculate_loewe_effect(dose1, dose2, params1, params2, eps=1e-6):
    """Calculate expected effect under Loewe additivity using an implicit solve.
    For dose pair (d1,d2), solve y in:
       d1 / D1(y) + d2 / D2(y) = 1
    where Di(y) is inverse monotherapy dose-effect relation.
    """
    if dose1 == 0 and dose2 == 0:
        return 0.0
    if dose1 == 0:
        return logistic_4PL(dose2, params2['EC50'], params2['Hill'])
    if dose2 == 0:
        return logistic_4PL(dose1, params1['EC50'], params1['Hill'])
    
    def loewe_residual(effect):
        d1_effect = logistic_inverse_effect(effect, params1, eps=eps)
        d2_effect = logistic_inverse_effect(effect, params2, eps=eps)
        return (dose1 / d1_effect) + (dose2 / d2_effect) - 1.0
    
    lower = eps
    upper = 1 - eps
    r_low = loewe_residual(lower)
    r_high = loewe_residual(upper)

    if np.isnan(r_low) or np.isnan(r_high):
        raise ValueError("Invalid Loewe residual values at bounds.")
    
    if r_low == 0:
        return lower
    if r_high == 0:
        return upper
    
    if r_low * r_high > 0:
        raise ValueError(
            "Could not bracket a Loewe solution for the provided dose pair. "
            f"Residual at lower bound: {r_low}, residual at upper bound: {r_high}."
        )
    
    return brentq(loewe_residual, lower, upper)


def calculate_expected_effect(dose1, dose2, params1, params2, model='zip'):    
    """Dispatch expected-effect calculation across supported reference models."""
    model_key = model.lower()
    if model_key == 'zip':
        return calculate_zip_effect(dose1, dose2, params1, params2)
    if model_key == 'bliss':
        return calculate_bliss_effect(dose1, dose2, params1, params2)
    if model_key == 'hsa':
        return calculate_hsa_effect(dose1, dose2, params1, params2)
    if model_key == 'loewe':    
        return calculate_loewe_effect(dose1, dose2, params1, params2)
    raise ValueError(f"Unsupported model '{model}'. Supported: zip, bliss, hsa, loewe")


def calculate_model_scores(data, params_drug1, params_drug2, model='zip',
                           drug_col1='dose_E', drug_col2='dose_X',
                           observed_col='inhibition'):
    """Create standardized per-dose model scores for a given model.
    Output columns:
      - expected_effect_{model}
      - deviation_{model} = observed - expected
      - effect_direction_{model}
    """
    if observed_col not in data.columns:
        raise ValueError(f"Missing observed column '{observed_col}' in data")
    
    results = data.copy()
    model_key = model.lower()
    expected_col = f'expected_effect_{model_key}'
    deviation_col = f'deviation_{model_key}'
    direction_col = f'effect_direction_{model_key}'

    combo_mask = (results[drug_col1] > 0) & (results[drug_col2] > 0)
    results[expected_col] = np.nan

    def _expected(row):
        return calculate_expected_effect(
            row[drug_col1], row[drug_col2], params_drug1, params_drug2, model=model_key
        )
    if combo_mask.any():
        results.loc[combo_mask, expected_col] = results.loc[combo_mask].apply(_expected, axis=1)

    results[deviation_col] = np.nan
    results.loc[combo_mask, deviation_col] = (
        results.loc[combo_mask, observed_col] - results.loc[combo_mask, expected_col]
    )

    results[direction_col] = 'neutral'
    results.loc[combo_mask & (results[deviation_col] > 0), direction_col] = 'synergistic'
    results.loc[combo_mask & (results[deviation_col] < 0), direction_col] = 'antagonistic'

    return results


def classify_model_consensus(model_score_df, models=('zip', 'bliss', 'hsa', 'loewe'),
                             consensus_rule='majority'):
    """Classify consensus across multiple model-specific effect directions.

    consensus_rule:
      - 'majority': class with highest count among {synergistic, antagonistic, neutral}
      - 'unanimous': only if all non-neutral classes match, else neutral
    """
    results = model_score_df.copy()
    direction_cols = [f'effect_direction_{m.lower()}' for m in models]
    missing = [c for c in direction_cols if c not in results.columns]
    if missing:
        raise ValueError(f"Missing direction columns for consensus: {missing}")
    
    def _consensus_row(row):
        labels = [row[c] for c in direction_cols]
        counts = {
            'synergistic': labels.count('synergistic'),
            'antagonistic': labels.count('antagonistic'),
            'neutral': labels.count('neutral')
        }
        if consensus_rule == 'majority':
            max_count = max(counts.values())
            winners = [k for k, v in counts.items() if v == max_count]
            return winners[0] if len(winners) == 1 else 'neutral'
        if consensus_rule == 'unanimous':
            non_neutral = [lab for lab in labels if lab != 'neutral']
            if len(non_neutral) == len(labels) and len(set(non_neutral)) == 1:
                return non_neutral[0]
            return 'neutral'
        raise ValueError("consensus_rule must be 'majority' or 'unanimous'")
    
    results['consensus_effect_type'] = results.apply(_consensus_row, axis=1)
    return results


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


def bootstrap_model_scores(data, params_drug1, params_drug2, models=('zip', 'bliss', 'hsa', 'loewe'),
                           n_bootstrap=1000, confidence_level=0.95,
                           drug_col1='dose_E', drug_col2='dose_X'):
    """Bootstrap uncertainty estimation for multi-model synergy scoring.

    For each model in *models*, samples monotherapy parameters from their
    estimated covariance distributions, recalculates potency shifts (ZIP/Bliss
    only) and the per-model deviation for every unique combination dose pair,
    and returns mean ± CI per dose pair per model.

    Parameters
    ----------
    data : DataFrame
        Experimental data (must contain drug_col1, drug_col2, 'inhibition').
    params_drug1 : dict
        Fitted monotherapy parameters for drug 1, incl. 'covariance_matrix'.
    params_drug2 : dict
        Fitted monotherapy parameters for drug 2, incl. 'covariance_matrix'.
    models : tuple of str
        Any subset of ('zip', 'bliss', 'hsa', 'loewe').
    n_bootstrap : int
        Number of bootstrap iterations.
    confidence_level : float
        Confidence level for percentile CI (default 0.95).
    drug_col1, drug_col2 : str
        Column names for the two drugs.

    Returns
    -------
    summary_df : DataFrame
        One row per unique combination dose pair, one set of columns per model:
        {model}_mean, {model}_var, {model}_lower, {model}_upper,
        {model}_p_value, {model}_p_adjusted, {model}_significant, {model}_effect_type
    bootstrap_arrays : dict
        Raw bootstrap deviations: { model: ndarray (n_combinations, n_bootstrap) }
    """
    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_bootstrap), desc="Multi-model bootstrap")
    except ImportError:
        iterator = range(n_bootstrap)

    for p, name in [(params_drug1, 'params_drug1'), (params_drug2, 'params_drug2')]:
        if 'covariance_matrix' not in p:
            raise ValueError(f"Missing covariance_matrix in {name}")

    combo_mask = (data[drug_col1] > 0) & (data[drug_col2] > 0)
    combo_data = data[combo_mask].copy()
    if len(combo_data) == 0:
        raise ValueError("No combination data found in dataset.")

    unique_combos = combo_data.drop_duplicates([drug_col1, drug_col2])
    n_combos = len(unique_combos)

    params1_mean = np.array([params_drug1['EC50'], params_drug1['Hill']])
    params2_mean = np.array([params_drug2['EC50'], params_drug2['Hill']])
    cov1 = params_drug1['covariance_matrix']
    cov2 = params_drug2['covariance_matrix']

    # Raw iteration arrays: deviations (observed - expected) per model
    boot_arrays = {m: np.zeros((n_combos, n_bootstrap)) for m in models}

    for i in iterator:
        p1_samp = np.random.multivariate_normal(params1_mean, cov1)
        p2_samp = np.random.multivariate_normal(params2_mean, cov2)
        p1 = {'EC50': p1_samp[0], 'Hill': p1_samp[1], 'covariance_matrix': cov1}
        p2 = {'EC50': p2_samp[0], 'Hill': p2_samp[1], 'covariance_matrix': cov2}

        # Potency shifts needed for ZIP/Bliss observed model
        try:
            ps = get_potency_shifts(data, p1, p2, drug_col1, drug_col2)
        except Exception:
            ps = None

        for j, (_, row) in enumerate(unique_combos.iterrows()):
            d1, d2 = row[drug_col1], row[drug_col2]
            # Observed effect: mean of replicates for this dose pair
            obs = data[(data[drug_col1] == d1) & (data[drug_col2] == d2)]['inhibition'].mean()

            for m in models:
                try:
                    if m in ('zip', 'bliss'):
                        exp = calculate_expected_effect(d1, d2, p1, p2, model=m)
                        if ps is not None:
                            # Use ZIP potency-shift observed model deviation (delta score)
                            y_c_1to2 = potency_shift_model(d1, ps['E_to_X']['EC50'],
                                                           ps['E_to_X']['Hill'],
                                                           logistic_4PL(d2, p2['EC50'], p2['Hill']))
                            y_c_2to1 = potency_shift_model(d2, ps['X_to_E']['EC50'],
                                                           ps['X_to_E']['Hill'],
                                                           logistic_4PL(d1, p1['EC50'], p1['Hill']))
                            y_obs_model = (y_c_1to2 + y_c_2to1) / 2
                            deviation = y_obs_model - exp
                        else:
                            deviation = obs - exp
                    else:
                        exp = calculate_expected_effect(d1, d2, p1, p2, model=m)
                        deviation = obs - exp
                    boot_arrays[m][j, i] = deviation
                except Exception:
                    boot_arrays[m][j, i] = np.nan

    alpha = 1 - confidence_level
    result_df = unique_combos[[drug_col1, drug_col2]].copy().reset_index(drop=True)

    for m in models:
        arr = boot_arrays[m]
        means = np.nanmean(arr, axis=1)
        vrs = np.nanvar(arr, axis=1)
        lowers = np.nanpercentile(arr, 100 * alpha / 2, axis=1)
        uppers = np.nanpercentile(arr, 100 * (1 - alpha / 2), axis=1)

        p_values = np.zeros(n_combos)
        min_p = 1.0 / n_bootstrap
        for j in range(n_combos):
            col = arr[j, :]
            col = col[~np.isnan(col)]
            if len(col) == 0:
                p_values[j] = 1.0
                continue
            if means[j] > 0:
                p_values[j] = max(min(np.sum(col <= 0) / len(col) * 2, 1.0), min_p)
            else:
                p_values[j] = max(min(np.sum(col >= 0) / len(col) * 2, 1.0), min_p)

        _, p_adj, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        sig = p_adj < alpha
        conditions = [
            (sig & (means > 0)),
            (sig & (means < 0)),
            ~sig
        ]
        choices = ['synergistic', 'antagonistic', 'neutral']
        effect = np.select(conditions, choices, default='neutral')

        result_df[f'{m}_mean'] = means
        result_df[f'{m}_var'] = vrs
        result_df[f'{m}_lower'] = lowers
        result_df[f'{m}_upper'] = uppers
        result_df[f'{m}_p_value'] = p_values
        result_df[f'{m}_p_adjusted'] = p_adj
        result_df[f'{m}_significant'] = sig
        result_df[f'{m}_effect_type'] = effect

    return result_df, boot_arrays


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
