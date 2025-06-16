import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt


def logistic_4PL(x, EC50, Hill, bottom=0, top=1):
    """
    4-parameter logistic function with bottom=0 and top=1 as defaults.
    
    Parameters:
    x: array-like, dose values
    EC50: float, dose at half-maximal effect (midpoint)
    Hill: float, Hill slope parameter
    bottom: float, minimum effect (usually 0)
    top: float, maximum effect (usually 1)
    
    Returns:
    y: array-like, effect values
    """
    # Ensure EC50 is positive to avoid division by zero
    EC50 = max(EC50, 1e-6)

    # Handle scalar vs array input
    if np.isscalar(x):
        if x == 0:
            return bottom
        return (bottom + top * (x/EC50)**Hill ) / (1 + (x/EC50)**Hill)
    else:
        # Handle array input
        result = np.zeros_like(x, dtype=float)
        mask_zero = (x == 0)
        mask_nonzero = ~mask_zero

        result[mask_zero] = bottom
        result[mask_nonzero] = (bottom + top * (x[mask_nonzero]/EC50)**Hill) / (1 + (x[mask_nonzero]/EC50)**Hill)
        return result


def calculate_r_squared(y_values, y_fit):
    """
    Calculate R-squared value for goodness of fit.
    
    Parameters:
    y_values: array-like, observed effect values
    y_fit: array-like, fitted effect values
    Returns:
    r_squared: float, R-squared value
    """
    # Ensure inputs have compatible shapes
    y_values = np.asarray(y_values).flatten()
    y_fit = np.asarray(y_fit).flatten()

    # Calculate residuals
    residuals = y_values - y_fit

    # Calculate residual sum of squares (ss_res)
    ss_res = np.sum(residuals**2)
    
    # Calculate total sum of squares (ss_tot), a measure of total variance
    ss_tot = np.sum((y_values - np.mean(y_values))**2)
        
    # Compute R-squared as 1 minus the ratio of ss_res to ss_tot
    # Handle the case where ss_tot is close to zero (constant y values)
    if ss_tot < 1e-10:
        return 0.0

    # Compute R-squared as 1 minus the ratio of ss_res to ss_tot
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def fit_monotherapy(data, drug_col, effect_col='inhibition', 
                    initial_guess=None, bounds=None):
    """
    Fit monotherapy data to logistic function.
    
    Parameters:
    data: DataFrame with dose and effect data
    drug_col: str, column name for drug dose
    effect_col: str, column name for effect (Default: 'inhibition')
    initial_guess: tuple or None, optional initial guesses for EC50 and Hill (Default: None)
    bounds: tuple, bounds for the parameters (Default: ([0, 1e-6], [1e5, 1e5]))

    Returns:
    dict: Dictionary with fitted parameters and metrics
        - 'EC50': float, fitted EC50 value
        - 'Hill': float, fitted Hill slope parameter
        - 'errors': array-like, standard deviations of the fitted parameters (derived from the diagonal of the covariance matrix)
        - 'r_squared': float, R-squared value for goodness of fit
        - 'covariance_matrix': array, covariance matrix of the fitted parameters
        - 'residuals': array, residuals of the fit (observed - fitted)
    """
    # Validate that the drug column contains positive values
    if not any(data[drug_col] > 0):
        raise ValueError(f"The column '{drug_col}' must contain positive values.")
    
    # Filter for non-zero doses
    nonzero_data = data[data[drug_col] > 0].copy()
    
    x_values = nonzero_data[drug_col].values
    y_values = nonzero_data[effect_col].values

    # Initial parameter guess
    # Use median of non-zero data for initial guess
    p0 = [np.median(x_values), 1] if initial_guess is None else initial_guess
    # Set bounds for EC50 and Hill slope
    if bounds is None:
        bounds = ([1e-6, 0.01], [1e5, 20])  
    else:
        bounds

    # Fit with fixed bottom=0, top=1
    params, covariance = curve_fit(
        lambda x, EC50, Hill: logistic_4PL(x, EC50, Hill), 
        x_values, y_values, p0=p0, bounds=bounds
    )
    
    EC50 = params[0]    
    errors = np.sqrt(np.diag(covariance))
    
    return {
        'EC50': EC50,
        'Hill': params[1],
        'errors': errors,
        'r_squared': calculate_r_squared(y_values, logistic_4PL(x_values, *params)),
        'covariance_matrix': covariance,
        'residuals': y_values - logistic_4PL(x_values, *params)
    }