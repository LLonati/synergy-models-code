"""Statistical power analysis and bootstrap diagnostics for synergy models.

Provides three self-contained capabilities:
1. Bootstrap convergence diagnostics (CI width vs. iteration count)
2. Empirical CI coverage validation on synthetic data
3. Minimum detectable effect (MDE) tables

All functions are stateless — inputs are numpy arrays / DataFrames,
outputs are DataFrames and matplotlib Figures.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import logging
from typing import cast

logger = logging.getLogger(__name__)


def _as_float(value) -> float:
    """Coerce scalar-ish pandas/numpy values to float for numeric helpers."""
    return float(np.asarray(value).item())

# ---------------------------------------------------------------------------
# 1. Bootstrap convergence diagnostics
# ---------------------------------------------------------------------------

def convergence_diagnostics(bootstrap_iterations, dose_pairs_df,
                             confidence_level=0.95, step_size=50):
    """Compute CI width as a function of bootstrap iteration count.

    Parameters
    ----------
    bootstrap_iterations : ndarray, shape (n_combos, n_bootstrap)
        Raw per-iteration deviation values from ``bootstrap_delta_scores`` or
        ``bootstrap_model_scores``.
    dose_pairs_df : DataFrame
        Must contain 'dose_E' and 'dose_X' columns (one row per combo,
        matching the row order of ``bootstrap_iterations``).
    confidence_level : float
        Target CI level (default 0.95).
    step_size : int
        Evaluate diagnostics every ``step_size`` iterations (default 50).

    Returns
    -------
    DataFrame with columns:
        dose_E, dose_X, n_iterations, mean, ci_lower, ci_upper, ci_width
    """
    n_combos, n_bootstrap = bootstrap_iterations.shape
    if len(dose_pairs_df) != n_combos:
        raise ValueError(
            f"dose_pairs_df has {len(dose_pairs_df)} rows but bootstrap_iterations "
            f"has {n_combos} — they must match."
        )

    alpha = 1 - confidence_level
    checkpoints = list(range(step_size, n_bootstrap + 1, step_size))
    if not checkpoints or checkpoints[-1] < n_bootstrap:
        checkpoints.append(n_bootstrap)

    dose_e = dose_pairs_df['dose_E'].values
    dose_x = dose_pairs_df['dose_X'].values

    records = []
    for k in checkpoints:
        subset = bootstrap_iterations[:, :k]
        means = np.nanmean(subset, axis=1)
        lowers = np.nanpercentile(subset, 100 * alpha / 2, axis=1)
        uppers = np.nanpercentile(subset, 100 * (1 - alpha / 2), axis=1)
        widths = uppers - lowers
        for j in range(n_combos):
            records.append({
                'dose_E': dose_e[j],
                'dose_X': dose_x[j],
                'n_iterations': k,
                'mean': means[j],
                'ci_lower': lowers[j],
                'ci_upper': uppers[j],
                'ci_width': widths[j],
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Empirical CI coverage test
# ---------------------------------------------------------------------------

def empirical_ci_coverage(true_params_E, true_params_X, true_params_shifts=None,
                           n_simulations=200, n_bootstrap=500,
                           noise_level=0.05, confidence_level=0.95,
                           n_doses=6, synergy_strength=1.5,
                           drug_col1='dose_E', drug_col2='dose_X'):
    """Estimate empirical CI coverage via simulation on synthetic data.

    For each simulation, fresh noisy data are generated with known true
    parameters, monotherapy curves are fitted, ``bootstrap_delta_scores``
    is run, and the oracle delta (computed on noiseless data from true
    params) is checked for containment inside the bootstrap CI.

    Parameters
    ----------
    true_params_E, true_params_X : dict
        Must contain 'EC50' and 'Hill'. Covariance matrices are estimated
        from the first simulation's fit and reused (conservative approach).
    true_params_shifts : dict or None
        If None, potency shifts are re-fitted per simulation.
    n_simulations : int
        Number of independent datasets to generate (default 200).
    n_bootstrap : int
        Bootstrap iterations per simulation (default 500).
    noise_level : float
        Gaussian noise std applied to inhibition values.
    confidence_level : float
        Target CI level.
    n_doses : int
        Passed to the data generator (default 6).
    synergy_strength : float
        Controls magnitude of true synergy in generated data (default 1.5).
    drug_col1, drug_col2 : str
        Column names for the two drugs.

    Returns
    -------
    coverage_df : DataFrame
        Columns: dose_E, dose_X, coverage_rate, n_simulations
    summary : dict
        Keys: mean_coverage, min_coverage, max_coverage, target, n_simulations
    """
    # Import here to avoid circular dependency; these live in sibling modules
    from src.monotherapy import fit_monotherapy, logistic_4PL
    from src.synergy import (bootstrap_delta_scores, calculate_delta_scores,
                              get_potency_shifts, potency_shift_model)

    # --- data generator (same logic as generate_mock_data in tests) ---
    def _make_data(seed):
        rng = np.random.default_rng(seed)
        doses_E_vals = np.array([0, 0.1, 0.3, 1, 3, 10])[:n_doses]
        doses_X_vals = np.array([0, 0.5, 1, 2, 4, 8])[:n_doses]
        pairs = [(e, x) for e in doses_E_vals for x in doses_X_vals]
        df = pd.DataFrame(pairs, columns=[drug_col1, drug_col2])

        def _y1(d):
            dose: float = _as_float(d)
            return 0.0 if dose == 0 else _as_float(logistic_4PL(dose, true_params_E['EC50'], true_params_E['Hill']))

        def _y2(d):
            dose: float = _as_float(d)
            return 0.0 if dose == 0 else _as_float(logistic_4PL(dose, true_params_X['EC50'], true_params_X['Hill']))

        df['y1'] = df[drug_col1].map(_y1)
        df['y2'] = df[drug_col2].map(_y2)

        shifts = true_params_shifts if true_params_shifts is not None else {
            'E_to_X': {'EC50': 0.5, 'Hill': 1.3},
            'X_to_E': {'EC50': 5.0, 'Hill': 1.1},
        }

        def _true_obs(row):
            if row[drug_col1] == 0 or row[drug_col2] == 0:
                return max(row['y1'], row['y2'])
            y_c1 = potency_shift_model(row[drug_col1], shifts['E_to_X']['EC50'],
                                        shifts['E_to_X']['Hill'], row['y2'])
            y_c2 = potency_shift_model(row[drug_col2], shifts['X_to_E']['EC50'],
                                        shifts['X_to_E']['Hill'], row['y1'])
            return min(1.0, (y_c1 + y_c2) / 2 + synergy_strength * row['y1'] * row['y2'])

        df['true_effect'] = df.apply(_true_obs, axis=1)
        noise = rng.normal(0, noise_level, size=len(df))
        df['inhibition'] = np.clip(df['true_effect'] + noise, 0, 1)
        return df

    # Oracle: noiseless delta scores computed from true parameters
    data_oracle = _make_data(seed=0)
    data_oracle['inhibition'] = data_oracle['true_effect']
    mono_E_oracle = data_oracle[data_oracle[drug_col2] == 0]
    mono_X_oracle = data_oracle[data_oracle[drug_col1] == 0]
    params_E_oracle = fit_monotherapy(mono_E_oracle, drug_col1, 'inhibition')
    params_X_oracle = fit_monotherapy(mono_X_oracle, drug_col2, 'inhibition')
    shifts_oracle = get_potency_shifts(data_oracle, params_E_oracle, params_X_oracle,
                                       drug_col1, drug_col2)
    oracle_deltas = calculate_delta_scores(data_oracle, params_E_oracle, params_X_oracle, shifts_oracle)
    combo_mask = (oracle_deltas[drug_col1] > 0) & (oracle_deltas[drug_col2] > 0)
    oracle_combos = oracle_deltas[combo_mask][[drug_col1, drug_col2, 'delta_score']].drop_duplicates(
        [drug_col1, drug_col2]).reset_index(drop=True)
    n_combos = len(oracle_combos)

    # Accumulate containment counts
    contained = np.zeros(n_combos, dtype=int)
    valid_sims = 0

    for sim in range(n_simulations):
        try:
            data = _make_data(seed=sim + 1)
            mono_E = data[data[drug_col2] == 0]
            mono_X = data[data[drug_col1] == 0]
            params_E = fit_monotherapy(mono_E, drug_col1, 'inhibition')
            params_X = fit_monotherapy(mono_X, drug_col2, 'inhibition')

            if 'covariance_matrix' not in params_E or 'covariance_matrix' not in params_X:
                logger.warning(f"Simulation {sim}: monotherapy fit failed, skipping.")
                continue

            boot_df, _ = bootstrap_delta_scores(
                data, params_E, params_X,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                drug_col1=drug_col1, drug_col2=drug_col2
            )
            boot_df = cast(pd.DataFrame, boot_df)

            for j, row in enumerate(oracle_combos.itertuples(index=False)):
                dose1 = getattr(row, drug_col1)
                dose2 = getattr(row, drug_col2)
                true_delta = row.delta_score
                match = boot_df[(boot_df[drug_col1] == dose1) &
                                (boot_df[drug_col2] == dose2)]
                if len(match) == 0:
                    continue
                lo = match['delta_score_lower'].values[0]
                hi = match['delta_score_upper'].values[0]
                if lo <= true_delta <= hi:
                    contained[j] += 1

            valid_sims += 1

        except Exception as e:
            logger.warning(f"Simulation {sim} failed: {e}")
            continue

    if valid_sims == 0:
        raise RuntimeError("All simulations failed — check input parameters.")

    coverage_rates = contained / valid_sims
    coverage_df = oracle_combos.copy()
    coverage_df['coverage_rate'] = coverage_rates
    coverage_df['n_simulations'] = valid_sims
    coverage_df = coverage_df.drop(columns='delta_score')

    summary = {
        'mean_coverage': float(np.mean(coverage_rates)),
        'min_coverage': float(np.min(coverage_rates)),
        'max_coverage': float(np.max(coverage_rates)),
        'target': confidence_level,
        'n_simulations': valid_sims,
        'n_bootstrap': n_bootstrap,
    }
    return coverage_df, summary


# ---------------------------------------------------------------------------
# 3. Minimum detectable effect
# ---------------------------------------------------------------------------

def minimum_detectable_effect(n_bootstrap=1000, power_target=0.80, alpha=0.05,
                               n_simulations=200, effect_sizes=None,
                               noise_level=0.05, n_doses=6,
                               drug_col1='dose_E', drug_col2='dose_X'):
    """Estimate minimum detectable effect (MDE) for the bootstrap test.

    For each candidate effect size (expressed as ``synergy_strength`` units,
    which map to delta-score magnitude), ``n_simulations`` synthetic datasets
    are generated and tested; the detected fraction (empirical power) is
    recorded. The MDE is the smallest effect size at which the detected
    fraction exceeds ``power_target``.

    Parameters
    ----------
    n_bootstrap : int
        Bootstrap iterations per simulation (default 1000).
    power_target : float
        Desired power level (default 0.80).
    alpha : float
        Significance level after BH-FDR correction (default 0.05).
    n_simulations : int
        Simulations per effect-size point (default 200).
    effect_sizes : array-like or None
        Synergy strength values to sweep. Default:
        ``np.arange(0.1, 2.0, 0.2)``.
    noise_level : float
        Gaussian noise std (default 0.05).
    n_doses : int
        Number of dose levels per drug (default 6).
    drug_col1, drug_col2 : str
        Column names for the two drugs.

    Returns
    -------
    mde_df : DataFrame
        Columns: effect_size, detected_fraction, is_mde, n_bootstrap,
                 alpha, power_target
    """
    from src.monotherapy import fit_monotherapy, logistic_4PL
    from src.synergy import bootstrap_delta_scores, potency_shift_model

    if effect_sizes is None:
        effect_sizes = np.arange(0.1, 2.0, 0.2)
    effect_sizes = np.asarray(effect_sizes)

    true_params_E = {'EC50': 0.2, 'Hill': 1.5}
    true_params_X = {'EC50': 3.0, 'Hill': 1.2}
    true_shifts = {
        'E_to_X': {'EC50': 0.5, 'Hill': 1.3},
        'X_to_E': {'EC50': 5.0, 'Hill': 1.1},
    }

    doses_E_vals = np.array([0, 0.1, 0.3, 1, 3, 10])[:n_doses]
    doses_X_vals = np.array([0, 0.5, 1, 2, 4, 8])[:n_doses]

    def _make_data(seed, synergy_strength):
        rng = np.random.default_rng(seed)
        pairs = [(e, x) for e in doses_E_vals for x in doses_X_vals]
        df = pd.DataFrame(pairs, columns=[drug_col1, drug_col2])

        def _y1(d):
            dose: float = _as_float(d)
            return 0.0 if dose == 0 else _as_float(logistic_4PL(dose, true_params_E['EC50'], true_params_E['Hill']))

        def _y2(d):
            dose: float = _as_float(d)
            return 0.0 if dose == 0 else _as_float(logistic_4PL(dose, true_params_X['EC50'], true_params_X['Hill']))

        df['y1'] = df[drug_col1].map(_y1)
        df['y2'] = df[drug_col2].map(_y2)

        def _true_obs(row):
            if row[drug_col1] == 0 or row[drug_col2] == 0:
                return max(row['y1'], row['y2'])
            y_c1 = potency_shift_model(row[drug_col1], true_shifts['E_to_X']['EC50'],
                                        true_shifts['E_to_X']['Hill'], row['y2'])
            y_c2 = potency_shift_model(row[drug_col2], true_shifts['X_to_E']['EC50'],
                                        true_shifts['X_to_E']['Hill'], row['y1'])
            return min(1.0, (y_c1 + y_c2) / 2 + synergy_strength * row['y1'] * row['y2'])

        df['true_effect'] = df.apply(_true_obs, axis=1)
        noise = rng.normal(0, noise_level, size=len(df))
        df['inhibition'] = np.clip(df['true_effect'] + noise, 0, 1)
        return df

    records = []
    for es in effect_sizes:
        n_detected = 0
        n_valid = 0

        for sim in range(n_simulations):
            try:
                data = _make_data(seed=sim, synergy_strength=float(es))
                mono_E = data[data[drug_col2] == 0]
                mono_X = data[data[drug_col1] == 0]
                params_E = fit_monotherapy(mono_E, drug_col1, 'inhibition')
                params_X = fit_monotherapy(mono_X, drug_col2, 'inhibition')

                if 'covariance_matrix' not in params_E or 'covariance_matrix' not in params_X:
                    continue

                boot_df, _ = bootstrap_delta_scores(
                    data, params_E, params_X,
                    n_bootstrap=n_bootstrap,
                    confidence_level=1 - alpha,
                    drug_col1=drug_col1, drug_col2=drug_col2
                )
                boot_df = cast(pd.DataFrame, boot_df)

                # "Detected" if any dose pair shows significant synergy after FDR
                n_sig = boot_df['significant_adjusted'].sum()
                if n_sig > 0:
                    n_detected += 1
                n_valid += 1

            except Exception as e:
                logger.debug(f"MDE sim {sim} effect={es:.3f} failed: {e}")
                continue

        detected_fraction = n_detected / n_valid if n_valid > 0 else np.nan
        records.append({
            'effect_size': float(es),
            'detected_fraction': detected_fraction,
            'n_valid_sims': n_valid,
            'n_bootstrap': n_bootstrap,
            'alpha': alpha,
            'power_target': power_target,
        })
        logger.info(f"MDE sweep: effect={es:.3f}, detected={detected_fraction:.3f} ({n_valid} valid sims)")

    mde_df = pd.DataFrame(records)
    # Tag smallest effect size at or above power target (after smoothing NaN)
    above_target = mde_df[mde_df['detected_fraction'] >= power_target]
    mde_value = above_target['effect_size'].min() if len(above_target) > 0 else np.nan
    mde_df['is_mde'] = mde_df['effect_size'] == mde_value
    return mde_df


# ---------------------------------------------------------------------------
# 4. Visualisation helpers
# ---------------------------------------------------------------------------

def plot_convergence_diagnostics(convergence_df, dose_pairs=None, figsize=(14, 5),
                                  confidence_level=0.95):
    """Plot CI width and CI bounds vs. bootstrap iteration count.

    Parameters
    ----------
    convergence_df : DataFrame
        Output of ``convergence_diagnostics``.
    dose_pairs : list of (dose_E, dose_X) tuples, or None
        Subset of dose pairs to highlight. If None, all are shown.
    figsize : tuple
    confidence_level : float, used for axis labels only.

    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    all_pairs = convergence_df[['dose_E', 'dose_X']].drop_duplicates().values.tolist()
    if dose_pairs is not None:
        pairs_to_plot = dose_pairs
    else:
        pairs_to_plot = all_pairs

    ax_width, ax_ci = axes

    ci_pct = int(confidence_level * 100)

    # Panel 1: CI width vs N
    for de, dx in pairs_to_plot:
        sub = convergence_df[(convergence_df['dose_E'] == de) & (convergence_df['dose_X'] == dx)]
        ax_width.plot(sub['n_iterations'], sub['ci_width'],
                      color='steelblue', alpha=0.3, linewidth=1)

    # Bold mean across all pairs
    mean_width = convergence_df.groupby('n_iterations')['ci_width'].mean()
    ax_width.plot(mean_width.index, mean_width.values,
                  color='darkblue', linewidth=2.5, label='Mean across dose pairs')
    ax_width.set_xlabel('Bootstrap iterations')
    ax_width.set_ylabel(f'{ci_pct}% CI width')
    ax_width.set_title('CI width convergence')
    ax_width.legend(fontsize=9)

    # Panel 2: Mean ± CI width at final N (one bar per dose pair)
    final_n = convergence_df['n_iterations'].max()
    final = convergence_df[convergence_df['n_iterations'] == final_n].reset_index(drop=True)
    labels = [f'{row.dose_E}/{row.dose_X}' for _, row in final.iterrows()]
    x = np.arange(len(final))
    ax_ci.bar(x, final['ci_width'], color='steelblue', alpha=0.7)
    ax_ci.set_xticks(x)
    ax_ci.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax_ci.set_ylabel(f'{ci_pct}% CI width at N={final_n}')
    ax_ci.set_title(f'CI width per dose pair at N={final_n}')

    fig.tight_layout()
    return fig


def plot_mde_curve(mde_df, figsize=(8, 5)):
    """Plot empirical power (detected fraction) vs. effect size with MDE annotation.

    Parameters
    ----------
    mde_df : DataFrame
        Output of ``minimum_detectable_effect``.
    figsize : tuple

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    power_target = mde_df['power_target'].iloc[0]
    alpha = mde_df['alpha'].iloc[0]
    n_bootstrap = mde_df['n_bootstrap'].iloc[0]

    ax.plot(mde_df['effect_size'], mde_df['detected_fraction'],
            marker='o', color='steelblue', linewidth=2, markersize=6, label='Empirical power')
    ax.axhline(power_target, color='tomato', linestyle='--', linewidth=1.5,
               label=f'Power target = {power_target:.0%}')

    mde_row = mde_df[mde_df['is_mde']]
    if len(mde_row) > 0:
        mde_val = mde_row['effect_size'].values[0]
        ax.axvline(mde_val, color='darkorange', linestyle=':', linewidth=1.5,
                   label=f'MDE = {mde_val:.3f}')
        ax.annotate(f'MDE={mde_val:.3f}',
                    xy=(mde_val, power_target),
                    xytext=(mde_val + 0.02, power_target - 0.08),
                    fontsize=9, color='darkorange',
                    arrowprops=dict(arrowstyle='->', color='darkorange'))

    ax.set_xlabel('Effect size (synergy strength)')
    ax.set_ylabel('Detected fraction (empirical power)')
    ax.set_title(f'Minimum Detectable Effect  |  N={n_bootstrap} bootstrap, α={alpha}')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_ci_coverage(coverage_df, summary, figsize=(10, 4)):
    """Bar chart of per-dose-pair CI coverage rates with the target line.

    Parameters
    ----------
    coverage_df : DataFrame
        Output of ``empirical_ci_coverage``.
    summary : dict
        Output of ``empirical_ci_coverage``.
    figsize : tuple

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    labels = [f'{row.dose_E}/{row.dose_X}' for _, row in coverage_df.iterrows()]
    x = np.arange(len(coverage_df))

    bars = ax.bar(x, coverage_df['coverage_rate'], color='steelblue', alpha=0.75)
    ax.axhline(summary['target'], color='tomato', linestyle='--', linewidth=1.5,
               label=f'Target = {summary["target"]:.0%}')
    ax.axhline(summary['mean_coverage'], color='navy', linestyle='-', linewidth=1.2,
               label=f'Mean = {summary["mean_coverage"]:.3f}')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Coverage rate')
    ax.set_ylim(0, 1.1)
    ax.set_title(f'Empirical CI coverage  |  {summary["n_simulations"]} simulations, '
                 f'N={summary["n_bootstrap"]} bootstrap')
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
