"""Repeated trials, thickness sweeps and attenuation-length fits."""

from dataclasses import dataclass

import numpy as np

from .transport import ABSORBED, REFLECTED, TRANSMITTED, Layer, SimulationResult, simulate


@dataclass
class TrialStatistics:
    """
    Mean and standard error of the mean of the reflected, transmitted and absorbed
    fractions over repeated independent runs.

    'result' is the combined SimulationResult of all trials (trajectories, absorption
    depths, per-layer absorption).
    """
    n_neutrons: int
    n_trials: int
    mean: dict
    sem: dict
    result: SimulationResult


def run_trials(layers, n_neutrons, n_trials, rng=None, n_tracked=0):
    """
    Repeat a simulation and collect statistics on the fractions of each fate.

    Parameters:
    - layers (list of Layer): Layer stack.
    - n_neutrons (int): Neutrons per trial.
    - n_trials (int): Number of independent trials (at least 2).
    - rng (np.random.Generator, optional): Random number generator.
    - n_tracked (int): Number of neutron trajectories to record.

    Returns:
    - TrialStatistics
    """
    if n_trials < 2:
        raise ValueError('At least two trials are needed to estimate an error.')
    rng = np.random.default_rng() if rng is None else rng

    # All trials run as one batch (much faster), then neutrons are split into trials
    result = simulate(layers, n_neutrons * n_trials, rng, n_tracked)
    fates = result.fates.reshape(n_trials, n_neutrons)
    fractions = np.column_stack([(fates == fate).mean(axis=1) for fate in (REFLECTED, TRANSMITTED, ABSORBED)])
    # Sample standard deviation (ddof=1) divided by sqrt(n) is the error on the mean
    sem = fractions.std(axis=0, ddof=1) / np.sqrt(n_trials)
    keys = ('reflected', 'transmitted', 'absorbed')
    return TrialStatistics(n_neutrons, n_trials,
                           dict(zip(keys, fractions.mean(axis=0))), dict(zip(keys, sem)), result)


def thickness_sweep(material, thicknesses, n_neutrons, n_trials, rng=None):
    """
    Run trials for a single material at several thicknesses.

    Returns:
    - dict: 'thickness' plus, for each fate, '<fate>' (mean fractions) and
      '<fate>_sem' (errors on the mean), all as arrays matching the thicknesses.
    """
    rng = np.random.default_rng() if rng is None else rng
    results = {'thickness': np.asarray(thicknesses, dtype=float)}
    for fate in ('reflected', 'transmitted', 'absorbed'):
        results[fate] = np.zeros(len(thicknesses))
        results[fate + '_sem'] = np.zeros(len(thicknesses))

    for i, thickness in enumerate(thicknesses):
        stats = run_trials([Layer(material, thickness)], n_neutrons, n_trials, rng)
        for fate in ('reflected', 'transmitted', 'absorbed'):
            results[fate][i] = stats.mean[fate]
            results[fate + '_sem'][i] = stats.sem[fate]
    return results


@dataclass
class AttenuationFit:
    """
    Weighted linear fit of ln(T) = slope * L + intercept.

    The attenuation length is -1/slope. Points with zero transmission or zero error are
    excluded, which is recorded in 'used'.
    """
    slope: float
    intercept: float
    attenuation_length: float
    attenuation_length_error: float
    reduced_chi_squared: float
    used: np.ndarray


def fit_attenuation(thickness, transmitted, transmitted_sem):
    """
    Fit an exponential attenuation law to transmitted fractions.

    Parameters:
    - thickness (array): Slab thicknesses in cm.
    - transmitted (array): Mean transmitted fractions.
    - transmitted_sem (array): Errors on the mean transmitted fractions.

    Returns:
    - AttenuationFit, or None if fewer than three usable points remain.
    """
    thickness = np.asarray(thickness, dtype=float)
    transmitted = np.asarray(transmitted, dtype=float)
    transmitted_sem = np.asarray(transmitted_sem, dtype=float)

    used = (transmitted > 0) & (transmitted_sem > 0)
    if np.count_nonzero(used) < 3:
        return None

    L = thickness[used]
    log_t = np.log(transmitted[used])
    log_t_error = transmitted_sem[used] / transmitted[used]  # Error propagation for ln(T)

    # Weights are 1/sigma; 'unscaled' keeps the covariance based on the stated errors
    coefficients, covariance = np.polyfit(L, log_t, 1, w=1 / log_t_error, cov='unscaled')
    slope, intercept = coefficients
    slope_error = np.sqrt(covariance[0, 0])

    residuals = (log_t - np.polyval(coefficients, L)) / log_t_error
    degrees_of_freedom = len(L) - 2
    reduced_chi_squared = np.sum(residuals**2) / degrees_of_freedom if degrees_of_freedom > 0 else np.nan

    return AttenuationFit(
        slope=slope,
        intercept=intercept,
        attenuation_length=-1 / slope,
        attenuation_length_error=slope_error / slope**2,
        reduced_chi_squared=reduced_chi_squared,
        used=used,
    )
