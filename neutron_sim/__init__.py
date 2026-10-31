"""Monte Carlo simulation of thermal neutrons through shielding layers."""

from .analysis import AttenuationFit, TrialStatistics, fit_attenuation, run_trials, thickness_sweep
from .materials import MATERIALS, Material
from .transport import Layer, SimulationResult, simulate

__all__ = [
    'MATERIALS', 'Material', 'Layer', 'SimulationResult', 'simulate',
    'TrialStatistics', 'run_trials', 'thickness_sweep', 'AttenuationFit', 'fit_attenuation',
]
