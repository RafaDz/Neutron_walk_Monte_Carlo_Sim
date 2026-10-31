"""Monte Carlo transport of thermal neutrons through a stack of slab layers."""

from dataclasses import dataclass, field

import numpy as np

from .materials import Material

# Fate codes stored per neutron in SimulationResult.fates
REFLECTED, TRANSMITTED, ABSORBED = 0, 1, 2


@dataclass(frozen=True)
class Layer:
    """A slab of a material with a thickness in cm along x."""
    material: Material
    thickness: float

    def __post_init__(self):
        if self.thickness <= 0:
            raise ValueError('Layer thickness must be positive.')


@dataclass
class SimulationResult:
    """
    Outcome of one simulation run.

    Attributes:
    - n_neutrons (int): Number of neutrons simulated.
    - reflected, transmitted, absorbed (int): Neutron counts for each fate.
    - fates (np.ndarray): Fate of each neutron (REFLECTED, TRANSMITTED or ABSORBED).
    - absorbed_per_layer (np.ndarray): Absorbed neutrons in each layer.
    - absorption_depths (np.ndarray): x position (cm) of every absorption.
    - paths (list of np.ndarray): (n_points, 3) trajectories of the tracked neutrons.
    - collisions (int): Number of real collisions (scattering or absorption).
    - fictitious_collisions (int): Number of rejected (fictitious) collisions.
    """
    n_neutrons: int
    reflected: int
    transmitted: int
    absorbed: int
    fates: np.ndarray
    absorbed_per_layer: np.ndarray
    absorption_depths: np.ndarray
    paths: list = field(default_factory=list)
    collisions: int = 0
    fictitious_collisions: int = 0

    @property
    def fractions(self):
        """Fractions of reflected, transmitted and absorbed neutrons."""
        n = self.n_neutrons
        return self.reflected / n, self.transmitted / n, self.absorbed / n


def isotropic_directions(count, rng):
    """
    Generate unit vectors uniformly distributed over the sphere.

    The polar angle uses theta = arccos(1 - 2u), which avoids clustering at the poles.
    """
    phi = rng.uniform(0, 2 * np.pi, count)
    cos_theta = 1 - 2 * rng.uniform(0, 1, count)
    sin_theta = np.sqrt(1 - cos_theta**2)
    return np.column_stack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta))


def simulate(layers, n_neutrons, rng=None, n_tracked=0):
    """
    Simulate a beam of neutrons hitting a stack of layers at normal incidence.

    The neutrons enter at x = 0 moving in +x. Layers are stacked along x in the given order
    and are infinite in y and z. Scattering is isotropic and elastic, with no energy loss.

    Delta (Woodcock) tracking is used: flight distances are sampled with the largest total
    cross-section of the stack, and a collision at position x is real with probability
    Sigma(x) / Sigma_max, otherwise it is fictitious and the neutron carries on in the same
    direction. For a single layer every collision is real, which is ordinary analog tracking.

    Parameters:
    - layers (list of Layer): Layers in the order the beam meets them.
    - n_neutrons (int): Number of neutrons to simulate.
    - rng (np.random.Generator, optional): Random number generator.
    - n_tracked (int): Number of neutrons whose full trajectories are recorded.

    Returns:
    - SimulationResult
    """
    if not layers:
        raise ValueError('At least one layer is needed.')
    rng = np.random.default_rng() if rng is None else rng

    sigma_total = np.array([layer.material.sigma_total for layer in layers])
    p_absorption = np.array([layer.material.absorption_probability for layer in layers])
    layer_ends = np.cumsum([layer.thickness for layer in layers])
    total_thickness = layer_ends[-1]

    sigma_max = sigma_total.max()
    real_probability = sigma_total / sigma_max
    majorant_mean_free_path = 1 / sigma_max

    position = np.zeros((n_neutrons, 3))
    direction = np.tile([1.0, 0.0, 0.0], (n_neutrons, 1))
    ids = np.arange(n_neutrons)

    n_tracked = min(n_tracked, n_neutrons)
    paths = [[np.zeros(3)] for _ in range(n_tracked)]

    fates = np.empty(n_neutrons, dtype=np.int8)
    collisions = fictitious = 0
    absorbed_per_layer = np.zeros(len(layers), dtype=int)
    absorption_depths = []

    while len(ids) > 0:
        # Fly to the next (real or fictitious) collision site
        step = rng.exponential(majorant_mean_free_path, len(ids))
        position += step[:, None] * direction
        x = position[:, 0]

        if n_tracked:
            for k in np.flatnonzero(ids < n_tracked):
                paths[ids[k]].append(position[k].copy())

        # Neutrons leaving the stack
        is_reflected = x < 0
        is_transmitted = x > total_thickness
        fates[ids[is_reflected]] = REFLECTED
        fates[ids[is_transmitted]] = TRANSMITTED
        inside = ~(is_reflected | is_transmitted)

        # Collision handling for neutrons still inside
        layer_index = np.minimum(np.searchsorted(layer_ends, x), len(layers) - 1)
        is_real = inside & (rng.uniform(size=len(ids)) < real_probability[layer_index])
        is_absorbed = is_real & (rng.uniform(size=len(ids)) < p_absorption[layer_index])
        is_scattered = is_real & ~is_absorbed

        fates[ids[is_absorbed]] = ABSORBED
        collisions += np.count_nonzero(is_real)
        fictitious += np.count_nonzero(inside & ~is_real)
        absorbed_per_layer += np.bincount(layer_index[is_absorbed], minlength=len(layers))
        absorption_depths.append(x[is_absorbed])

        # Scattered neutrons get a new direction; fictitious collisions keep theirs
        direction[is_scattered] = isotropic_directions(np.count_nonzero(is_scattered), rng)

        alive = inside & ~is_absorbed
        position, direction, ids = position[alive], direction[alive], ids[alive]

    return SimulationResult(
        n_neutrons=n_neutrons,
        reflected=int(np.count_nonzero(fates == REFLECTED)),
        transmitted=int(np.count_nonzero(fates == TRANSMITTED)),
        absorbed=int(np.count_nonzero(fates == ABSORBED)),
        fates=fates,
        absorbed_per_layer=absorbed_per_layer,
        absorption_depths=np.concatenate(absorption_depths),
        paths=[np.array(path) for path in paths],
        collisions=collisions,
        fictitious_collisions=fictitious,
    )
