"""Shielding materials and their thermal-neutron cross-sections."""

from dataclasses import dataclass

import scipy.constants as const

AVOGADRO_NUMBER = const.Avogadro  # particles/mol
BARN_TO_CM2 = 1e-24  # 1 barn = 1e-24 cm^2


@dataclass(frozen=True)
class Material:
    """
    A homogeneous shielding material.

    Microscopic cross-sections are given per formula unit (molecule or atom)
    for thermal (2200 m/s) neutrons.

    Parameters:
    - name (str): Display name.
    - density (float): Density in g/cm^3.
    - molar_mass (float): Molar mass of one formula unit in g/mol.
    - absorption_cross_section (float): Microscopic absorption cross-section in barns.
    - scattering_cross_section (float): Microscopic scattering cross-section in barns.
    """
    name: str
    density: float
    molar_mass: float
    absorption_cross_section: float
    scattering_cross_section: float

    def __post_init__(self):
        if self.density <= 0 or self.molar_mass <= 0:
            raise ValueError(f'{self.name}: density and molar mass must be positive.')
        if self.absorption_cross_section < 0 or self.scattering_cross_section < 0:
            raise ValueError(f'{self.name}: cross-sections cannot be negative.')
        if self.absorption_cross_section + self.scattering_cross_section == 0:
            raise ValueError(f'{self.name}: at least one cross-section must be non-zero.')

    @property
    def number_density(self):
        """Formula units per cm^3."""
        return self.density * AVOGADRO_NUMBER / self.molar_mass

    @property
    def sigma_absorption(self):
        """Macroscopic absorption cross-section in cm^-1."""
        return self.number_density * self.absorption_cross_section * BARN_TO_CM2

    @property
    def sigma_scattering(self):
        """Macroscopic scattering cross-section in cm^-1."""
        return self.number_density * self.scattering_cross_section * BARN_TO_CM2

    @property
    def sigma_total(self):
        """Total macroscopic cross-section in cm^-1."""
        return self.sigma_absorption + self.sigma_scattering

    @property
    def mean_free_path(self):
        """Total mean free path in cm."""
        return 1 / self.sigma_total

    @property
    def absorption_mean_free_path(self):
        """Mean free path for absorption only in cm (infinite if the material does not absorb)."""
        return 1 / self.sigma_absorption if self.sigma_absorption > 0 else float('inf')

    @property
    def scattering_mean_free_path(self):
        """Mean free path for scattering only in cm (infinite if the material does not scatter)."""
        return 1 / self.sigma_scattering if self.sigma_scattering > 0 else float('inf')

    @property
    def absorption_probability(self):
        """Probability that a collision is an absorption."""
        return self.sigma_absorption / self.sigma_total


# The first four materials use the values of the original notebook.
# The others use bound-atom thermal cross-sections (Sears, Neutron News 3 (1992) 26;
# NIST neutron scattering lengths table) summed over the atoms of the formula unit.
# For hydrogenous materials the effective scattering cross-section depends on chemical
# binding and neutron energy, so treat those values as approximate.
MATERIALS = {m.name: m for m in [
    Material('Water', 1.00, 18.0153, 0.6652, 103.0),
    Material('Lead', 11.35, 207.2, 0.158, 11.221),
    Material('Graphite', 1.67, 12.011, 0.0045, 4.74),
    Material('Beryllium', 1.85, 9.012, 0.0076, 7.63),
    Material('Heavy water', 1.107, 20.028, 0.00123, 19.51),
    Material('Polyethylene', 0.94, 14.027, 0.6687, 169.6),
    Material('Boron carbide', 2.52, 55.255, 3068.0, 26.51),
    Material('Cadmium', 8.65, 112.41, 2520.0, 6.5),
    Material('Gadolinium', 7.90, 157.25, 49700.0, 180.0),
    Material('Iron', 7.874, 55.845, 2.56, 11.62),
    Material('Aluminium', 2.70, 26.982, 0.231, 1.503),
    Material('Copper', 8.96, 63.546, 3.78, 8.03),
    Material('Tungsten', 19.25, 183.84, 18.3, 4.6),
    Material('Zirconium', 6.52, 91.224, 0.185, 6.46),
]}
