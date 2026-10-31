# Neutron Shielding Monte Carlo

Monte Carlo simulation of thermal neutrons passing through shielding layers, with an
interactive web interface for changing material properties, layer stacks and simulation settings.

A beam of neutrons enters a stack of slabs at normal incidence. Each neutron is followed through
exponentially distributed flights, isotropic scattering and absorption until it is **reflected**,
**transmitted** or **absorbed**. Multi-layer shields use Woodcock (delta) tracking.

The original project report is kept in `Neutron_shielding_Monte_Carlo.ipynb`.

## Quick start

```bash
pip install -r requirements.txt
python server.py --open
```

This opens http://localhost:8000 in your browser (use `--port` to pick another port).
The front-end is plain HTML, CSS and JavaScript in `web/`. `server.py` serves it and runs the
simulations through a small JSON API, using only the Python standard library. Plotly.js is served
from the installed `plotly` package, so the page also works offline.

## The interface

| Tab | What it does |
| --- | --- |
| **Materials** | Edit density, molar mass and absorption/scattering cross-sections of any material, or add your own. Shows macroscopic cross-sections, mean free paths, absorption probability per collision and diffusion length. Edits are saved in the browser. |
| **Shield simulation** | Build a stack of layers (any materials, any thicknesses). Shows reflected/transmitted/absorbed fractions with errors, absorption in each layer, a histogram of absorption depth and 3D neutron trajectories. |
| **Thickness sweep** | Simulates single-material slabs over a range of thicknesses, plots all three fractions and fits ln(T) against thickness to find the attenuation length. The fit range is adjustable, and the results can be downloaded as CSV. |
| **Model** | Describes the physics model and its limitations. |

The sidebar sets the number of neutrons per trial, the number of trials used for error estimates, and the random seed.

### API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/materials` | Built-in material library |
| `POST /api/derive` | Derived quantities for a list of materials |
| `POST /api/simulate` | Simulate a layer stack |
| `POST /api/sweep` | Thickness sweep for one material |
| `POST /api/fit` | Attenuation-length fit over a thickness range |

## Using the package from Python

```python
import numpy as np
from neutron_sim import MATERIALS, Layer, Material, run_trials, thickness_sweep, fit_attenuation

rng = np.random.default_rng(0)

# Lead followed by water, 10 cm each
stats = run_trials([Layer(MATERIALS['Lead'], 10), Layer(MATERIALS['Water'], 10)],
                   n_neutrons=10_000, n_trials=10, rng=rng)
print(stats.mean, stats.sem)

# A custom material: density (g/cm^3), molar mass (g/mol), sigma_a and sigma_s (barns)
borated = Material('My shield', 1.2, 30.0, 50.0, 40.0)

# Attenuation length of water from a thickness sweep
sweep = thickness_sweep(MATERIALS['Water'], np.linspace(1, 20, 20), 10_000, 10, rng)
fit = fit_attenuation(sweep['thickness'], sweep['transmitted'], sweep['transmitted_sem'])
print(fit.attenuation_length, fit.attenuation_length_error)
```

## Project layout

```
neutron_sim/
  materials.py   Material class and the material library
  transport.py   Vectorised Monte Carlo transport (Woodcock tracking)
  analysis.py    Repeated trials, thickness sweeps, attenuation fits
server.py        Web server and JSON API
web/             HTML/CSS/JS front-end
tests/           pytest tests, including analytic checks and API tests
```

Run the tests with `pytest`.

## Materials

Water, lead, graphite and beryllium use the values of the original notebook. The other materials
(heavy water, polyethylene, boron carbide, cadmium, gadolinium, iron, aluminium, copper,
tungsten, zirconium) use bound-atom thermal cross-sections from the NIST neutron scattering
lengths table (Sears, *Neutron News* 3, 1992), summed over the atoms of each formula unit.
Values for hydrogenous materials depend on chemical binding, so treat them as approximate.

## Model assumptions

- One-speed (thermal) neutrons: there is no energy loss and no energy dependence of cross-sections.
- Isotropic elastic scattering in the laboratory frame.
- Slabs are infinite in y and z.

## Changes from the notebook

- All neutrons are simulated in vectorised batches, so large runs are much faster.
- Any number of layers is supported, with correct Woodcock tracking (a new real/fictitious test at
  every collision, direction kept on fictitious collisions, region-dependent absorption).
- Errors are standard errors of the mean (`ddof=1`, divided by √trials), and the fit covariance
  uses the stated errors (`cov='unscaled'`), so the reduced χ² is meaningful.
- The random-number generator demonstrations from the notebook are not included.
