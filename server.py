"""
Web server for the neutron shielding simulation.

Serves the HTML/CSS/JS front-end in web/ and a small JSON API that calls the neutron_sim package.
Uses only the Python standard library (plus the simulation's own dependencies).

Run with:  python server.py            (then open http://localhost:8000)
Options:   python server.py --port 8080 --open
"""

import argparse
import json
import math
import webbrowser
from dataclasses import asdict
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from neutron_sim import MATERIALS, Layer, Material, fit_attenuation, run_trials, thickness_sweep

WEB_DIR = Path(__file__).parent / 'web'
MAX_NEUTRONS_PER_RUN = 2_000_000  # neutrons per trial x trials, to keep requests responsive
MAX_TRACKED = 50
MAX_SWEEP_POINTS = 100
FATES = ('reflected', 'transmitted', 'absorbed')


class ApiError(ValueError):
    """An invalid request; reported to the client with status 400."""


def finite_or_none(value):
    """JSON has no inf/nan, so send them as null."""
    value = float(value)
    return value if math.isfinite(value) else None


def parse_material(data):
    try:
        return Material(str(data['name']).strip(), float(data['density']), float(data['molar_mass']),
                        float(data['absorption_cross_section']), float(data['scattering_cross_section']))
    except (KeyError, TypeError) as error:
        raise ApiError(f'Incomplete material: {error}') from error
    except ValueError as error:
        raise ApiError(str(error)) from error


def material_key(material):
    return (material.name, material.density, material.molar_mass,
            material.absorption_cross_section, material.scattering_cross_section)


def parse_int(payload, name, low, high):
    try:
        value = int(payload[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ApiError(f'{name} must be an integer.') from error
    if not low <= value <= high:
        raise ApiError(f'{name} must be between {low} and {high}.')
    return value


def parse_run_settings(payload):
    n_neutrons = parse_int(payload, 'n_neutrons', 1, MAX_NEUTRONS_PER_RUN)
    n_trials = parse_int(payload, 'n_trials', 2, 100)
    if n_neutrons * n_trials > MAX_NEUTRONS_PER_RUN:
        raise ApiError(f'Neutrons per trial x trials must be at most {MAX_NEUTRONS_PER_RUN:,}.')
    seed = parse_int(payload, 'seed', 0, 2**32 - 1)
    return n_neutrons, n_trials, seed


def derived_quantities(material):
    """Macroscopic cross-sections, mean free paths and diffusion length of a material."""
    return {
        'sigma_absorption': material.sigma_absorption,
        'sigma_scattering': material.sigma_scattering,
        'sigma_total': material.sigma_total,
        'absorption_mean_free_path': finite_or_none(material.absorption_mean_free_path),
        'scattering_mean_free_path': finite_or_none(material.scattering_mean_free_path),
        'mean_free_path': material.mean_free_path,
        'absorption_probability': material.absorption_probability,
        'diffusion_length': (1 / math.sqrt(3 * material.sigma_absorption * material.sigma_total)
                             if material.sigma_absorption > 0 else None),
    }


def api_materials(_payload=None):
    """The built-in material library."""
    return {'materials': [asdict(material) for material in MATERIALS.values()]}


def api_derive(payload):
    """Derived quantities for a list of (possibly edited) materials; invalid ones get an error."""
    results = []
    for data in payload.get('materials', []):
        try:
            results.append({'derived': derived_quantities(parse_material(data))})
        except ApiError as error:
            results.append({'error': str(error)})
    return {'results': results}


@lru_cache(maxsize=64)
def cached_stack_run(layer_keys, n_neutrons, n_trials, seed, n_tracked):
    layers = [Layer(Material(*key), thickness) for key, thickness in layer_keys]
    stats = run_trials(layers, n_neutrons, n_trials, np.random.default_rng(seed), n_tracked)
    result = stats.result
    total_neutrons = n_neutrons * n_trials
    layer_ends = np.cumsum([thickness for _, thickness in layer_keys])
    counts, edges = np.histogram(result.absorption_depths, bins=60, range=(0, layer_ends[-1]))
    return {
        'mean': {fate: float(stats.mean[fate]) for fate in FATES},
        'sem': {fate: float(stats.sem[fate]) for fate in FATES},
        'n_neutrons': n_neutrons,
        'n_trials': n_trials,
        'total_neutrons': total_neutrons,
        'collisions': int(result.collisions),
        'fictitious_collisions': int(result.fictitious_collisions),
        'layer_ends': layer_ends.tolist(),
        'absorbed_per_layer': (result.absorbed_per_layer / total_neutrons).tolist(),
        'absorption_histogram': {'edges': edges.tolist(), 'counts': counts.tolist()},
        'paths': [np.round(path, 4).tolist() for path in result.paths],
    }


def api_simulate(payload):
    """Simulate a layer stack. Payload: layers [{material, thickness}], run settings, n_tracked."""
    layers = payload.get('layers') or []
    if not layers:
        raise ApiError('Add at least one layer.')
    layer_keys = []
    for layer in layers:
        material = parse_material(layer.get('material') or {})
        try:
            thickness = float(layer['thickness'])
        except (KeyError, TypeError, ValueError) as error:
            raise ApiError('Every layer needs a thickness.') from error
        if not 0 < thickness < 1e4:
            raise ApiError('Layer thicknesses must be positive (in cm).')
        layer_keys.append((material_key(material), thickness))
    n_tracked = parse_int(payload, 'n_tracked', 0, MAX_TRACKED)
    return cached_stack_run(tuple(layer_keys), *parse_run_settings(payload), n_tracked)


@lru_cache(maxsize=256)
def cached_sweep(key, thicknesses, n_neutrons, n_trials, seed):
    sweep = thickness_sweep(Material(*key), thicknesses, n_neutrons, n_trials, np.random.default_rng(seed))
    return {name: values.tolist() for name, values in sweep.items()}


def api_sweep(payload):
    """Thickness sweep for one material (the front-end calls this once per material, to show progress)."""
    material = parse_material(payload.get('material') or {})
    try:
        thickness_min = float(payload['thickness_min'])
        thickness_max = float(payload['thickness_max'])
    except (KeyError, TypeError, ValueError) as error:
        raise ApiError('Thickness range must be numbers.') from error
    if not 0 < thickness_min < thickness_max < 1e4:
        raise ApiError('Thickness range must satisfy 0 < minimum < maximum.')
    n_points = parse_int(payload, 'n_points', 3, MAX_SWEEP_POINTS)
    thicknesses = tuple(np.linspace(thickness_min, thickness_max, n_points).tolist())
    return {'material': material.name,
            **cached_sweep(material_key(material), thicknesses, *parse_run_settings(payload))}


def api_fit(payload):
    """Fit ln(T) against thickness over [fit_min, fit_max]."""
    try:
        thickness = np.asarray(payload['thickness'], dtype=float)
        transmitted = np.asarray(payload['transmitted'], dtype=float)
        sem = np.asarray(payload['transmitted_sem'], dtype=float)
        fit_min = float(payload.get('fit_min', -np.inf))
        fit_max = float(payload.get('fit_max', np.inf))
    except (KeyError, TypeError, ValueError) as error:
        raise ApiError('Fit needs thickness, transmitted and transmitted_sem arrays.') from error
    if not thickness.shape == transmitted.shape == sem.shape:
        raise ApiError('Fit arrays must have the same length.')

    in_range = (thickness >= fit_min - 1e-9) & (thickness <= fit_max + 1e-9)
    fit = fit_attenuation(thickness[in_range], transmitted[in_range], sem[in_range])
    if fit is None:
        return {'fit': None}
    return {'fit': {
        'slope': fit.slope,
        'intercept': fit.intercept,
        'attenuation_length': fit.attenuation_length,
        'attenuation_length_error': fit.attenuation_length_error,
        'reduced_chi_squared': finite_or_none(fit.reduced_chi_squared),
        'n_points': int(np.count_nonzero(fit.used)),
    }}


ROUTES = {
    ('GET', '/api/materials'): api_materials,
    ('POST', '/api/derive'): api_derive,
    ('POST', '/api/simulate'): api_simulate,
    ('POST', '/api/sweep'): api_sweep,
    ('POST', '/api/fit'): api_fit,
}


def plotly_bundle():
    """Path of the plotly.js bundle shipped with the plotly Python package, if installed."""
    try:
        import plotly
    except ImportError:
        return None
    path = Path(plotly.__file__).parent / 'package_data' / 'plotly.min.js'
    return path if path.exists() else None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def send_json(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_api(self, method):
        route = ROUTES.get((method, self.path.split('?')[0]))
        if route is None:
            self.send_json(404, {'error': 'Unknown API endpoint.'})
            return
        try:
            payload = {}
            if method == 'POST':
                length = int(self.headers.get('Content-Length', 0))
                payload = json.loads(self.rfile.read(length) or b'{}')
                if not isinstance(payload, dict):
                    raise ApiError('Request body must be a JSON object.')
            self.send_json(200, route(payload))
        except json.JSONDecodeError:
            self.send_json(400, {'error': 'Request body is not valid JSON.'})
        except ApiError as error:
            self.send_json(400, {'error': str(error)})
        except Exception as error:  # Report unexpected failures instead of dropping the connection
            self.log_error('Error handling %s: %r', self.path, error)
            self.send_json(500, {'error': f'Internal error: {error}'})

    def do_GET(self):
        if self.path.startswith('/api/'):
            self.handle_api('GET')
        elif self.path == '/vendor/plotly.min.js' and plotly_bundle():
            body = plotly_bundle().read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/javascript')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'max-age=86400')
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def do_POST(self):
        self.handle_api('POST')

    def log_message(self, format, *args):
        if self.path.startswith('/api/'):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description='Neutron shielding Monte Carlo web interface.')
    parser.add_argument('--host', default='127.0.0.1', help='Address to listen on (default 127.0.0.1).')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on (default 8000).')
    parser.add_argument('--open', action='store_true', help='Open the page in a web browser.')
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f'http://{"localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host}:{args.port}'
    print(f'Neutron shielding Monte Carlo running at {url}  (Ctrl+C to stop)')
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
