import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import server

WATER = {'name': 'Water', 'density': 1.0, 'molar_mass': 18.0153,
         'absorption_cross_section': 0.6652, 'scattering_cross_section': 103.0}
LEAD = {'name': 'Lead', 'density': 11.35, 'molar_mass': 207.2,
        'absorption_cross_section': 0.158, 'scattering_cross_section': 11.221}
SETTINGS = {'n_neutrons': 2000, 'n_trials': 3, 'seed': 1}


def test_materials_lists_library():
    names = [m['name'] for m in server.api_materials()['materials']]
    assert names[:4] == ['Water', 'Lead', 'Graphite', 'Beryllium']


def test_derive_reports_invalid_materials_individually():
    results = server.api_derive({'materials': [WATER, {**WATER, 'density': -1}]})['results']
    assert results[0]['derived']['mean_free_path'] == pytest.approx(0.289, abs=1e-3)
    assert 'error' in results[1]


def test_derive_sends_infinite_mean_free_path_as_null():
    non_absorber = {**WATER, 'absorption_cross_section': 0}
    derived = server.api_derive({'materials': [non_absorber]})['results'][0]['derived']
    assert derived['absorption_mean_free_path'] is None
    assert derived['diffusion_length'] is None


def test_simulate_returns_consistent_result():
    result = server.api_simulate({'layers': [{'material': LEAD, 'thickness': 5},
                                             {'material': WATER, 'thickness': 5}],
                                  'n_tracked': 3, **SETTINGS})
    assert sum(result['mean'].values()) == pytest.approx(1)
    assert result['layer_ends'] == [5, 10]
    assert len(result['paths']) == 3
    assert sum(result['absorption_histogram']['counts']) == pytest.approx(
        result['mean']['absorbed'] * result['total_neutrons'])
    json.dumps(result)  # must be JSON serialisable


@pytest.mark.parametrize('payload, message', [
    ({'layers': [], 'n_tracked': 0, **SETTINGS}, 'at least one layer'),
    ({'layers': [{'material': WATER, 'thickness': -1}], 'n_tracked': 0, **SETTINGS}, 'positive'),
    ({'layers': [{'material': WATER, 'thickness': 1}], 'n_tracked': 0, **SETTINGS, 'n_trials': 1}, 'n_trials'),
    ({'layers': [{'material': WATER, 'thickness': 1}], 'n_tracked': 0, **SETTINGS,
      'n_neutrons': 1_000_000, 'n_trials': 10}, 'at most'),
])
def test_simulate_rejects_bad_requests(payload, message):
    with pytest.raises(server.ApiError, match=message):
        server.api_simulate(payload)


def test_sweep_and_fit():
    sweep = server.api_sweep({'material': LEAD, 'thickness_min': 1, 'thickness_max': 10, 'n_points': 5, **SETTINGS})
    assert sweep['thickness'] == pytest.approx([1, 3.25, 5.5, 7.75, 10])
    fit = server.api_fit({**sweep, 'fit_min': 3, 'fit_max': 10})['fit']
    assert fit['n_points'] == 4
    assert 5 < fit['attenuation_length'] < 15


@pytest.fixture
def running_server():
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{httpd.server_address[1]}'
    httpd.shutdown()


def post(url, payload):
    request = Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(request) as response:
        return json.load(response)


def test_http_round_trip(running_server):
    with urlopen(running_server + '/') as response:
        assert b'Neutron Shielding Monte Carlo' in response.read()
    with urlopen(running_server + '/api/materials') as response:
        assert json.load(response)['materials']
    result = post(running_server + '/api/simulate',
                  {'layers': [{'material': WATER, 'thickness': 2}], 'n_tracked': 0, **SETTINGS})
    assert sum(result['mean'].values()) == pytest.approx(1)


def test_http_errors_are_json(running_server):
    with pytest.raises(HTTPError) as error:
        post(running_server + '/api/simulate', {'layers': []})
    assert error.value.code == 400
    assert 'layer' in json.load(error.value)['error']
