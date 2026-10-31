import numpy as np
import pytest

from neutron_sim import MATERIALS, Layer, Material, fit_attenuation, run_trials, simulate

WATER = MATERIALS['Water']
LEAD = MATERIALS['Lead']


def within_binomial(count, n, p, n_sigma=4):
    """Check a count against a binomial expectation."""
    return abs(count - n * p) <= n_sigma * np.sqrt(n * p * (1 - p))


def test_water_mean_free_paths_match_notebook():
    assert WATER.mean_free_path == pytest.approx(0.289, abs=1e-3)
    assert WATER.absorption_mean_free_path == pytest.approx(44.972, abs=1e-3)
    assert LEAD.mean_free_path == pytest.approx(2.664, abs=1e-3)


def test_invalid_material_rejected():
    with pytest.raises(ValueError):
        Material('Bad', -1, 10, 1, 1)
    with pytest.raises(ValueError):
        Material('Empty', 1, 10, 0, 0)


def test_every_neutron_has_one_fate():
    result = simulate([Layer(WATER, 5), Layer(LEAD, 5)], 5000, np.random.default_rng(1))
    assert result.reflected + result.transmitted + result.absorbed == 5000
    assert result.absorbed_per_layer.sum() == result.absorbed
    assert len(result.absorption_depths) == result.absorbed
    assert np.all((result.absorption_depths >= 0) & (result.absorption_depths <= 10))


def test_pure_absorber_transmission_is_exponential():
    absorber = Material('Absorber', 1.0, 10.0, 20.0, 0.0)
    n, thickness = 20000, 2.0
    result = simulate([Layer(absorber, thickness)], n, np.random.default_rng(2))
    expected = np.exp(-absorber.sigma_total * thickness)
    assert within_binomial(result.transmitted, n, expected)
    assert result.reflected == 0


def test_woodcock_stack_of_absorbers():
    # Different cross-sections, so the thinner layer uses fictitious collisions
    weak = Material('Weak', 1.0, 10.0, 5.0, 0.0)
    strong = Material('Strong', 1.0, 10.0, 30.0, 0.0)
    n = 20000
    result = simulate([Layer(weak, 2.0), Layer(strong, 0.5)], n, np.random.default_rng(3))
    expected = np.exp(-weak.sigma_total * 2.0 - strong.sigma_total * 0.5)
    assert within_binomial(result.transmitted, n, expected)
    assert result.fictitious_collisions > 0


def test_split_layer_matches_single_layer():
    rng = np.random.default_rng(4)
    split = run_trials([Layer(WATER, 5), Layer(WATER, 5)], 5000, 5, rng)
    single = run_trials([Layer(WATER, 10)], 5000, 5, rng)
    for fate in ('reflected', 'transmitted', 'absorbed'):
        error = np.hypot(split.sem[fate], single.sem[fate])
        assert abs(split.mean[fate] - single.mean[fate]) <= 5 * error + 1e-3


def test_water_10cm_matches_notebook_results():
    # Notebook: about 20% absorbed, 79.7% reflected, 0.3% transmitted
    stats = run_trials([Layer(WATER, 10)], 10000, 4, np.random.default_rng(5))
    assert stats.mean['absorbed'] == pytest.approx(0.200, abs=0.01)
    assert stats.mean['reflected'] == pytest.approx(0.797, abs=0.01)
    assert stats.mean['transmitted'] == pytest.approx(0.003, abs=0.002)


def test_same_seed_gives_same_result():
    layers = [Layer(LEAD, 3)]
    a = simulate(layers, 1000, np.random.default_rng(6))
    b = simulate(layers, 1000, np.random.default_rng(6))
    assert a.fractions == b.fractions


def test_tracked_paths_start_at_origin_and_first_step_is_along_x():
    result = simulate([Layer(LEAD, 3)], 100, np.random.default_rng(7), n_tracked=5)
    assert len(result.paths) == 5
    for path in result.paths:
        assert np.allclose(path[0], 0)
        assert path[1][0] > 0 and np.allclose(path[1][1:], 0)


def test_fit_recovers_attenuation_length():
    thickness = np.linspace(1, 10, 10)
    transmitted = 0.8 * np.exp(-thickness / 3.0)
    fit = fit_attenuation(thickness, transmitted, 0.01 * transmitted)
    assert fit.attenuation_length == pytest.approx(3.0)
    assert fit.reduced_chi_squared == pytest.approx(0, abs=1e-12)


def test_fit_skips_zero_points_and_needs_three():
    thickness = np.array([1.0, 2.0, 3.0, 4.0])
    transmitted = np.array([0.5, 0.2, 0.0, 0.0])
    assert fit_attenuation(thickness, transmitted, 0.01 * np.ones(4)) is None
