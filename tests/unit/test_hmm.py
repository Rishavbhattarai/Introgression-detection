"""Check the HMM algorithms against brute-force enumeration of all hidden paths."""

import itertools
import json
import math

import numpy as np
import pytest

from hmmix.artemis import Calculate_loglikelihood
from hmmix.hmm_functions import (
    Calculate_Posterior_probabillities,
    Emission_probs_poisson,
    GetProbability,
    HMMParam,
    Hybrid_path,
    Make_inhomogeneous_transition_matrix,
    PMAP_path,
    Simulate_from_transition_matrix,
    TrainBaumWelsch,
    Viterbi_path,
    backward,
    forward,
    get_default_HMM_parameters,
    poisson_probability_underflow_safe,
    read_HMM_parameters_from_file,
    write_HMM_to_file,
)


@pytest.fixture(params=[2, 3], ids=['2states', '3states'])
def model(request):
    if request.param == 2:
        params = HMMParam(['Human', 'Archaic'], [0.7, 0.3], [[0.9, 0.1], [0.2, 0.8]], [0.1, 1.5])
    else:
        params = HMMParam(['A', 'B', 'C'], [0.5, 0.3, 0.2],
                          [[0.8, 0.15, 0.05], [0.1, 0.7, 0.2], [0.25, 0.25, 0.5]], [0.05, 0.8, 3.0])
    obs = np.array([0, 2, 1, 0, 4, 3, 0])
    weights = np.array([1.0, 0.5, 1.0, 0.9, 1.0, 0.3, 1.0])
    mutrates = np.array([1.0, 1.2, 0.8, 1.0, 1.1, 1.0, 0.5])
    return params, obs, weights, mutrates


def poisson_pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def brute_force(params, E):
    """Return {path: joint probability} for every hidden path."""
    n, k = E.shape
    joint = {}
    for path in itertools.product(range(k), repeat=n):
        p = params.starting_probabilities[path[0]] * E[0, path[0]]
        for t in range(1, n):
            p *= params.transitions[path[t - 1], path[t]] * E[t, path[t]]
        joint[path] = p
    return joint


def test_poisson_probability_matches_pmf():
    for k in range(0, 30, 3):
        for lam in [1e-3, 0.04, 0.4, 2.5, 17.0]:
            assert poisson_probability_underflow_safe(k, lam) == pytest.approx(poisson_pmf(k, lam), rel=1e-12)


def test_emission_probabilities(model):
    params, obs, weights, mutrates = model
    E = Emission_probs_poisson(params.emissions, obs, weights, mutrates)
    for t, o in enumerate(obs):
        for s, lam in enumerate(params.emissions):
            assert E[t, s] == pytest.approx(poisson_pmf(o, lam * weights[t] * mutrates[t]), rel=1e-12)


def test_likelihood_matches_brute_force(model):
    params, obs, weights, mutrates = model
    E = Emission_probs_poisson(params.emissions, obs, weights, mutrates)
    expected = math.log(sum(brute_force(params, E).values()))
    assert GetProbability(params, weights, obs, mutrates) == pytest.approx(expected, rel=1e-12)


def test_forward_backward_rows(model):
    params, obs, weights, mutrates = model
    E = Emission_probs_poisson(params.emissions, obs, weights, mutrates)
    fwd, scales = forward(E, params.transitions, params.starting_probabilities)
    bwd = backward(E, params.transitions, scales)
    # scaled forward variables are normalised at every step
    np.testing.assert_allclose(fwd.sum(axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(bwd[-1], 1.0)
    np.testing.assert_allclose((fwd * bwd).sum(axis=1), 1.0, rtol=1e-12)


def test_posterior_matches_brute_force(model):
    params, obs, weights, mutrates = model
    E = Emission_probs_poisson(params.emissions, obs, weights, mutrates)
    joint = brute_force(params, E)
    total = sum(joint.values())
    n, k = E.shape
    expected = np.zeros((k, n))
    for path, p in joint.items():
        for t, s in enumerate(path):
            expected[s, t] += p / total
    posterior = Calculate_Posterior_probabillities(E, params)
    np.testing.assert_allclose(posterior, expected, rtol=1e-10)
    np.testing.assert_array_equal(PMAP_path(posterior), expected.argmax(axis=0))


def test_viterbi_matches_brute_force(model):
    params, obs, weights, mutrates = model
    E = Emission_probs_poisson(params.emissions, obs, weights, mutrates)
    joint = brute_force(params, E)
    best = max(joint, key=joint.get)
    np.testing.assert_array_equal(Viterbi_path(E, params), best)
    # and the log likelihood used by artemis agrees with the brute-force joint probability
    assert Calculate_loglikelihood(np.array(best), E, params.starting_probabilities, params.transitions) == pytest.approx(math.log(joint[best]))


def test_hybrid_limits(model):
    params, obs, weights, mutrates = model
    E = Emission_probs_poisson(params.emissions, obs, weights, mutrates)
    posterior = Calculate_Posterior_probabillities(E, params)
    logpost = np.log(posterior.T)
    viterbi = Hybrid_path(E, params.starting_probabilities, params.transitions, logpost, 1.0)
    pmap = Hybrid_path(E, params.starting_probabilities, params.transitions, logpost, 0.0)
    np.testing.assert_array_equal(viterbi, Viterbi_path(E, params))
    np.testing.assert_array_equal(pmap, PMAP_path(posterior))


def test_baum_welch_increases_likelihood(model):
    params, obs, weights, mutrates = model
    ll = GetProbability(params, weights, obs, mutrates)
    for _ in range(10):
        params = TrainBaumWelsch(params, weights, obs, mutrates)
        new_ll = GetProbability(params, weights, obs, mutrates)
        assert new_ll >= ll - 1e-10
        ll = new_ll
        np.testing.assert_allclose(params.transitions.sum(axis=1), 1.0)
        assert params.starting_probabilities.sum() == pytest.approx(1.0)
        assert np.all(params.emissions > 0)


def test_inhomogeneous_matrix_is_posterior_chain(model):
    params, obs, weights, mutrates = model
    E = Emission_probs_poisson(params.emissions, obs, weights, mutrates)
    start, matrix = Make_inhomogeneous_transition_matrix(E, params)
    posterior = Calculate_Posterior_probabillities(E, params)
    np.testing.assert_allclose(start, posterior[:, 0], rtol=1e-10)
    np.testing.assert_allclose(matrix[1:].sum(axis=2), 1.0, rtol=1e-10)
    # propagating the posterior through the chain reproduces the posterior at every step
    marginal = start.copy()
    for t in range(1, len(obs)):
        marginal = marginal @ matrix[t]
        np.testing.assert_allclose(marginal, posterior[:, t], rtol=1e-8)


def test_simulated_paths_have_posterior_marginals():
    params = HMMParam(['Human', 'Archaic'], [0.7, 0.3], [[0.9, 0.1], [0.2, 0.8]], [0.1, 1.5])
    obs = np.array([0, 2, 1, 0, 4, 3, 0])
    ones = np.ones(len(obs))
    E = Emission_probs_poisson(params.emissions, obs, ones, ones)
    start, matrix = Make_inhomogeneous_transition_matrix(E, params)
    posterior = Calculate_Posterior_probabillities(E, params)
    np.random.seed(1)
    paths = np.array([Simulate_from_transition_matrix(start, matrix) for _ in range(4000)])
    np.testing.assert_allclose(paths.mean(axis=0), posterior[1], atol=0.03)


def test_parameter_file_roundtrip(tmp_path):
    params = HMMParam(['Human', 'Archaic'], [0.9, 0.1], [[0.99, 0.01], [0.02, 0.98]], [0.04, 0.4])
    write_HMM_to_file(params, tmp_path / 'p.json')
    loaded = read_HMM_parameters_from_file(tmp_path / 'p.json')
    for key in ['state_names', 'starting_probabilities', 'transitions', 'emissions']:
        np.testing.assert_array_equal(getattr(loaded, key), getattr(params, key))
    assert set(json.loads((tmp_path / 'p.json').read_text())) == {'state_names', 'starting_probabilities', 'transitions', 'emissions'}


def test_default_parameters():
    params = read_HMM_parameters_from_file(None)
    default = get_default_HMM_parameters()
    np.testing.assert_array_equal(params.transitions, default.transitions)
    assert list(params.state_names) == ['Human', 'Archaic']
