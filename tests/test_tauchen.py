import numpy as np
import pytest

from ar1disc import DiscreteMarkovChain, tauchen


# ---- 遷移行列としての基本性質 ----------------------------------------

def test_rows_sum_to_one():
    mc = tauchen(n=11, rho=0.9, sigma=0.1)
    assert np.allclose(mc.P.sum(axis=1), 1.0)


def test_probabilities_nonnegative():
    mc = tauchen(n=11, rho=0.9, sigma=0.1)
    assert (mc.P >= 0).all()


def test_grid_centered_at_long_run_mean():
    rho, mu = 0.8, 0.5
    mc = tauchen(n=9, rho=rho, sigma=0.2, mu=mu)
    assert np.isclose(mc.grid.mean(), mu / (1 - rho))


# ---- QuantEcon との突き合わせ -------------------------------------

@pytest.mark.parametrize("n, rho, sigma, mu, n_std", [
    (5, 0.5, 1.0, 0.0, 3),
    (21, 0.945, 0.025, 0.0, 3),       # Arellano (2008) に近い値
    (15, 0.9, 0.1, 0.3, 4),
    (7, -0.4, 0.5, 1.0, 2),
])
def test_matches_quantecon(n, rho, sigma, mu, n_std):
    qe = pytest.importorskip("quantecon")
    ours = tauchen(n, rho, sigma, mu=mu, n_std=n_std)
    theirs = qe.markov.tauchen(n, rho, sigma, mu=mu, n_std=n_std)
    assert np.allclose(ours.grid, theirs.state_values)
    assert np.allclose(ours.P, theirs.P)


# ---- 近似の精度 ----------------------------------------------------

def test_moments_close_when_rho_moderate():
    """rho が中程度なら、格子を細かくすると分散と自己相関が理論値に近づく。"""
    rho, sigma = 0.5, 1.0
    mc = tauchen(n=101, rho=rho, sigma=sigma)
    pi = mc.stationary()
    m = pi @ mc.grid
    var = pi @ (mc.grid - m) ** 2
    cov1 = pi @ ((mc.grid - m) * (mc.P @ (mc.grid - m)))
    assert np.isclose(var, sigma**2 / (1 - rho**2), rtol=0.02)
    assert np.isclose(cov1 / var, rho, atol=0.01)


# ---- DiscreteMarkovChain のメソッド -----------------------------------

def test_stationary_is_fixed_point():
    mc = tauchen(n=11, rho=0.9, sigma=0.1)
    pi = mc.stationary()
    assert np.isclose(pi.sum(), 1.0)
    assert np.allclose(pi @ mc.P, pi)


def test_simulate_reproducible_with_seed():
    mc = tauchen(n=7, rho=0.8, sigma=0.2)
    a = mc.simulate(500, seed=1)
    b = mc.simulate(500, seed=1)
    assert a.shape == (500,)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) <= set(mc.grid)


def test_simulate_init_state():
    mc = tauchen(n=7, rho=0.8, sigma=0.2)
    assert mc.simulate(10, init=0, seed=0)[0] == mc.grid[0]


# ---- 入力チェック --------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(n=1, rho=0.5, sigma=1.0),
    dict(n=5, rho=1.0, sigma=1.0),
    dict(n=5, rho=0.5, sigma=0.0),
])
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        tauchen(**kwargs)


def test_chain_rejects_bad_matrix():
    with pytest.raises(ValueError):
        DiscreteMarkovChain(grid=[0.0, 1.0], P=[[0.5, 0.4], [0.5, 0.5]])
