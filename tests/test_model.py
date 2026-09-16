import numpy as np
import pytest

from cartpole import (CartPoleParams, controllability_rank, linearize, lqr_gain,
                      nonlinear_dynamics, pole_placement_gain, simulate)


def numerical_jacobian(f, x0, eps=1e-6):
    n = len(x0)
    J = np.zeros((len(f(x0)), n))
    for j in range(n):
        d = np.zeros(n)
        d[j] = eps
        J[:, j] = (f(x0 + d) - f(x0 - d)) / (2 * eps)
    return J


def test_linearization_matches_nonlinear_model():
    p = CartPoleParams()
    A, B = linearize(p)
    x_eq = np.zeros(4)
    A_num = numerical_jacobian(lambda s: nonlinear_dynamics(s, 0.0, p), x_eq)
    B_num = (nonlinear_dynamics(x_eq, 1e-6, p) - nonlinear_dynamics(x_eq, -1e-6, p)) / 2e-6
    assert np.allclose(A, A_num, atol=1e-6)
    assert np.allclose(B[:, 0], B_num, atol=1e-6)


def test_upright_is_open_loop_unstable_but_controllable():
    A, B = linearize(CartPoleParams())
    assert np.max(np.real(np.linalg.eigvals(A))) > 0
    assert controllability_rank(A, B) == 4


@pytest.mark.parametrize("gain_fn", [pole_placement_gain, lqr_gain])
def test_closed_loop_is_stable(gain_fn):
    A, B = linearize(CartPoleParams())
    K = gain_fn(A, B)
    assert np.max(np.real(np.linalg.eigvals(A - B @ K))) < 0


@pytest.mark.parametrize("gain_fn", [pole_placement_gain, lqr_gain])
def test_controller_recovers_from_small_tilt(gain_fn):
    p = CartPoleParams()
    A, B = linearize(p)
    r = simulate(gain_fn(A, B), p, np.array([0, 0, 0.15, 0]), T=8.0)
    assert not r.fell
    assert abs(r.X[-1, 2]) < 1e-3 and abs(r.X[-1, 0]) < 1e-2


def test_open_loop_falls():
    p = CartPoleParams()
    r = simulate(None, p, np.array([0, 0, 0.15, 0]), T=5.0)
    assert r.fell
