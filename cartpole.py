"""Cart-pole (inverted pendulum on a cart): model, linearisation, controllers, simulation.

State vector: X = [x, x_dot, phi, phi_dot]
    x    cart position [m]
    phi  pole angle measured from the upright position [rad]
Input: u = horizontal force on the cart [N]

Conventions follow the classic CTMS inverted-pendulum derivation, so the
linear model here is identical to the one used in the interactive demo:

    (M + m) x'' + b x' - m L phi'' cos(phi) + m L phi'^2 sin(phi) = u
    (I + m L^2) phi'' - m g L sin(phi) = m L x'' cos(phi)

with I = m L^2 (pole treated as a point mass at distance L from the pivot).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.signal import place_poles


@dataclass
class CartPoleParams:
    M: float = 1.0   # cart mass [kg]
    m: float = 0.2   # pole mass [kg]
    L: float = 0.5   # pivot -> pole centre of mass [m]
    g: float = 9.8   # gravity [m/s^2]
    b: float = 0.1   # cart viscous friction [N*s/m]

    @property
    def I(self) -> float:
        """Pole moment of inertia about the pivot (point mass)."""
        return self.m * self.L ** 2


# --------------------------------------------------------------------------
# Linear model
# --------------------------------------------------------------------------
def linearize(p: CartPoleParams) -> tuple[np.ndarray, np.ndarray]:
    """Return (A, B) of the model linearised about the upright equilibrium."""
    M, m, L, g, b, I = p.M, p.m, p.L, p.g, p.b, p.I
    D = I * (M + m) + M * m * L ** 2
    A = np.array([
        [0, 1, 0, 0],
        [0, -(I + m * L ** 2) * b / D, (m ** 2 * g * L ** 2) / D, 0],
        [0, 0, 0, 1],
        [0, -(m * L * b) / D, m * g * L * (M + m) / D, 0],
    ])
    B = np.array([[0.0], [(I + m * L ** 2) / D], [0.0], [m * L / D]])
    return A, B


def controllability_rank(A: np.ndarray, B: np.ndarray) -> int:
    n = A.shape[0]
    blocks, AB = [B], B
    for _ in range(1, n):
        AB = A @ AB
        blocks.append(AB)
    return int(np.linalg.matrix_rank(np.hstack(blocks)))


# --------------------------------------------------------------------------
# Controllers: both return a 1x4 gain K for u = -K (X - X_ref)
# --------------------------------------------------------------------------
DEFAULT_POLES = (-3.0, -3.1, -3.2, -3.3)

# LQR weightings: "gentle" spends little control effort (slow cart return),
# "aggressive" weights the angle heavily and control cheaply, which lands
# close to the pole-placement design above.
LQR_PRESETS = {
    "gentle": (np.diag([10.0, 1.0, 10.0, 1.0]), np.array([[1.0]])),
    "aggressive": (np.diag([10.0, 1.0, 100.0, 1.0]), np.array([[0.1]])),
}
DEFAULT_Q, DEFAULT_R = LQR_PRESETS["gentle"]


def pole_placement_gain(A, B, poles=DEFAULT_POLES) -> np.ndarray:
    return place_poles(A, B, list(poles)).gain_matrix


def lqr_gain(A, B, Q=DEFAULT_Q, R=DEFAULT_R) -> np.ndarray:
    """Infinite-horizon continuous LQR: minimise the integral of (X'QX + u'Ru)."""
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


# --------------------------------------------------------------------------
# Nonlinear plant + integrator
# --------------------------------------------------------------------------
def nonlinear_dynamics(state: np.ndarray, u: float, p: CartPoleParams) -> np.ndarray:
    """X_dot for the full nonlinear cart-pole."""
    _, xd, phi, phid = state
    s, c = np.sin(phi), np.cos(phi)
    M, m, L, g, b, I = p.M, p.m, p.L, p.g, p.b, p.I
    mass = np.array([[M + m, -m * L * c],
                     [-m * L * c, I + m * L ** 2]])
    rhs = np.array([u - b * xd - m * L * phid ** 2 * s,
                    m * g * L * s])
    xdd, phidd = np.linalg.solve(mass, rhs)
    return np.array([xd, xdd, phid, phidd])


def linear_dynamics(state: np.ndarray, u: float, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ state + B[:, 0] * u


def rk4_step(f: Callable[[np.ndarray, float], np.ndarray], state: np.ndarray, u: float, dt: float) -> np.ndarray:
    k1 = f(state, u)
    k2 = f(state + 0.5 * dt * k1, u)
    k3 = f(state + 0.5 * dt * k2, u)
    k4 = f(state + dt * k3, u)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


# --------------------------------------------------------------------------
# Batch simulation
# --------------------------------------------------------------------------
@dataclass
class SimResult:
    t: np.ndarray
    X: np.ndarray        # (N, 4)
    u: np.ndarray        # (N,) control force (after saturation)
    fell: bool           # |phi| exceeded pi/2 at some point


def simulate(K: np.ndarray | None,
             p: CartPoleParams,
             x0: np.ndarray,
             T: float = 10.0,
             dt: float = 0.01,
             x_ref_fn: Callable[[float], float] | None = None,
             disturbance_fn: Callable[[float], float] | None = None,
             u_max: float | None = None,
             nonlinear: bool = True) -> SimResult:
    """Simulate the closed-loop cart-pole.

    K              1x4 state-feedback gain, or None for open loop.
    x_ref_fn(t)    cart position set-point (default 0).
    disturbance_fn(t)  external force on the cart [N] (default 0).
    u_max          actuator saturation |u| <= u_max (None = unlimited).
    """
    A, B = linearize(p)
    if nonlinear:
        f = lambda s, u: nonlinear_dynamics(s, u, p)          # noqa: E731
    else:
        f = lambda s, u: linear_dynamics(s, u, A, B)          # noqa: E731
    n = int(round(T / dt)) + 1
    t = np.arange(n) * dt
    X = np.zeros((n, 4))
    U = np.zeros(n)
    X[0] = x0
    fell = False
    for i in range(n - 1):
        ref = np.array([x_ref_fn(t[i]) if x_ref_fn else 0.0, 0.0, 0.0, 0.0])
        u = 0.0 if K is None else -float((K @ (X[i] - ref)).item())
        if u_max is not None:
            u = float(np.clip(u, -u_max, u_max))
        U[i] = u
        d = disturbance_fn(t[i]) if disturbance_fn else 0.0
        X[i + 1] = rk4_step(f, X[i], u + d, dt)
        if abs(X[i + 1, 2]) > np.pi / 2:
            fell = True
            X[i + 1:] = X[i + 1]
            break
    U[-1] = U[-2]
    return SimResult(t, X, U, fell)


def settling_time(t: np.ndarray, y: np.ndarray, t_from: float, tol: float) -> float | None:
    """Time after t_from at which |y| stays below tol for good (None if never)."""
    idx = np.where(t >= t_from)[0]
    outside = idx[np.abs(y[idx]) > tol]
    if len(outside) == 0:
        return 0.0
    last = outside[-1]
    if last == idx[-1]:
        return None
    return float(t[last + 1] - t_from)
