"""Pole placement vs LQR on the nonlinear cart-pole.

Two experiments, both run on the *nonlinear* plant with gains designed on the
linear model:

1. Standard scenario  -> docs/comparison.png
   start tilted 0.15 rad, 20 N kick for 0.2 s at t = 3 s, cart set-point
   step to +1 m at t = 6 s.  Prints settling times, peak angle and control effort.

2. Recovery envelope -> docs/recovery_envelope.png
   for several actuator limits u_max, the largest initial tilt each
   controller can still recover from.

Usage:
    python compare_controllers.py            # both experiments, write figures
    python compare_controllers.py --no-show  # same, but do not open windows
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from cartpole import (LQR_PRESETS, CartPoleParams, linearize, lqr_gain,
                      pole_placement_gain, settling_time, simulate)

DOCS = Path(__file__).resolve().parent / "docs"

KICK_T, KICK_DUR, KICK_N = 3.0, 0.2, 20.0
STEP_T, STEP_X = 6.0, 1.0
T_END = 14.0


def kick(t: float) -> float:
    return KICK_N if KICK_T <= t < KICK_T + KICK_DUR else 0.0


def x_ref(t: float) -> float:
    return STEP_X if t >= STEP_T else 0.0


def controllers(p: CartPoleParams) -> dict[str, np.ndarray]:
    A, B = linearize(p)
    return {
        "Pole placement": pole_placement_gain(A, B),
        "LQR gentle (R=1)": lqr_gain(A, B, *LQR_PRESETS["gentle"]),
        "LQR aggressive (R=0.1)": lqr_gain(A, B, *LQR_PRESETS["aggressive"]),
    }


# --------------------------------------------------------------------------
def standard_scenario(p: CartPoleParams, show: bool) -> list[dict]:
    """Plot the combined scenario; compute metrics on isolated kick / step runs."""
    gains = controllers(p)
    x0 = np.array([0.0, 0.0, 0.15, 0.0])
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    rows = []
    for name, K in gains.items():
        r = simulate(K, p, x0, T=T_END, x_ref_fn=x_ref, disturbance_fn=kick)
        axes[0].plot(r.t, np.degrees(r.X[:, 2]), label=name)
        axes[1].plot(r.t, r.X[:, 0], label=name)
        axes[2].plot(r.t, r.u, label=name)

        # metrics on isolated experiments so the two events do not interfere
        rk = simulate(K, p, np.zeros(4), T=KICK_T + 8.0, disturbance_fn=kick)
        rs = simulate(K, p, np.zeros(4), T=8.0, x_ref_fn=lambda t: STEP_X)
        rows.append({
            "controller": name,
            "K": np.round(K.ravel(), 2),
            "ts_phi_kick": settling_time(rk.t, rk.X[:, 2], KICK_T + KICK_DUR, np.radians(1.0)),
            "peak_phi_kick": float(np.degrees(np.max(np.abs(rk.X[:, 2])))),
            "ts_x_step": settling_time(rs.t, rs.X[:, 0] - STEP_X, 0.0, 0.02),
            "peak_phi_step": float(np.degrees(np.max(np.abs(rs.X[:, 2])))),
            "peak_u": float(np.max(np.abs(r.u))),
            "effort": float(np.trapezoid(r.u ** 2, r.t)),
        })

    axes[0].set_ylabel("pole angle [deg]")
    axes[1].set_ylabel("cart position [m]")
    axes[2].set_ylabel("control force u [N]")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.axvspan(KICK_T, KICK_T + KICK_DUR, color="orange", alpha=0.25)
        ax.axvline(STEP_T, color="gray", ls="--", lw=1)
    axes[1].plot([0, STEP_T, STEP_T, T_END], [0, 0, STEP_X, STEP_X], "k:", lw=1, label="set-point")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper left")
    fig.suptitle("Nonlinear cart-pole: 0.15 rad initial tilt, 20 N kick at 3 s, +1 m step at 6 s")
    fig.tight_layout()
    fig.savefig(DOCS / "comparison.png", dpi=130)
    if show:
        plt.show()
    plt.close(fig)
    return rows


# --------------------------------------------------------------------------
def max_recoverable_tilt(K: np.ndarray, p: CartPoleParams, u_max: float) -> float:
    """Largest initial tilt [rad] from which the pole returns upright (bisection)."""
    def ok(phi0: float) -> bool:
        r = simulate(K, p, np.array([0.0, 0.0, phi0, 0.0]), T=6.0, u_max=u_max)
        return (not r.fell) and abs(r.X[-1, 2]) < 0.02 and abs(r.X[-1, 0]) < 5.0

    lo, hi = 0.0, np.pi / 2 - 0.01
    if not ok(0.02):
        return 0.0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def recovery_envelope(p: CartPoleParams, show: bool) -> dict[str, list[float]]:
    gains = controllers(p)
    u_limits = [5, 10, 20, 40, 80, 160]
    result = {}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, K in gains.items():
        tilts = [np.degrees(max_recoverable_tilt(K, p, um)) for um in u_limits]
        result[name] = tilts
        ax.plot(u_limits, tilts, marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xticks(u_limits, [str(u) for u in u_limits])
    ax.set_xlabel("actuator limit u_max [N]")
    ax.set_ylabel("max recoverable initial tilt [deg]")
    ax.set_title("Recovery envelope on the nonlinear plant")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "recovery_envelope.png", dpi=130)
    if show:
        plt.show()
    plt.close(fig)
    return {"u_max": u_limits, **result}


# --------------------------------------------------------------------------
def fmt(v):
    return "never" if v is None else f"{v:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-show", action="store_true", help="only save figures to docs/")
    args = ap.parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    DOCS.mkdir(exist_ok=True)
    p = CartPoleParams()

    rows = standard_scenario(p, show=not args.no_show)
    print("| controller | K | kick: peak angle [deg] | kick: settle <1 deg [s] | step: settle x <2 cm [s] | step: peak angle [deg] | peak u [N] | effort int(u^2) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['controller']} | {r['K']} | {r['peak_phi_kick']:.1f} | {fmt(r['ts_phi_kick'])} | "
              f"{fmt(r['ts_x_step'])} | {r['peak_phi_step']:.1f} | {r['peak_u']:.1f} | {r['effort']:.0f} |")

    env = recovery_envelope(p, show=not args.no_show)
    print()
    print("| u_max [N] | " + " | ".join(str(u) for u in env["u_max"]) + " |")
    print("|---|" + "---|" * len(env["u_max"]))
    for name in controllers(p):
        print(f"| {name} | " + " | ".join(f"{v:.1f}" for v in env[name]) + " |")


if __name__ == "__main__":
    main()
