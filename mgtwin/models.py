"""
RMS-timescale models.

Only the minimum needed to exercise the identification and gate machinery: a
single grid-forming converter supplying an islanded load, which is the standard
load-rejection / load-step experiment.

The library form is what the identifier fits. `ramp_limit_pu_s` is present so
that a *truth* trajectory can be generated with a structure the library form
does not contain -- a real BESS ramp limit. That is how model-form mismatch gets
injected on purpose, rather than being discovered in commissioning.

Convention (declared, checked, never inferred)
---------------------------------------------
- omega in pu of nominal frequency; 1.0 = nominal.
- Pm is the *measured* converter output in pu on converter rating, GENERATOR
  positive: a positive Pm means injection.
- rp_pu is the P-f droop: steady state omega = 1 - rp_pu * (Pm - p_ref).
  So rp_pu = 0.05 means 5% droop, i.e. full output swings frequency by 5%.
"""

from __future__ import annotations

import numpy as np


def _rhs(
    state: np.ndarray,
    p_e: float,
    rp_pu: float,
    tp_sec: float,
    h_sec: float,
    p_ref: float,
    ramp_limit_pu_s: float | None,
) -> np.ndarray:
    """
    d/dt [omega, Pm].

    2H * d(omega)/dt = (p_ref - Pm) - (omega - 1) / rp_pu
    Tp * d(Pm)/dt    = p_e - Pm            [optionally rate-limited]

    The omega equation has the droop term as a restoring feedback, so the
    steady state is omega = 1 + rp_pu * (p_ref - Pm), i.e. output above
    reference pulls frequency down. Sign errors here invert the whole response,
    which is why the convention is written out above rather than assumed.
    """
    omega, p_m = state

    d_omega = ((p_ref - p_m) - (omega - 1.0) / rp_pu) / (2.0 * h_sec)
    d_pm = (p_e - p_m) / tp_sec

    if ramp_limit_pu_s is not None:
        d_pm = float(np.clip(d_pm, -ramp_limit_pu_s, ramp_limit_pu_s))

    return np.array([d_omega, d_pm])


def simulate_gfm(
    t: np.ndarray,
    p_load: np.ndarray,
    rp_pu: float,
    tp_sec: float,
    h_sec: float,
    p_ref: float = 0.0,
    ramp_limit_pu_s: float | None = None,
    omega0: float = 1.0,
) -> np.ndarray:
    """
    Fixed-step RK4 over the supplied time vector.

    Fixed step, not adaptive, on purpose: adaptive stepping hides marginal
    stability behind a shrinking dt, so a model that is numerically fragile
    looks converged. If this integrator goes unstable the model deserves to
    look unstable.

    Returns array of shape (len(t), 2) with columns [omega_pu, p_m_pu].
    """
    t = np.asarray(t, dtype=float)
    p_load = np.asarray(p_load, dtype=float)
    if t.shape != p_load.shape:
        raise ValueError("t and p_load must have the same shape")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")

    out = np.empty((t.size, 2), dtype=float)
    state = np.array([omega0, p_load[0]])
    out[0] = state

    for k in range(t.size - 1):
        dt = t[k + 1] - t[k]
        # Zero-order hold on load between samples, midpoint for the half steps.
        pe0, pe1 = p_load[k], p_load[k + 1]
        pe_mid = 0.5 * (pe0 + pe1)

        args = (rp_pu, tp_sec, h_sec, p_ref, ramp_limit_pu_s)
        k1 = _rhs(state, pe0, *args)
        k2 = _rhs(state + 0.5 * dt * k1, pe_mid, *args)
        k3 = _rhs(state + 0.5 * dt * k2, pe_mid, *args)
        k4 = _rhs(state + dt * k3, pe1, *args)

        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[k + 1] = state

    return out


def load_step(
    t_end: float = 8.0,
    dt: float = 0.002,
    t_step: float = 1.0,
    p_before: float = 0.20,
    p_after: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard load-step stimulus. Returns (t, p_load)."""
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    p = np.where(t < t_step, p_before, p_after)
    return t, p


def frequency_hz(omega_pu: np.ndarray, f_nom: float = 50.0) -> np.ndarray:
    return np.asarray(omega_pu) * f_nom
