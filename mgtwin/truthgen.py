"""
Truth generators for identification evals.

Each generator produces a load-step response from a plant that either IS the
library form (nominal) or contains a mechanism the library form deliberately
lacks -- the mechanisms real gensets and BESS actually have and datasheets
don't mention. The identifier is then scored on whether it REFUSES when the
form is wrong, not on how close its (necessarily biased) estimates land.

Kept separate from models.py on purpose: models.py is the clean library form
the identifier fits; this module is the adversary. The two must never share
an RHS, or the eval tests the copy machine.

Mechanisms
----------
ramp_limit     BESS power ramp limiter (also in models.py for the demo)
deadband       governor frequency deadband: no droop response inside +/-db
saturation     output power clamp at p_max: the step drives the source
               into its limit and frequency departs the droop line entirely
extra_lag      a second lag in the power path (actuator / turbine): the
               response is 3rd order where the library form is 2nd
knee_droop     piecewise droop: nominal slope inside a band, steeper beyond
               (FFR-style curves)
fast_unmodeled a small 10 ms unmodeled lag: MINOR mismatch whose bias is
               negligible -- the over-refusal probe
quantized      nominal plant, measurement quantized to a realistic PMU/relay
               step -- the measurement-imperfection probe
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlantParams:
    rp_pu: float = 0.05
    tp_sec: float = 0.20
    h_sec: float = 2.0
    p_ref: float = 0.0
    # -- mechanisms the library form does not contain -----------------------
    ramp_limit_pu_s: float | None = None
    deadband_hz: float | None = None       # governor deadband, +/- in Hz
    p_max_pu: float | None = None          # output clamp
    extra_lag_sec: float | None = None     # second power-path lag
    knee: tuple[float, float] | None = None  # (band_hz, rp_outside)
    f_nom: float = 60.0


def _rhs(state: np.ndarray, p_e: float, pp: PlantParams) -> np.ndarray:
    """
    State: [omega_pu, p_m, (p_m2 if extra_lag)].

    Same droop convention as models.simulate_gfm; the mechanisms wrap it.
    """
    omega, p_m = state[0], state[1]
    p_for_swing = state[2] if pp.extra_lag_sec is not None else p_m

    # droop / governor term with optional deadband and knee
    d_omega_pu = omega - 1.0
    d_f_hz = d_omega_pu * pp.f_nom
    rp = pp.rp_pu
    gov_err = d_omega_pu
    if pp.deadband_hz is not None:
        if abs(d_f_hz) <= pp.deadband_hz:
            gov_err = 0.0
        else:
            gov_err = (d_f_hz - np.sign(d_f_hz) * pp.deadband_hz) / pp.f_nom
    if pp.knee is not None:
        band_hz, rp_out = pp.knee
        if abs(d_f_hz) > band_hz:
            rp = rp_out

    d_omega = ((pp.p_ref - p_for_swing) - gov_err / rp) / (2.0 * pp.h_sec)

    d_pm = (p_e - p_m) / pp.tp_sec
    if pp.ramp_limit_pu_s is not None:
        d_pm = float(np.clip(d_pm, -pp.ramp_limit_pu_s, pp.ramp_limit_pu_s))
    if pp.p_max_pu is not None and p_m >= pp.p_max_pu and d_pm > 0:
        d_pm = 0.0

    out = [d_omega, d_pm]
    if pp.extra_lag_sec is not None:
        p_m2 = state[2]
        out.append((p_m - p_m2) / pp.extra_lag_sec)
    return np.array(out)


def simulate_plant(t: np.ndarray, p_load: np.ndarray, pp: PlantParams) -> np.ndarray:
    """RK4, fixed step; returns omega_pu(t)."""
    t = np.asarray(t, float)
    p_load = np.asarray(p_load, float)
    n_state = 3 if pp.extra_lag_sec is not None else 2
    state = np.zeros(n_state)
    state[0] = 1.0
    state[1] = p_load[0]
    if n_state == 3:
        state[2] = p_load[0]

    omega = np.empty(t.size)
    omega[0] = state[0]
    for k in range(t.size - 1):
        dt = t[k + 1] - t[k]
        pe0, pe1 = p_load[k], p_load[k + 1]
        pem = 0.5 * (pe0 + pe1)
        k1 = _rhs(state, pe0, pp)
        k2 = _rhs(state + 0.5 * dt * k1, pem, pp)
        k3 = _rhs(state + 0.5 * dt * k2, pem, pp)
        k4 = _rhs(state + dt * k3, pe1, pp)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        omega[k + 1] = state[0]
    return omega


# ---------------------------------------------------------------------------
# Scenario battery
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    params: PlantParams
    mismatch: bool               # is the library form structurally wrong?
    quantize_pu: float | None = None  # measurement quantization step
    note: str = ""
    step: tuple[float, float] = (0.20, 0.60)


def battery() -> list[Scenario]:
    base = dict(rp_pu=0.05, tp_sec=0.20, h_sec=2.0)
    return [
        Scenario("nominal", PlantParams(**base), mismatch=False,
                 note="library form exact; baseline for true confidence"),
        Scenario("nominal_high_noise", PlantParams(**base), mismatch=False,
                 note="same plant, 5x measurement noise (false-alarm probe)"),
        Scenario("quantized", PlantParams(**base), mismatch=False,
                 quantize_pu=1e-4,
                 note="nominal plant, 6 mHz measurement quantization"),
        Scenario("fast_unmodeled", PlantParams(**base, extra_lag_sec=0.005),
                 mismatch=False,
                 note="5 ms unmodeled lag: bias ~2.5%, genuinely minor; "
                      "counts as over-refusal if refused"),
        Scenario("ramp_limit", PlantParams(**base, ramp_limit_pu_s=0.30),
                 mismatch=True, note="BESS ramp limiter vs 0.4 pu step"),
        Scenario("deadband", PlantParams(**base, deadband_hz=0.20),
                 mismatch=True, note="governor deadband +/-0.2 Hz"),
        Scenario("saturation", PlantParams(**base, p_max_pu=0.50),
                 mismatch=True, step=(0.20, 0.60),
                 note="output clamp at 0.5 pu; step demands 0.6"),
        Scenario("extra_lag", PlantParams(**base, extra_lag_sec=0.15),
                 mismatch=True,
                 note="150 ms actuator lag: 3rd-order truth, 2nd-order fit"),
        Scenario("knee_droop", PlantParams(**base, knee=(0.15, 0.02)),
                 mismatch=True,
                 note="droop steepens to 2% beyond +/-0.15 Hz (FFR-style)"),
    ]


def make_measurement(
    sc: Scenario,
    seed: int,
    dt_sim: float = 0.002,
    t_end: float = 8.0,
    dt_meas: float = 0.02,
    sigma: float = 2.0e-4,
):
    """Simulate the truth plant and sample it PMU-style."""
    t = np.arange(0.0, t_end + 0.5 * dt_sim, dt_sim)
    p0, p1 = sc.step
    p = np.where(t < 1.0, p0, p1)
    pp = sc.params
    pp = PlantParams(**{**pp.__dict__, "p_ref": p0})
    omega = simulate_plant(t, p, pp)
    idx = np.arange(0, t.size, int(round(dt_meas / dt_sim)))
    rng = np.random.default_rng(seed)
    noise = sigma * (5.0 if sc.name == "nominal_high_noise" else 1.0)
    y = omega[idx] + rng.normal(0.0, noise, idx.size)
    if sc.quantize_pu is not None:
        y = np.round(y / sc.quantize_pu) * sc.quantize_pu
    return t, p, idx, y
