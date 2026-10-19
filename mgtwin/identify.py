"""
Parameter identification with an honest confidence report.

The point of this module is not the optimiser. Any optimiser will return a
number. The point is that the report refuses to claim confidence it has not
earned:

- Fisher-information confidence intervals are only valid if the model form is
  correct. Under model-form mismatch they are optimistically biased, often
  badly, while looking perfectly tight.
- So residual whiteness is tested first, and a failure DOWNGRADES the verdict
  regardless of how small the standard errors are.

An identifier that returns tight intervals on a misspecified model is worse than
one that returns nothing, because it converts an unknown into a false certainty
that then propagates into a protection study.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import stats
from scipy.optimize import least_squares


class Verdict(str, Enum):
    IDENTIFIED = "identified"
    WEAKLY_IDENTIFIED = "weakly_identified"
    NOT_IDENTIFIABLE = "not_identifiable"
    STRUCTURAL_MISMATCH = "structural_mismatch"


#: Relative standard error above which a parameter is treated as not pinned down.
REL_SE_LIMIT = 0.20
#: Condition number of the scaled sensitivity matrix above which the parameter
#: set is treated as practically unidentifiable from this experiment.
COND_LIMIT = 1.0e3
#: Significance level for the residual whiteness tests.
WHITENESS_ALPHA = 0.01


@dataclass
class WhitenessResult:
    ljung_box_q: float
    ljung_box_p: float
    lag1_autocorr: float
    runs_z: float
    runs_p: float
    white: bool

    def render(self) -> str:
        verdict = "white" if self.white else "STRUCTURED"
        return (
            f"residuals: {verdict} | "
            f"Ljung-Box Q={self.ljung_box_q:.1f} (p={self.ljung_box_p:.2e}), "
            f"lag-1 rho={self.lag1_autocorr:+.3f}, "
            f"runs z={self.runs_z:+.1f} (p={self.runs_p:.2e})"
        )


@dataclass
class IdentificationResult:
    names: list[str]
    theta: np.ndarray
    std_err: np.ndarray
    rel_se: np.ndarray
    cond_number: float
    weak_direction: dict[str, float]
    sigma_hat: float
    rmse: float
    whiteness: WhitenessResult
    verdict: Verdict
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"verdict: {self.verdict.value.upper()}"]
        lines.append(f"  {'parameter':<12}{'estimate':>12}{'std err':>12}{'rel se':>10}")
        for n, th, se, rse in zip(self.names, self.theta, self.std_err, self.rel_se):
            flag = "  <-- weak" if rse > REL_SE_LIMIT else ""
            lines.append(f"  {n:<12}{th:>12.5g}{se:>12.3g}{rse:>9.1%}{flag}")
        lines.append(f"  rmse = {self.rmse:.3e}   sigma_hat = {self.sigma_hat:.3e}")
        lines.append(f"  cond(scaled sensitivity) = {self.cond_number:.3e}")
        if self.weak_direction:
            combo = "  ".join(
                f"{n}:{v:+.2f}" for n, v in self.weak_direction.items()
            )
            lines.append(f"  least-determined direction: {combo}")
        lines.append("  " + self.whiteness.render())
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------


def _ljung_box(resid: np.ndarray, lags: int, n_params: int) -> tuple[float, float]:
    """
    Ljung-Box portmanteau test for autocorrelation up to `lags`.

    Q = n(n+2) * sum_{k=1..h} rho_k^2 / (n - k),  Q ~ chi2(h - n_params)
    """
    n = resid.size
    r = resid - resid.mean()
    denom = float(np.dot(r, r))
    q = 0.0
    for k in range(1, lags + 1):
        rho = float(np.dot(r[:-k], r[k:])) / denom
        q += rho * rho / (n - k)
    q *= n * (n + 2)
    dof = max(lags - n_params, 1)
    p = float(stats.chi2.sf(q, dof))
    return q, p


def _runs_test(resid: np.ndarray) -> tuple[float, float]:
    """
    Wald-Wolfowitz runs test on residual signs.

    Structured residuals produce far fewer sign changes than white noise, so a
    large negative z is direct evidence of an unmodelled deterministic trend.
    """
    signs = np.sign(resid - np.median(resid))
    signs = signs[signs != 0]
    n = signs.size
    n_pos = int(np.sum(signs > 0))
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return -np.inf, 0.0

    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    mu = 2.0 * n_pos * n_neg / n + 1.0
    var = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n)) / (n * n * (n - 1.0))
    if var <= 0:
        return 0.0, 1.0
    z = (runs - mu) / np.sqrt(var)
    p = float(2.0 * stats.norm.sf(abs(z)))
    return float(z), p


def assess_whiteness(
    resid: np.ndarray, n_params: int, alpha: float = WHITENESS_ALPHA
) -> WhitenessResult:
    lags = int(min(20, max(5, resid.size // 20)))
    q, p_lb = _ljung_box(resid, lags, n_params)
    r = resid - resid.mean()
    lag1 = float(np.dot(r[:-1], r[1:]) / np.dot(r, r))
    z, p_runs = _runs_test(resid)
    white = (p_lb > alpha) and (p_runs > alpha)
    return WhitenessResult(q, p_lb, lag1, z, p_runs, white)


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------


def identify(
    predict,
    y_meas: np.ndarray,
    theta0: dict[str, float],
    bounds: dict[str, tuple[float, float]],
) -> IdentificationResult:
    """
    Least-squares identification plus an identifiability and validity report.

    `predict(theta_dict) -> np.ndarray` must return a prediction the same shape
    as `y_meas`.
    """
    names = list(theta0.keys())
    x0 = np.array([theta0[n] for n in names], dtype=float)
    lo = np.array([bounds[n][0] for n in names], dtype=float)
    hi = np.array([bounds[n][1] for n in names], dtype=float)
    y_meas = np.asarray(y_meas, dtype=float)

    def resid_fn(x: np.ndarray) -> np.ndarray:
        return predict(dict(zip(names, x))) - y_meas

    sol = least_squares(resid_fn, x0, bounds=(lo, hi), x_scale="jac", method="trf")
    theta = sol.x
    resid = sol.fun
    n, p = resid.size, theta.size

    dof = max(n - p, 1)
    sigma2 = float(np.dot(resid, resid) / dof)
    sigma_hat = float(np.sqrt(sigma2))
    rmse = float(np.sqrt(np.mean(resid**2)))

    # Sensitivity matrix from the optimiser's Jacobian of the residual, which is
    # d(resid)/d(theta) = d(pred)/d(theta).
    jac = np.asarray(sol.jac, dtype=float)

    # Scale to relative sensitivities so the condition number is dimensionless
    # and does not merely reflect the units of the parameters.
    scale = np.where(np.abs(theta) > 0, np.abs(theta), 1.0)
    jac_scaled = jac * scale[None, :]

    # Fisher information and covariance.
    fim = jac.T @ jac / sigma2
    notes: list[str] = []
    try:
        cov = np.linalg.inv(fim)
        std_err = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    except np.linalg.LinAlgError:
        std_err = np.full(p, np.inf)
        notes.append("Fisher information matrix is singular; no finite intervals exist")

    rel_se = np.where(np.abs(theta) > 0, std_err / np.abs(theta), np.inf)

    # Condition number and the least-determined parameter combination.
    svals = np.linalg.svd(jac_scaled, compute_uv=False)
    cond = float(svals[0] / svals[-1]) if svals[-1] > 0 else np.inf
    _, _, vt = np.linalg.svd(jac_scaled, full_matrices=False)
    weak_vec = vt[-1]
    weak_direction = {nm: float(v) for nm, v in zip(names, weak_vec)}

    whiteness = assess_whiteness(resid, p)

    # -- verdict -----------------------------------------------------------
    # Order matters. Model-form mismatch is checked first and dominates, because
    # under mismatch the intervals below are not merely wrong but confidently
    # wrong, and a tight interval is then actively misleading.
    if not whiteness.white:
        verdict = Verdict.STRUCTURAL_MISMATCH
        notes.append(
            "Residuals are structured, so the model form does not match the data. "
            "The Fisher intervals above assume a correct model form and are "
            "therefore optimistically biased -- do not report them as confidence."
        )
        notes.append(
            "Estimates are biased by the missing structure. Add the missing "
            "mechanism (limiter, extra lag, nonlinearity) or restrict the fit to "
            "the region where the library form holds."
        )
    elif cond > COND_LIMIT or not np.all(np.isfinite(rel_se)):
        verdict = Verdict.NOT_IDENTIFIABLE
        dominant = max(weak_direction.items(), key=lambda kv: abs(kv[1]))
        notes.append(
            f"This experiment does not separate the parameters: the direction "
            f"shown above is nearly flat, dominated by {dominant[0]!r}. "
            f"Either fix a parameter from datasheet, or change the stimulus "
            f"(faster step, multi-sine, or a second operating point)."
        )
    elif np.any(rel_se > REL_SE_LIMIT):
        verdict = Verdict.WEAKLY_IDENTIFIED
        weak = [nm for nm, r in zip(names, rel_se) if r > REL_SE_LIMIT]
        notes.append(
            f"Marginal: {', '.join(weak)} exceed {REL_SE_LIMIT:.0%} relative "
            f"standard error. Usable for trend, not for a protection setting."
        )
    else:
        verdict = Verdict.IDENTIFIED

    return IdentificationResult(
        names=names,
        theta=theta,
        std_err=std_err,
        rel_se=rel_se,
        cond_number=cond,
        weak_direction=weak_direction,
        sigma_hat=sigma_hat,
        rmse=rmse,
        whiteness=whiteness,
        verdict=verdict,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Multi-experiment identification: the amplitude-consistency protocol
# ---------------------------------------------------------------------------

#: Cross-experiment parameter spread beyond this many combined standard
#: errors is treated as operating-point dependence.
CONSISTENCY_K = 4.0
#: ...with a relative floor, so ultra-tight fits don't flag ppm differences.
CONSISTENCY_REL_FLOOR = 0.03


@dataclass
class MultiIdentificationResult:
    results: list[IdentificationResult]
    names: list[str]
    theta_pooled: np.ndarray | None
    spread_rel: dict[str, float]
    verdict: Verdict
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"multi-experiment verdict: {self.verdict.value.upper()}"]
        for k, r in enumerate(self.results):
            est = ", ".join(f"{n}={v:.4g}" for n, v in zip(r.names, r.theta))
            lines.append(f"  exp {k}: {r.verdict.value:<20} {est}")
        if self.spread_rel:
            lines.append(
                "  cross-experiment spread: "
                + "  ".join(f"{n}:{s:.1%}" for n, s in self.spread_rel.items())
            )
        if self.theta_pooled is not None:
            lines.append(
                "  pooled: "
                + ", ".join(f"{n}={v:.4g}" for n, v in zip(self.names, self.theta_pooled))
            )
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def identify_multi(
    experiments: list[tuple],
    theta0: dict[str, float],
    bounds: dict[str, tuple[float, float]],
) -> MultiIdentificationResult:
    """
    Identification across multiple experiments at DIFFERENT operating points
    (e.g. two step amplitudes), with a consistency gate.

    `experiments`: list of (predict_fn, y_meas) pairs; each predict_fn maps a
    theta dict to a prediction the shape of its y_meas.

    Rationale: single-experiment residual whiteness is blind to amplitude-
    dependent nonlinearities (deadband, saturation, piecewise droop) because
    on one trajectory they masquerade as a parameter shift with near-white
    residuals -- rung 4 measured a 33% false-confidence rate for exactly
    that class. A linear model form, if correct, must return the SAME
    parameters at every amplitude; disagreement beyond the combined
    uncertainty is structural evidence no single experiment can supply.

    Verdict logic, strictest first:
      * any single experiment STRUCTURAL_MISMATCH / NOT_IDENTIFIABLE
        -> that verdict propagates;
      * parameters inconsistent across experiments -> STRUCTURAL_MISMATCH
        ("operating-point-dependent parameters");
      * any WEAKLY_IDENTIFIED -> WEAKLY_IDENTIFIED;
      * else IDENTIFIED, with an inverse-variance pooled estimate.
    """
    names = list(theta0.keys())
    results = [
        identify(predict, y, theta0, bounds) for predict, y in experiments
    ]

    notes: list[str] = []
    for bad in (Verdict.STRUCTURAL_MISMATCH, Verdict.NOT_IDENTIFIABLE):
        if any(r.verdict == bad for r in results):
            which = [k for k, r in enumerate(results) if r.verdict == bad]
            notes.append(
                f"experiment(s) {which} returned {bad.value}; see their notes"
            )
            return MultiIdentificationResult(
                results, names, None, {}, bad, notes
            )

    # Consistency across experiments.
    theta = np.array([r.theta for r in results])      # (n_exp, n_par)
    se = np.array([r.std_err for r in results])
    spread_rel: dict[str, float] = {}
    inconsistent: list[str] = []
    for j, n in enumerate(names):
        vals = theta[:, j]
        lo, hi = float(vals.min()), float(vals.max())
        mid = 0.5 * (lo + hi)
        rel = (hi - lo) / abs(mid) if mid else np.inf
        spread_rel[n] = rel
        tol = max(
            CONSISTENCY_K * float(np.sqrt(np.sum(se[:, j] ** 2))),
            CONSISTENCY_REL_FLOOR * abs(mid),
        )
        if (hi - lo) > tol:
            inconsistent.append(n)

    if inconsistent:
        notes.append(
            f"parameters {inconsistent} differ across operating points beyond "
            f"the combined uncertainty: the plant is nonlinear in a way the "
            f"model form does not contain (deadband, limiter, or piecewise "
            f"characteristic). Each single-experiment fit looked clean -- "
            f"this is precisely the mismatch class residual whiteness cannot "
            f"see, and why the multi-amplitude protocol exists."
        )
        return MultiIdentificationResult(
            results, names, None, spread_rel, Verdict.STRUCTURAL_MISMATCH, notes
        )

    # Pool (inverse-variance).
    w = 1.0 / np.clip(se, 1e-12, None) ** 2
    pooled = np.sum(w * theta, axis=0) / np.sum(w, axis=0)

    if any(r.verdict == Verdict.WEAKLY_IDENTIFIED for r in results):
        verdict = Verdict.WEAKLY_IDENTIFIED
        notes.append("at least one experiment only weakly identified")
    else:
        verdict = Verdict.IDENTIFIED
    return MultiIdentificationResult(
        results, names, pooled, spread_rel, verdict, notes
    )
