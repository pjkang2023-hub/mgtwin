"""
Identification eval harness.

Scores the identifier's REFUSAL BEHAVIOUR, not its parameter recovery.
Under model-form mismatch the estimates are biased no matter how good the
optimiser is; the only question that matters for the product is whether the
gate says so. Each (scenario, seed) run lands in one cell:

                          | bias < 5%            | bias >= 5%
    verdict = IDENTIFIED  | TRUE CONFIDENCE      | FALSE CONFIDENCE  <- forbidden
    verdict refuses       | OVER-REFUSAL         | TRUE REFUSAL

"Refuses" means WEAKLY_IDENTIFIED, NOT_IDENTIFIABLE or STRUCTURAL_MISMATCH:
anything short of a clean IDENTIFIED is treated as the gate declining to
certify. FALSE CONFIDENCE -- a clean verdict on estimates that are badly
wrong -- is the cell that converts an unknown into a false certainty and
then into a protection setting. The product claim is that this cell is
empty on the battery, and the claim is regression-pinned.

OVER-REFUSAL is the cost side: a gate that cries wolf on harmless
imperfections (quantization, a 10 ms parasitic lag) gets ignored by its
users. It is reported, not forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .identify import Verdict, identify, identify_multi
from .models import simulate_gfm
from .truthgen import Scenario, battery, make_measurement

BIAS_LIMIT = 0.05  # 5% relative error on any parameter = "badly wrong"

REFUSALS = {
    Verdict.WEAKLY_IDENTIFIED,
    Verdict.NOT_IDENTIFIABLE,
    Verdict.STRUCTURAL_MISMATCH,
}


@dataclass
class EvalRun:
    scenario: str
    mismatch: bool
    seed: int
    verdict: Verdict
    bias: dict[str, float]
    cell: str  # true_confidence / false_confidence / true_refusal / over_refusal

    @property
    def max_bias(self) -> float:
        return max(abs(b) for b in self.bias.values())


@dataclass
class EvalReport:
    runs: list[EvalRun] = field(default_factory=list)

    def cell_counts(self) -> dict[str, int]:
        out = {"true_confidence": 0, "false_confidence": 0,
               "true_refusal": 0, "over_refusal": 0}
        for r in self.runs:
            out[r.cell] += 1
        return out

    def false_confidence_rate(self) -> float:
        c = self.cell_counts()
        n = sum(c.values())
        return c["false_confidence"] / n if n else 0.0

    def by_scenario(self) -> dict[str, list[EvalRun]]:
        out: dict[str, list[EvalRun]] = {}
        for r in self.runs:
            out.setdefault(r.scenario, []).append(r)
        return out

    def render(self) -> str:
        lines = [
            f"{'scenario':<20}{'truth':<10}{'seeds':>6}"
            f"{'refused':>9}{'max bias':>10}   verdict distribution",
        ]
        for name, runs in self.by_scenario().items():
            refused = sum(1 for r in runs if r.verdict in REFUSALS)
            worst = max(r.max_bias for r in runs)
            dist: dict[str, int] = {}
            for r in runs:
                dist[r.verdict.value] = dist.get(r.verdict.value, 0) + 1
            dist_s = ", ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
            truth = "MISMATCH" if runs[0].mismatch else "nominal"
            lines.append(
                f"{name:<20}{truth:<10}{len(runs):>6}{refused:>9}"
                f"{worst:>9.1%}   {dist_s}"
            )
        c = self.cell_counts()
        n = sum(c.values())
        lines.append("")
        lines.append("confusion (verdict vs consequence):")
        lines.append(f"{'':24}{'bias < 5%':>14}{'bias >= 5%':>14}")
        lines.append(
            f"{'  IDENTIFIED':<24}{c['true_confidence']:>14}"
            f"{c['false_confidence']:>14}   <- forbidden cell"
        )
        lines.append(
            f"{'  refused':<24}{c['over_refusal']:>14}{c['true_refusal']:>14}"
        )
        lines.append(
            f"\nfalse-confidence rate: {c['false_confidence']}/{n} = "
            f"{self.false_confidence_rate():.1%}"
            f"    over-refusal rate: {c['over_refusal']}/{n} = "
            f"{c['over_refusal'] / n:.1%}"
        )
        return "\n".join(lines)


#: Second amplitude for the multi-experiment protocol: same initial point,
#: a deliberately smaller excursion, so amplitude-dependent mechanisms
#: (deadband, clamp, knee) present different effective parameters.
SMALL_STEP = (0.20, 0.35)


def _experiment(sc: Scenario, seed: int, step: tuple[float, float]):
    from dataclasses import replace as _replace

    sc2 = _replace(sc, step=step)
    t, p, idx, y = make_measurement(sc2, seed)
    p0 = step[0]

    def predict(theta: dict[str, float]) -> np.ndarray:
        return simulate_gfm(
            t, p, theta["rp_pu"], theta["tp_sec"], sc.params.h_sec, p_ref=p0
        )[idx, 0]

    return predict, y


def run_one(sc: Scenario, seed: int, protocol: str = "single") -> EvalRun:
    truth = {"rp_pu": sc.params.rp_pu, "tp_sec": sc.params.tp_sec}
    theta0 = {"rp_pu": 0.03, "tp_sec": 0.40}
    bounds = {"rp_pu": (0.005, 0.20), "tp_sec": (0.01, 2.0)}

    if protocol == "single":
        predict, y = _experiment(sc, seed, sc.step)
        res = identify(predict, y, theta0, bounds)
        theta_named = dict(zip(res.names, res.theta))
        verdict = res.verdict
    else:
        exps = [
            _experiment(sc, seed, sc.step),
            _experiment(sc, seed + 1000, SMALL_STEP),
        ]
        mres = identify_multi(exps, theta0, bounds)
        verdict = mres.verdict
        if mres.theta_pooled is not None:
            theta_named = dict(zip(mres.names, mres.theta_pooled))
        else:
            # refused: score bias on the large-step fit (what a naive user
            # would have taken)
            r0 = mres.results[0]
            theta_named = dict(zip(r0.names, r0.theta))
    bias = {n: (theta_named[n] - truth[n]) / truth[n] for n in truth}
    bad = max(abs(b) for b in bias.values()) >= BIAS_LIMIT
    refused = verdict in REFUSALS
    cell = (
        "true_refusal" if (refused and bad)
        else "over_refusal" if refused
        else "false_confidence" if bad
        else "true_confidence"
    )
    return EvalRun(sc.name, sc.mismatch, seed, verdict, bias, cell)


def run_battery(seeds: range = range(10), protocol: str = "single") -> EvalReport:
    rep = EvalReport()
    for sc in battery():
        for seed in seeds:
            rep.runs.append(run_one(sc, seed, protocol=protocol))
    return rep
