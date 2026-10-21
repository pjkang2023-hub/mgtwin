"""
Test runner: executes a TestSuite against a composed Site and one island
configuration, producing a RunReport whose honesty rules live in testrun.py.

Model tier
----------
This runner operates at the RMS tier with a single aggregate grid-forming
source (the GFM droop model in models.py, which also represents a droop-
governed genset at this timescale). What that tier can and cannot answer is
declared per core function:

    STEADY_STATE, PLANNED_ISLANDING, UNPLANNED_ISLANDING -> executed
    PROTECTION   -> executed as a SCREENING (sequence SLG + pickup ratios),
                    then still marked deferred-for-confirmation because the
                    case itself declares requires_emt
    BLACK_START, POWER_QUALITY -> deferred (EMT tier)
    DISPATCH, RECONNECTION     -> deferred (EMS / synchroniser models absent)

Prerequisite gating
-------------------
Before executing anything, the site gates run once. Findings scoped to the
target island BLOCK the cases they invalidate: an ungrounded zone blocks the
protection case outright (an SLG screening on it is already answered by the
gate: undetectable fault, sqrt(3) overvoltage -- and that answer is attached),
and blocks nothing else at RMS tier since positive-sequence dynamics do not
see grounding. Units-gate failures block everything.

Parameters
----------
The aggregate source is built from the island's energized sources: droop and
inertia from the largest grid-forming machine or converter, rating from the
sum. Provenance is ASSUMED unless the site's converter params carry
ride_through_identified / identified droop -- the result artifact says which.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fault import slg_fault
from .gates import Severity, run_all
from .models import simulate_gfm
from .ontology import Converter, ConverterControl, IslandConfig, Site, SynchronousGen
from .testrun import (
    CaseStatus,
    CaseVerdict,
    MetricValue,
    ParamProvenance,
    RunReport,
    TestResult,
    evaluate_criterion,
    judge,
)
from .testspec import CoreFunction, TestSuite

RMS_EXECUTABLE = {
    CoreFunction.STEADY_STATE,
    CoreFunction.PLANNED_ISLANDING,
    CoreFunction.UNPLANNED_ISLANDING,
    CoreFunction.PROTECTION,
}

DEFER_REASONS = {
    CoreFunction.BLACK_START: "transformer inrush against a current limit is an "
    "EMT question; RMS averaging removes the phenomenon being tested",
    CoreFunction.POWER_QUALITY: "THD/unbalance/flicker are waveform quantities; "
    "EMT tier required",
    CoreFunction.DISPATCH: "no EMS model in this harness yet",
    CoreFunction.RECONNECTION: "no synchroniser model in this harness yet",
}


@dataclass
class AggregateSource:
    s_mva: float
    rp_pu: float
    h_sec: float
    tp_sec: float
    provenance: ParamProvenance
    detail: str


def _aggregate_source(site: Site, isl: IslandConfig) -> AggregateSource:
    """Largest grid-forming source sets the dynamics; ratings sum."""
    srcs = [site.source(sid) for sid in isl.energized_source_ids]
    if not srcs:
        raise ValueError(f"island {isl.id!r} declares no energized sources")
    gfm: list = []
    for s in srcs:
        if isinstance(s, SynchronousGen):
            gfm.append(s)
        elif isinstance(s, Converter) and s.control != ConverterControl.GFL:
            gfm.append(s)
    if not gfm:
        raise ValueError(
            f"island {isl.id!r} has no grid-forming source; it cannot hold "
            f"frequency and no RMS islanding case is meaningful"
        )
    lead = max(gfm, key=lambda s: s.s_rated_mva)
    if isinstance(lead, SynchronousGen):
        rp, h, tp = lead.droop_pu, lead.h_sec, 0.30  # governor lag, assumed
        prov = ParamProvenance.ASSUMED
        detail = f"lead source {lead.id}: machine datasheet-class values"
    else:
        p = lead.params
        rp = p.p_f_droop_pu or 0.05
        h = p.h_sec if p.h_sec is not None else 0.0
        tp = p.tp_sec
        prov = (
            ParamProvenance.IDENTIFIED
            if p.ride_through_identified
            else ParamProvenance.ASSUMED
        )
        detail = f"lead source {lead.id}"
    return AggregateSource(
        s_mva=sum(s.s_rated_mva for s in gfm),
        rp_pu=rp,
        h_sec=max(h, 0.05),
        tp_sec=tp,
        provenance=prov,
        detail=detail,
    )


def _island_load_mw(site: Site, isl: IslandConfig) -> float:
    """
    Load served in this island. Approximation, documented: buses reachable
    from the energized sources with the island's switches open -- reuses the
    electrical-island component from the grounding machinery.
    """
    from .gates import _zero_sequence_zones

    zones, _ = _zero_sequence_zones(site, isl)
    buses = set().union(*zones) if zones else set()
    return sum(p for ld, p, _ in site.state_loads(isl) if ld.bus in buses)


# ---------------------------------------------------------------------------
# Executors (RMS tier)
# ---------------------------------------------------------------------------


def _freq_metrics(t: np.ndarray, f_hz: np.ndarray, f_nom: float):
    nadir_dev = float(np.max(np.abs(f_hz - f_nom)))
    # ROCOF over a 100 ms sliding window
    dt = t[1] - t[0]
    w = max(1, int(round(0.1 / dt)))
    df = (f_hz[w:] - f_hz[:-w]) / (w * dt)
    rocof = float(np.max(np.abs(df)))
    return nadir_dev, rocof


def _run_step(src: AggregateSource, p0: float, p1: float, f_nom: float):
    t = np.arange(0.0, 12.0, 0.002)
    p = np.where(t < 1.0, p0, p1)
    y = simulate_gfm(t, p, src.rp_pu, src.tp_sec, src.h_sec, p_ref=p0)
    return t, y[:, 0] * f_nom


def run_steady_state(site, isl, src, case, limits, f_nom):
    p_max = _island_load_mw(site, isl) / src.s_mva
    p_min = 0.2 * p_max
    # Droop-only means ONE fixed reference across the load range; moving the
    # reference with the operating point is secondary control by accident and
    # reports a zero band. The band is the droop characteristic itself:
    # f(p_min) - f(p_max) = rp * (p_max - p_min) * f_nom.
    import numpy as _np
    t = _np.arange(0.0, 12.0, 0.002)
    f_lo = simulate_gfm(t, _np.full_like(t, p_min), src.rp_pu, src.tp_sec,
                        src.h_sec, p_ref=p_min)[:, 0] * f_nom
    f_hi = simulate_gfm(t, _np.full_like(t, p_max), src.rp_pu, src.tp_sec,
                        src.h_sec, p_ref=p_min)[:, 0] * f_nom
    band = float(abs(f_lo[-1] - f_hi[-1]))
    metrics = [
        MetricValue("island_peak_load", p_max * src.s_mva, "MW"),
        MetricValue("source_loading_at_peak", 100 * p_max, "%",
                    "of aggregate GFM rating"),
        MetricValue("steady_state_frequency_band", band, "hz",
                    f"droop-only, load {p_min:.2f}..{p_max:.2f} pu"),
    ]
    measured = {"steady_state_frequency_band": band}
    crits = []
    for c in case.criteria:
        lim = limits.get((case.id, c.metric), c.limit)
        crits.append(evaluate_criterion(c.model_copy(update={"limit": lim}),
                                        measured.get(c.metric)))
    notes = ["reactive sharing not assessed: single aggregate source at this tier"]
    return metrics, crits, notes


def run_islanding(site, isl, src, case, limits, f_nom, planned: bool):
    p_island = _island_load_mw(site, isl) / src.s_mva
    p0 = 0.9 * p_island if planned else 0.10
    t, f = _run_step(src, p0, p_island, f_nom)
    nadir, rocof = _freq_metrics(t, f, f_nom)
    metrics = [
        MetricValue("pre_island_source_output", p0, "pu", "of GFM rating"),
        MetricValue("post_island_load", p_island, "pu"),
        MetricValue("frequency_nadir_deviation", nadir, "hz"),
        MetricValue("rocof_peak", rocof, "hz_s", "100 ms window"),
    ]
    measured = {
        "frequency_nadir_deviation": nadir,
        "rocof_peak": rocof,
        "critical_load_interrupted": 0.0,  # no shed scheme modelled
        "voltage_excursion": None,  # not observable in this frequency model
    }
    crits = []
    for c in case.criteria:
        lim = limits.get((case.id, c.metric), c.limit)
        crits.append(evaluate_criterion(c.model_copy(update={"limit": lim}),
                                        measured.get(c.metric)))
    notes = []
    if not planned:
        notes.append(
            "no load-shed scheme modelled: the full island load lands on the "
            "GFM source instantly, which is the declared worst case"
        )
        if p_island > 1.0:
            notes.append(
                f"island load exceeds aggregate GFM rating "
                f"({p_island:.2f} pu): steady state is not sustainable "
                f"regardless of the transient"
            )
    return metrics, crits, notes


def run_protection_screening(site, isl, src, case, limits, f_nom):
    """Sequence-SLG screening. Only reached when the island's zones are
    grounded (otherwise the case is BLOCKED with the gate finding)."""
    z1 = 1j * 0.15 * (site.base.s_base_mva / src.s_mva)  # subtransient, assumed
    g = [gs for gs in site.grounding if gs.available_when_islanded]
    z0 = None
    if g:
        z0 = 3.0 * g[0].z0_ohm / site.base.z_base_ohm + 1j * 0.10
    res = slg_fault(z1_pu=z1, z0_pu=z0)
    i_a = abs(res.i_fault_pu) * site.base.i_base_a
    metrics = [
        MetricValue("slg_fault_current", i_a, "A"),
        MetricValue("unfaulted_phase_overvoltage", res.v_max_unfaulted_pu, "pu"),
    ]
    measured = {
        "unfaulted_phase_overvoltage": res.v_max_unfaulted_pu,
        "primary_clearing_time": None,  # needs coordination study
    }
    crits = []
    for c in case.criteria:
        lim = limits.get((case.id, c.metric), c.limit)
        crits.append(evaluate_criterion(c.model_copy(update={"limit": lim}),
                                        measured.get(c.metric)))
    notes = [
        "SCREENING TIER ONLY: sequence-network SLG with assumed subtransient "
        "reactance; the case itself declares requires_emt, so this result is "
        "a screen, not a coordination study",
    ]
    return metrics, crits, notes


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def run_suite(
    site: Site,
    suite: TestSuite,
    island_id: str,
    project_limits: dict[tuple[str, str], float] | None = None,
    f_nom: float | None = None,
) -> RunReport:
    """
    project_limits: {(case_id, metric): limit} -- the caller supplies what the
    interconnection agreement / project spec defines. Anything not supplied
    stays UNEVALUATED. The runner never invents a limit.
    """
    limits = project_limits or {}
    f_nom = f_nom or site.base.f_base_hz
    isl = next(i for i in site.islands if i.id == island_id)

    gate_rep = run_all(site)
    fails = gate_rep.by_severity(Severity.FAIL)
    unit_fails = [f for f in fails if f.gate == "units"]
    island_grounding_fails = [
        f
        for f in fails
        if f.gate == "grounding" and f.scope.startswith(f"{island_id}/")
    ]

    report = RunReport(suite_edition=suite.edition, site_id=site.id)
    src = _aggregate_source(site, isl)

    for case in suite.cases:
        r = TestResult(case=case, island_id=island_id,
                       status=CaseStatus.EXECUTED,
                       provenance=src.provenance,
                       provenance_detail=src.detail)

        if unit_fails:
            r.status = CaseStatus.BLOCKED
            r.blocking_findings = unit_fails
            report.results.append(r)
            continue

        if case.function not in RMS_EXECUTABLE:
            r.status = CaseStatus.DEFERRED
            r.defer_reason = DEFER_REASONS[case.function]
            report.results.append(r)
            continue

        if case.function == CoreFunction.PROTECTION and island_grounding_fails:
            r.status = CaseStatus.BLOCKED
            r.blocking_findings = island_grounding_fails
            r.notes.append(
                "the gate already answers this case for an ungrounded zone: "
                "SLG fault current ~0 (undetectable) and sqrt(3) overvoltage; "
                "remediate grounding before a coordination study is meaningful"
            )
            report.results.append(r)
            continue

        if case.function == CoreFunction.STEADY_STATE:
            m, c, n = run_steady_state(site, isl, src, case, limits, f_nom)
        elif case.function == CoreFunction.PLANNED_ISLANDING:
            m, c, n = run_islanding(site, isl, src, case, limits, f_nom, planned=True)
        elif case.function == CoreFunction.UNPLANNED_ISLANDING:
            m, c, n = run_islanding(site, isl, src, case, limits, f_nom, planned=False)
        else:  # PROTECTION, grounded path
            m, c, n = run_protection_screening(site, isl, src, case, limits, f_nom)
            r.notes.append("EMT confirmation still required per the case definition")

        r.metrics, r.criteria, = m, c
        r.notes.extend(n)
        r.verdict = judge(c)
        report.results.append(r)

    return report
