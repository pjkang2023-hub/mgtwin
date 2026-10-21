"""
Regression tests.

The tests that matter here are the ones asserting that a gate FAILS. A harness
whose gates quietly stop firing is worse than no harness, because the report
still looks clean. So the negative cases are tested at least as carefully as the
positive ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mgtwin import (  # noqa: E402
    Base,
    Load,
    Bus,
    Converter,
    ConverterControl,
    ConverterParams,
    GroundingKind,
    GroundingSource,
    IslandConfig,
    OvercurrentRelay,
    Severity,
    Site,
    Switch,
    Verdict,
    gate_grounding,
    gate_protection,
    gate_units,
    identify,
    load_step,
    load_suite,
    simulate_gfm,
)
from mgtwin.identify import assess_whiteness  # noqa: E402

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_per_unit_base_arithmetic():
    b = Base(s_base_mva=10.0, v_base_kv=11.0, f_base_hz=50.0)
    assert b.z_base_ohm == pytest.approx(121.0 / 10.0)
    assert b.i_base_a == pytest.approx(10e6 / (3**0.5 * 11e3), rel=1e-9)
    # Round trip.
    assert b.pu_current(b.amps(1.37)) == pytest.approx(1.37)
    assert b.pu_impedance(b.ohms(0.08)) == pytest.approx(0.08)


def test_rebase_impedance():
    old = Base(s_base_mva=100.0, v_base_kv=11.0)
    new = Base(s_base_mva=10.0, v_base_kv=11.0)
    # Ten times smaller MVA base -> ten times smaller pu impedance.
    assert new.rebase_impedance(0.50, old) == pytest.approx(0.05)


def test_invalid_base_rejected():
    with pytest.raises(ValueError):
        Base(s_base_mva=-1.0, v_base_kv=11.0)
    with pytest.raises(ValueError):
        Base(s_base_mva=10.0, v_base_kv=11.0, f_base_hz=45.0)


# ---------------------------------------------------------------------------
# Ontology integrity
# ---------------------------------------------------------------------------


def _minimal_site(**overrides) -> Site:
    kwargs = dict(
        id="T",
        base=Base(s_base_mva=10.0, v_base_kv=11.0),
        buses=[Bus(id="B1", v_nom_kv=11.0)],
    )
    kwargs.update(overrides)
    return Site(**kwargs)


def test_dangling_bus_reference_rejected():
    with pytest.raises(ValueError, match="unknown bus"):
        _minimal_site(
            converters=[
                Converter(
                    id="C1",
                    bus="NOPE",
                    s_rated_mva=1.0,
                    control=ConverterControl.GFL,
                    params=ConverterParams(),
                )
            ]
        )


def test_gfm_without_droop_rejected():
    with pytest.raises(ValueError, match="no P-f droop"):
        Converter(
            id="C1",
            bus="B1",
            s_rated_mva=1.0,
            control=ConverterControl.GFM_DROOP,
            params=ConverterParams(),
        )


def test_vsm_without_inertia_rejected():
    with pytest.raises(ValueError, match="no inertia"):
        Converter(
            id="C1",
            bus="B1",
            s_rated_mva=1.0,
            control=ConverterControl.GFM_VSM,
            params=ConverterParams(p_f_droop_pu=0.05),
        )


def test_solid_grounding_with_large_impedance_rejected():
    with pytest.raises(ValueError, match="solid grounding"):
        GroundingSource(
            id="G",
            bus="B1",
            kind=GroundingKind.SOLID,
            z0_ohm=25.0,
            available_when_islanded=True,
        )


def test_steady_limit_above_transient_rejected():
    with pytest.raises(ValueError, match="steady-state current limit"):
        ConverterParams(i_limit_transient_pu=1.2, i_limit_steady_pu=1.5)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def test_units_gate_catches_ohms_in_a_pu_field():
    """A 12.1 ohm feeder entered as 12.1 pu must not pass."""
    site = _minimal_site(
        buses=[Bus(id="B1", v_nom_kv=11.0), Bus(id="B2", v_nom_kv=11.0)],
        branches=[
            dict(
                id="F1",
                from_bus="B1",
                to_bus="B2",
                r_pu=0.01,
                x_pu=12.1,  # ohms, not pu
                s_base_mva=10.0,
                v_base_kv=11.0,
                rating_mva=10.0,
            )
        ],
    )
    rep = gate_units(site)
    assert rep.failed
    assert any("plausible range" in f.message for f in rep.by_severity(Severity.FAIL))


def test_units_gate_catches_transformer_ratio_mismatch():
    from mgtwin import Transformer, WindingConfig

    site = _minimal_site(
        buses=[Bus(id="HV", v_nom_kv=33.0), Bus(id="LV", v_nom_kv=11.0)],
        transformers=[
            Transformer(
                id="TX",
                hv_bus="HV",
                lv_bus="LV",
                s_rated_mva=15.0,
                hv_kv=33.0,
                lv_kv=6.6,  # wrong: bus is 11 kV
                x_pu=0.08,
                hv_config=WindingConfig.WYE_GROUNDED,
                lv_config=WindingConfig.DELTA,
            )
        ],
    )
    rep = gate_units(site)
    assert rep.failed


def test_grounding_gate_fails_when_reference_is_outside_the_microgrid():
    site = _minimal_site(
        converters=[
            Converter(
                id="C1",
                bus="B1",
                s_rated_mva=1.0,
                control=ConverterControl.GFM_DROOP,
                params=ConverterParams(p_f_droop_pu=0.05),
            )
        ],
        grounding=[
            GroundingSource(
                id="G",
                bus="B1",
                kind=GroundingKind.SOLID,
                available_when_islanded=False,
            )
        ],
        islands=[
            IslandConfig(
                id="I1", grid_connected=False, energized_source_ids=["C1"]
            )
        ],
    )
    rep = gate_grounding(site)
    assert rep.failed
    fail = rep.by_severity(Severity.FAIL)[0]
    assert "zero-sequence" in fail.message
    assert fail.scope.startswith("I1/zone")


def test_grounding_zones_split_by_delta_winding():
    """
    The Banshee lesson as a unit test: a DYg transformer grounds its LV zone
    and nothing else. The MV zone with the source must FAIL even though the
    LV zone passes -- a site-level check would have been fooled.
    """
    from mgtwin import Transformer, WindingConfig

    site = _minimal_site(
        buses=[Bus(id="MV", v_nom_kv=13.8), Bus(id="LV", v_nom_kv=0.48)],
        transformers=[
            Transformer(
                id="TX",
                hv_bus="MV",
                lv_bus="LV",
                s_rated_mva=2.0,
                hv_kv=13.8,
                lv_kv=0.48,
                x_pu=0.0575,
                hv_config=WindingConfig.DELTA,
                lv_config=WindingConfig.WYE_GROUNDED,
            )
        ],
        converters=[
            Converter(
                id="GFM",
                bus="MV",
                s_rated_mva=2.0,
                control=ConverterControl.GFM_DROOP,
                params=ConverterParams(p_f_droop_pu=0.05),
            )
        ],
        loads=[Load(id="L", bus="LV", p_mw=0.5)],
        islands=[
            IslandConfig(id="I1", grid_connected=False, energized_source_ids=["GFM"])
        ],
    )
    rep = gate_grounding(site)
    findings = {f.scope: f.severity for f in rep.findings}
    lv = [s for s in findings if "0.48" in s]
    mv = [s for s in findings if "13.8" in s]
    assert lv and findings[lv[0]] == Severity.PASS
    assert mv and findings[mv[0]] == Severity.FAIL


def test_grounding_gate_passes_when_reference_survives_islanding():
    site = _minimal_site(
        grounding=[
            GroundingSource(
                id="G",
                bus="B1",
                kind=GroundingKind.LOW_RESISTANCE,
                z0_ohm=12.0,
                available_when_islanded=True,
            )
        ],
        islands=[IslandConfig(id="I1", grid_connected=False)],
    )
    rep = gate_grounding(site)
    assert not rep.failed


def test_grounding_gate_fails_when_nothing_declared():
    """Silence is not a pass. An undeclared grounding scheme is a finding."""
    site = _minimal_site(islands=[IslandConfig(id="I1", grid_connected=False)])
    assert gate_grounding(site).failed


def _converter_only_site(pickup_a: float) -> Site:
    return _minimal_site(
        converters=[
            Converter(
                id="BESS",
                bus="B1",
                s_rated_mva=5.0,
                control=ConverterControl.GFM_DROOP,
                params=ConverterParams(p_f_droop_pu=0.05, i_limit_transient_pu=1.2),
            )
        ],
        switches=[Switch(id="SW", from_bus="B1", to_bus="B1")],
        relays=[
            OvercurrentRelay(
                id="R", switch_id="SW", ct_ratio=600.0, pickup_a=pickup_a, time_dial=0.3
            )
        ],
        islands=[
            IslandConfig(id="I1", grid_connected=False, energized_source_ids=["BESS"])
        ],
    )


def test_protection_gate_fails_below_pickup():
    # 5 MVA at 11 kV -> 262 A rated, x1.2 limit -> 315 A available.
    rep = gate_protection(_converter_only_site(pickup_a=600.0))
    assert rep.failed
    assert any("never operate" in f.message for f in rep.by_severity(Severity.FAIL))


def test_protection_gate_fails_inside_dependability_margin():
    """Above pickup but below margin is still a FAIL, not a warning."""
    rep = gate_protection(_converter_only_site(pickup_a=250.0))  # ratio ~1.26
    assert rep.failed
    assert any("dependability margin" in f.message for f in rep.by_severity(Severity.FAIL))


def test_protection_gate_passes_with_adequate_margin():
    rep = gate_protection(_converter_only_site(pickup_a=150.0))  # ratio ~2.1
    assert not rep.failed


def test_protection_gate_warns_on_converter_dominance_even_when_passing():
    rep = gate_protection(_converter_only_site(pickup_a=150.0))
    assert any("current-limited converters" in f.message for f in rep.by_severity(Severity.WARN))


def test_converter_fault_contribution_is_current_limited_not_subtransient():
    """
    A converter must not be modelled as a voltage source behind reactance.
    5 MVA at 11 kV is 262 A rated; contribution should be ~315 A, not thousands.
    """
    base = Base(s_base_mva=10.0, v_base_kv=11.0)
    c = Converter(
        id="C",
        bus="B1",
        s_rated_mva=5.0,
        control=ConverterControl.GFM_DROOP,
        params=ConverterParams(p_f_droop_pu=0.05, i_limit_transient_pu=1.2),
    )
    assert c.fault_contribution_a(base) == pytest.approx(315.0, rel=0.01)


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------


def test_whiteness_accepts_white_noise():
    rng = np.random.default_rng(0)
    assert assess_whiteness(rng.normal(size=500), n_params=2).white


def test_whiteness_rejects_a_deterministic_trend():
    t = np.linspace(0, 1, 500)
    rng = np.random.default_rng(0)
    resid = 0.5 * np.sin(2 * np.pi * 2 * t) + 0.05 * rng.normal(size=t.size)
    assert not assess_whiteness(resid, n_params=2).white


# ---------------------------------------------------------------------------
# Identification: the honesty properties
# ---------------------------------------------------------------------------

RP, TP, H = 0.05, 0.20, 2.0


def _fit(dt_meas, sigma, t_end, ramp_limit, theta0, bounds, fixed, seed=7):
    t, p = load_step(t_end=t_end, dt=0.002)
    truth = simulate_gfm(t, p, RP, TP, H, ramp_limit_pu_s=ramp_limit)
    idx = np.arange(0, t.size, int(round(dt_meas / 0.002)))
    rng = np.random.default_rng(seed)
    y = truth[idx, 0] + rng.normal(0.0, sigma, idx.size)

    def predict(theta):
        return simulate_gfm(
            t,
            p,
            theta.get("rp_pu", fixed.get("rp_pu")),
            theta.get("tp_sec", fixed.get("tp_sec")),
            theta.get("h_sec", fixed.get("h_sec")),
        )[idx, 0]

    return identify(predict, y, theta0, bounds)


def test_well_posed_case_recovers_truth():
    res = _fit(
        0.02,
        2e-4,
        8.0,
        None,
        {"rp_pu": 0.03, "tp_sec": 0.40},
        {"rp_pu": (0.005, 0.20), "tp_sec": (0.01, 2.0)},
        {"h_sec": H},
    )
    assert res.verdict == Verdict.IDENTIFIED
    assert res.theta[0] == pytest.approx(RP, rel=0.02)
    assert res.theta[1] == pytest.approx(TP, rel=0.05)
    assert res.whiteness.white


def test_historian_grade_data_is_reported_not_identifiable():
    """
    The critical property: when the experiment cannot support the parameter set,
    the gate must refuse rather than return the optimiser's answer.
    """
    res = _fit(
        1.0,
        1e-3,
        20.0,
        None,
        {"rp_pu": 0.03, "tp_sec": 0.40, "h_sec": 1.0},
        {"rp_pu": (0.005, 0.20), "tp_sec": (0.01, 2.0), "h_sec": (0.05, 20.0)},
        {},
    )
    assert res.verdict == Verdict.NOT_IDENTIFIABLE
    assert res.cond_number > 1e3
    # And the point of the refusal: the returned estimate is badly wrong.
    assert abs(res.theta[1] - TP) / TP > 0.5


def test_model_form_mismatch_overrides_tight_intervals():
    """
    The single most important behaviour in the package. Under misspecification
    the Fisher intervals are tight and wrong, so the verdict must be driven by
    residual structure, not by interval width.
    """
    res = _fit(
        0.02,
        2e-4,
        8.0,
        0.30,
        {"rp_pu": 0.03, "tp_sec": 0.40},
        {"rp_pu": (0.005, 0.20), "tp_sec": (0.01, 2.0)},
        {"h_sec": H},
        seed=11,
    )
    assert res.verdict == Verdict.STRUCTURAL_MISMATCH
    assert not res.whiteness.white
    # Intervals look reassuring...
    assert res.rel_se[1] < 0.05
    # ...while the estimate is badly biased. That is the trap being closed.
    assert abs(res.theta[1] - TP) / TP > 0.5
    assert any("optimistically biased" in n for n in res.notes)


# ---------------------------------------------------------------------------
# Model numerics
# ---------------------------------------------------------------------------


def test_droop_steady_state_matches_analytic_value():
    """omega_ss = 1 - rp * P_load. Sign errors here invert the whole response."""
    t, p = load_step(t_end=30.0, dt=0.002, p_before=0.30, p_after=0.30)
    y = simulate_gfm(t, p, RP, TP, H)
    assert y[-1, 0] == pytest.approx(1.0 - RP * 0.30, abs=1e-6)
    assert y[-1, 1] == pytest.approx(0.30, abs=1e-6)


def test_load_increase_lowers_frequency():
    t, p = load_step()
    y = simulate_gfm(t, p, RP, TP, H)
    assert y[-1, 0] < y[0, 0]


def test_ramp_limit_slows_the_response():
    t, p = load_step()
    fast = simulate_gfm(t, p, RP, TP, H)
    slow = simulate_gfm(t, p, RP, TP, H, ramp_limit_pu_s=0.30)
    # Same steady state, later arrival.
    assert slow[-1, 0] == pytest.approx(fast[-1, 0], abs=1e-4)
    i = np.searchsorted(t, 1.3)
    assert slow[i, 1] < fast[i, 1]


def test_simulator_rejects_non_monotonic_time():
    with pytest.raises(ValueError):
        simulate_gfm(np.array([0.0, 0.1, 0.05]), np.zeros(3), RP, TP, H)


# ---------------------------------------------------------------------------
# Test spec
# ---------------------------------------------------------------------------


def test_suite_loads_and_carries_no_invented_limits():
    suite = load_suite("ieee2030_8_2018")
    assert suite.cases
    # Every clause reference must be blank: none is reproduced from the standard.
    assert all(c.clause == "" for c in suite.cases)
    # Any limit that IS set must not claim the standard as its source unless
    # it is a structural zero (e.g. "no critical load interrupted").
    for c in suite.cases:
        for cr in c.criteria:
            if cr.limit is not None:
                assert cr.limit_source != "standard" or cr.limit == 0.0


def test_emt_cases_are_flagged():
    suite = load_suite("ieee2030_8_2018")
    ids = {c.id for c in suite.emt_cases()}
    # Black start inrush and fault coordination cannot be settled at RMS.
    assert {"BLK-01", "PROT-01"} <= ids


# ---------------------------------------------------------------------------
# Rung 1: Banshee translation and composition (skipped if data absent)
# ---------------------------------------------------------------------------

import os as _os

BANSHEE_DATA = Path(
    _os.environ.get(
        "BANSHEE_DATA",
        "/home/claude/banshee_raw/Supporting Data for Banshee Benchmark Paper/"
        "00 models and data for system verification",
    )
)

banshee = pytest.mark.skipif(
    not BANSHEE_DATA.exists(), reason="Banshee EPHCC release not present"
)


@banshee
def test_banshee_case_translates_with_expected_shape():
    from mgtwin.matpower_io import parse_matpower_m

    for name, load_mw in (
        ("case_banshee_feeder_all_v5.m", 12.825),
        ("case_banshee_feeder_all_v4.m", 0.0),  # v4 carries loads as shunts
    ):
        mpc = parse_matpower_m(BANSHEE_DATA / "Matpower" / name)
        assert mpc["baseMVA"] == 20.0
        assert mpc["bus"].shape == (40, 13)
        assert mpc["branch"].shape[0] == 39
        assert mpc["bus"][:, 2].sum() == pytest.approx(load_mw, abs=1e-6)
        kvs = sorted(set(mpc["bus"][:, 9]))
        assert kvs == pytest.approx([0.208, 0.48, 4.16, 13.8])


@banshee
def test_banshee_v4_reproduces_published_matpower_currents():
    """
    The rung-1 acceptance test: solve the translated v4 case and require the
    large published currents (>400 A) to reproduce to <0.05%, excluding the
    bus-9 subtree whose demand the release itself marks as a placeholder.
    """
    import csv as _csv

    from examples.banshee_rung1 import (
        CASE_V4,
        CSV_PATH,
        _feeder_of_end,
        solve_and_currents,
    )

    _, cands = solve_and_currents(CASE_V4)
    placeholder_subtree = {"8-9:from", "8-9:to", "2-6:from", "6-7:from"}
    checked = 0
    with CSV_PATH.open() as fh:
        for row in _csv.DictReader(fh):
            m_a = float(row["MATPOWER_A"])
            if m_a < 400.0:
                continue
            fdr = int(row["METER"][2])
            pool = [
                (bn, cv) for bn, cv in cands if _feeder_of_end(bn) == fdr
            ]
            bn, cv = min(pool, key=lambda bc: abs(bc[1] - m_a))
            if bn in placeholder_subtree:
                continue
            assert abs(cv - m_a) / m_a < 5e-4, (row["METER"], bn, m_a, cv)
            checked += 1
    assert checked >= 4  # CB219, CB309, CB218, CB308, CB212, CB307...


@banshee
def test_banshee_composition_is_blocked_for_the_right_reasons():
    """
    Composing from the release alone MUST block, and specifically on grounding:
    positive-sequence data cannot carry it. If this test ever passes gates, the
    pipeline has started inventing zero-sequence facts.
    """
    from mgtwin import run_all
    from mgtwin.banshee import compose_banshee
    from examples.banshee_rung1 import CASE_V5, RELAYS

    from examples.banshee_rung1 import MDL

    site = compose_banshee(
        CASE_V5, relay_dir=RELAYS, mdl_path=MDL if MDL.exists() else None
    )
    assert len(site.buses) == 40
    assert len(site.loads) == 18
    assert len(site.relays) == 3
    rep = run_all(site)
    assert rep.failed
    fails = {f.gate for f in rep.by_severity(Severity.FAIL)}
    assert "grounding" in fails
    assert "protection" in fails  # PV-only island vs release relay settings
    if MDL.exists():
        # With winding data the finding sharpens: 22 DYg transformers, and the
        # grounding FAILs are exactly the four islanded 13.8 kV zones -- every
        # LV pocket passes on its own service transformer Yg.
        assert len(site.transformers) == 22
        g_fails = [
            f for f in rep.by_severity(Severity.FAIL) if f.gate == "grounding"
        ]
        assert len(g_fails) == 4
        assert all("13.8kV" in f.scope for f in g_fails)
        assert {f.scope.split("/")[0] for f in g_fails} == {
            "ISL_ALL", "ISL_F1", "ISL_F2", "ISL_F3",
        }


# ---------------------------------------------------------------------------
# Rung 2: unbalanced solver vs IEEE 4-node, and SLG fault module
# ---------------------------------------------------------------------------


def test_ieee4_both_connections_match_published_solutions():
    """
    Acceptance test for the phase-frame solver: both transformer connections
    must reproduce the published IEEE 4-node unbalanced step-down solutions
    within the rounding of the published values (1 V / 0.1 deg).
    """
    from examples.ieee4_rung2 import REFS, build
    from mgtwin.unbalanced import polar

    for conn in ("GrY-GrY", "D-GrY"):
        v_nodes, _ = build(conn).solve()
        got = {"v3": polar(v_nodes[2]), "v4": polar(v_nodes[3])}
        for node in ("v3", "v4"):
            for (mr, ar), (mc, ac) in zip(REFS[conn][node], got[node]):
                assert abs(mc - mr) < 2.0, (conn, node, mr, mc)
                assert abs(ac - ar) < 0.15, (conn, node, ar, ac)


def test_dgy_thirty_degree_shift_and_zero_sequence_blocking():
    """The DYg low side must lag the YgYg low side by 30 degrees, and the
    delta side must carry no zero-sequence current."""
    import numpy as np

    from examples.ieee4_rung2 import build
    from mgtwin.unbalanced import polar

    # Exactly 30 degrees holds at NO load (pure connection rotation). Under
    # the unbalanced load the per-phase shift drifts a fraction of a degree --
    # the published solutions themselves show 29.4 on phase c -- so testing
    # the loaded case against 30.0 would be testing a false premise.
    f_yy, f_dy = build("GrY-GrY"), build("D-GrY")
    for f in (f_yy, f_dy):
        f.load_kw = (0.0, 0.0, 0.0)
    v_yy, _ = f_yy.solve()
    v_dy, _ = f_dy.solve()
    for ph in range(3):
        a_yy = polar(v_yy[3])[ph][1]
        a_dy = polar(v_dy[3])[ph][1]
        shift = (a_yy - a_dy) % 360.0
        assert abs(shift - 30.0) < 1e-9, (ph, shift)

    # Zero-sequence current on the delta (source) side must vanish
    # (checked on the LOADED case, where I0 on the wye side is large).
    feeder = build("D-GrY")
    _, i_load = feeder.solve()
    tx = feeder.elements[1]
    i_hi = tx.d @ i_load
    i0_hi = abs(np.sum(i_hi)) / 3.0
    i0_lo = abs(np.sum(i_load)) / 3.0
    assert i0_lo > 50.0          # unbalanced wye load has real I0
    assert i0_hi < 1e-6 * i0_lo  # and none of it crosses the delta


def test_ladder_refuses_to_return_unconverged():
    import numpy as np
    import pytest as _pt

    from examples.ieee4_rung2 import Z_Y, LOAD_KW, LOAD_PF
    from mgtwin.unbalanced import LineSegment, RadialFeeder, balanced_source_ln

    f = RadialFeeder(
        v_source_ln=balanced_source_ln(12.47),
        elements=[LineSegment(Z_Y, 2000.0)],
        load_kw=LOAD_KW,
        load_pf=LOAD_PF,
    )
    with _pt.raises(RuntimeError, match="did not converge"):
        f.solve(max_iter=1)


def test_slg_ungrounded_is_exactly_sqrt3():
    import math

    from mgtwin.fault import slg_fault

    res = slg_fault(z1_pu=0.01 + 0.8j, z0_pu=None)
    assert not res.grounded
    assert res.i_fault_pu == 0.0
    assert res.v_max_unfaulted_pu == pytest.approx(math.sqrt(3.0), rel=1e-12)


def test_slg_resistive_grounding_exceeds_sqrt3():
    """The lesson worth pinning: a dominantly resistive Z0 against a reactive
    Z1 does NOT relieve the temporary overvoltage -- the worst unfaulted phase
    exceeds sqrt(3). Low-resistance grounding buys detection, not relief."""
    import math

    from mgtwin.fault import slg_fault

    res = slg_fault(z1_pu=0.013 + 0.77j, z0_pu=3.8 + 0.1j)
    assert res.grounded
    assert abs(res.i_fault_pu) > 0.5
    assert res.v_max_unfaulted_pu > math.sqrt(3.0)


def test_slg_effective_grounding_relieves_overvoltage():
    from mgtwin.fault import slg_fault

    res = slg_fault(z1_pu=0.013 + 0.77j, z0_pu=0.05 + 1.6j)
    assert res.v_max_unfaulted_pu < 1.4
    assert abs(res.i_fault_pu) > 0.8


def test_slg_faulted_phase_is_zero_for_bolted_fault():
    from mgtwin.fault import slg_fault

    res = slg_fault(z1_pu=0.02 + 0.5j, z0_pu=0.05 + 1.0j, zf_pu=0.0)
    assert abs(res.v_abc_pu[0]) < 1e-12


# ---------------------------------------------------------------------------
# Rung 3: the test runner's honesty taxonomy
# ---------------------------------------------------------------------------


@banshee
def _rung3_fixture():
    from examples.banshee_rung3 import DATA, EXAMPLE_LIMITS
    from mgtwin.banshee import compose_banshee
    from mgtwin.testspec import load_suite

    site = compose_banshee(
        DATA / "Matpower/case_banshee_feeder_all_v5.m",
        relay_dir=DATA / "OPAL-RT/relay_settings",
        mdl_path=DATA / "OPAL-RT/BANSHEE.mdl",
    )
    return site, load_suite("ieee2030_8_2018"), EXAMPLE_LIMITS


@banshee
def test_runner_taxonomy_on_banshee_isl_f1():
    from mgtwin.runner import run_suite
    from mgtwin.testrun import CaseStatus, CaseVerdict

    site, suite, limits = _rung3_fixture()
    rep = run_suite(site, suite, "ISL_F1", project_limits=limits)
    by_id = {r.case.id: r for r in rep.results}

    assert by_id["ISL-P-01"].verdict == CaseVerdict.PASS
    assert by_id["ISL-U-01"].verdict == CaseVerdict.FAIL
    assert by_id["SS-01"].verdict == CaseVerdict.PASS
    assert by_id["PROT-01"].status == CaseStatus.BLOCKED
    assert any("grounding" == f.gate for f in by_id["PROT-01"].blocking_findings)
    deferred = {cid for cid, r in by_id.items() if r.status == CaseStatus.DEFERRED}
    assert deferred == {"DISP-01", "DISP-02", "REC-01", "BLK-01", "PQ-01"}

    c = rep.counts()
    assert (c["executed"], c["blocked"], c["deferred"]) == (3, 1, 5)


@banshee
def test_runner_never_invents_limits():
    """Without project limits, every criterion whose spec limit is None must
    stay UNEVALUATED -- including on cases that executed cleanly."""
    from mgtwin.runner import run_suite
    from mgtwin.testrun import CaseStatus, CaseVerdict

    site, suite, _ = _rung3_fixture()
    rep = run_suite(site, suite, "ISL_F1", project_limits=None)
    for r in rep.results:
        if r.status != CaseStatus.EXECUTED:
            continue
        for cr in r.criteria:
            if cr.criterion.limit is None:
                assert not cr.evaluated
        # only the structural zero-limit criteria can be evaluated
        assert r.verdict in (CaseVerdict.INDETERMINATE, CaseVerdict.PASS)


@banshee
def test_prot_unblocks_into_screening_after_remediation():
    from mgtwin.ontology import GroundingKind, GroundingSource
    from mgtwin.runner import run_suite
    from mgtwin.testrun import CaseStatus, CaseVerdict

    site, suite, _ = _rung3_fixture()
    site2 = site.model_copy(deep=True)
    site2.grounding.append(
        GroundingSource(
            id="G_ZZ_F1", bus="B2", kind=GroundingKind.LOW_RESISTANCE,
            z0_ohm=12.0, available_when_islanded=True,
        )
    )
    rep = run_suite(site2, suite, "ISL_F1")
    prot = next(r for r in rep.results if r.case.id == "PROT-01")
    assert prot.status == CaseStatus.EXECUTED
    assert prot.verdict == CaseVerdict.INDETERMINATE  # no limits supplied
    ov = next(m for m in prot.metrics if m.metric == "unfaulted_phase_overvoltage")
    # The rung-2 lesson resurfaces through the runner: LRG exceeds sqrt(3).
    assert ov.value > 1.73


@banshee
def test_assumed_parameter_pass_carries_condition():
    from mgtwin.runner import run_suite
    from mgtwin.testrun import CaseVerdict, ParamProvenance

    site, suite, limits = _rung3_fixture()
    rep = run_suite(site, suite, "ISL_F1", project_limits=limits)
    isl_p = next(r for r in rep.results if r.case.id == "ISL-P-01")
    assert isl_p.verdict == CaseVerdict.PASS
    assert isl_p.provenance == ParamProvenance.ASSUMED
    assert "conditional" in isl_p.render()


def test_runner_rejects_island_without_gfm():
    """An island with only grid-following sources cannot hold frequency; the
    runner must refuse, not simulate nonsense."""
    from mgtwin.runner import _aggregate_source

    site = _minimal_site(
        converters=[
            Converter(
                id="PV", bus="B1", s_rated_mva=1.0,
                control=ConverterControl.GFL, params=ConverterParams(),
            )
        ],
        islands=[
            IslandConfig(id="I1", grid_connected=False, energized_source_ids=["PV"])
        ],
    )
    with pytest.raises(ValueError, match="no grid-forming source"):
        _aggregate_source(site, site.islands[0])


# ---------------------------------------------------------------------------
# Rung 4: adversarial identification evals
# ---------------------------------------------------------------------------


def _scenario(name):
    from mgtwin.truthgen import battery as _battery

    return next(s for s in _battery() if s.name == name)


def test_single_experiment_is_blind_to_amplitude_class_mismatch():
    """
    The rung-4 finding, pinned as a fact about the method: on one step, a
    governor deadband passes whiteness with a clean IDENTIFIED verdict while
    the droop estimate is >5% biased. This is why the multi-amplitude
    protocol exists; if this test ever fails, the finding text in the README
    is stale and should be revisited.
    """
    from mgtwin.evals import run_one

    r = run_one(_scenario("deadband"), seed=0, protocol="single")
    assert r.cell == "false_confidence"


def test_multi_amplitude_catches_deadband_and_saturation():
    from mgtwin.evals import REFUSALS, run_one

    for name in ("deadband", "saturation", "knee_droop"):
        for seed in (0, 1):
            r = run_one(_scenario(name), seed=seed, protocol="multi")
            assert r.verdict in REFUSALS, (name, seed, r.verdict)
            assert r.cell == "true_refusal", (name, seed, r.cell)


def test_multi_amplitude_keeps_nominal_and_minor_probes_identified():
    """Stricter about the right things: nominal, quantized and the 5 ms
    parasitic-lag probes must NOT be refused by the consistency gate."""
    from mgtwin.evals import run_one
    from mgtwin.identify import Verdict

    for name in ("nominal", "quantized", "fast_unmodeled"):
        r = run_one(_scenario(name), seed=0, protocol="multi")
        assert r.verdict == Verdict.IDENTIFIED, (name, r.verdict)
        assert r.cell == "true_confidence", (name, r.cell)


def test_shape_class_mismatch_caught_by_both_protocols():
    from mgtwin.evals import REFUSALS, run_one

    for protocol in ("single", "multi"):
        r = run_one(_scenario("ramp_limit"), seed=0, protocol=protocol)
        assert r.verdict in REFUSALS, (protocol, r.verdict)


def test_identify_multi_pools_consistent_experiments():
    """Two nominal experiments at different amplitudes must pool into a
    single tighter estimate, not be refused."""
    import numpy as np

    from mgtwin.evals import SMALL_STEP, _experiment
    from mgtwin.identify import Verdict, identify_multi

    sc = _scenario("nominal")
    exps = [_experiment(sc, 0, sc.step), _experiment(sc, 1000, SMALL_STEP)]
    res = identify_multi(
        exps,
        theta0={"rp_pu": 0.03, "tp_sec": 0.40},
        bounds={"rp_pu": (0.005, 0.20), "tp_sec": (0.01, 2.0)},
    )
    assert res.verdict == Verdict.IDENTIFIED
    assert res.theta_pooled is not None
    assert res.theta_pooled[0] == pytest.approx(0.05, rel=0.02)
    assert res.theta_pooled[1] == pytest.approx(0.20, rel=0.05)
    assert max(res.spread_rel.values()) < 0.03



def test_declared_it_zone_warns_with_detector_requirement_not_fail():
    """
    v0.6.1 (CAMPAIGN-003 lesson): an ungrounded zone that the design DECLARES
    as an IT system (GroundingSource kind=UNGROUNDED) is a WARN carrying the
    detector requirement -- neither a PASS (it has no zero-sequence source)
    nor a FAIL (it is a recognised design, declared).
    """
    from mgtwin import GroundingKind, GroundingSource, Transformer, WindingConfig

    site = _minimal_site(
        buses=[Bus(id="MV", v_nom_kv=33.0), Bus(id="LV", v_nom_kv=0.69)],
        transformers=[Transformer(id="TX", hv_bus="MV", lv_bus="LV", s_rated_mva=5.5,
            hv_kv=33.0, lv_kv=0.69, x_pu=0.06, hv_config=WindingConfig.DELTA,
            lv_config=WindingConfig.WYE_UNGROUNDED)],
        converters=[Converter(id="PCS", bus="LV", s_rated_mva=5.5,
            control=ConverterControl.GFM_DROOP, params=ConverterParams(p_f_droop_pu=0.04))],
        grounding=[
            GroundingSource(id="GT", bus="MV", kind=GroundingKind.LOW_RESISTANCE, z0_ohm=38.0, available_when_islanded=True),
            GroundingSource(id="IT_LV", bus="LV", kind=GroundingKind.UNGROUNDED, available_when_islanded=True,
                            notes="declared IT, IMD fitted"),
        ],
        islands=[IslandConfig(id="I1", grid_connected=False, energized_source_ids=["PCS"])],
    )
    rep = gate_grounding(site)
    by_scope = {f.scope: f for f in rep.findings if "zone" in f.scope}
    mv = next(f for s, f in by_scope.items() if "33kV" in s)
    lv = next(f for s, f in by_scope.items() if "0.69kV" in s)
    assert mv.severity == Severity.PASS
    assert lv.severity == Severity.WARN and "DECLARED" in lv.message and "detector" in lv.message
    assert not rep.failed


def test_forming_gate_fails_section_of_grid_following_only():
    """v0.6.2 (CAMPAIGN-003 topology study): a sectionalised bus whose only
    sources are grid-following has no reference and must FAIL."""
    from mgtwin.gates import gate_forming

    site = _minimal_site(
        buses=[Bus(id="A", v_nom_kv=33.0), Bus(id="B", v_nom_kv=33.0)],
        switches=[Switch(id="TIE", from_bus="A", to_bus="B")],
        converters=[
            Converter(id="GFM", bus="A", s_rated_mva=5.5, control=ConverterControl.GFM_DROOP,
                      params=ConverterParams(p_f_droop_pu=0.04)),
            Converter(id="GFL", bus="B", s_rated_mva=2.2, control=ConverterControl.GFL, params=ConverterParams()),
        ],
        islands=[
            IslandConfig(id="COUPLED", grid_connected=False, energized_source_ids=["GFM", "GFL"]),
            IslandConfig(id="SPLIT", grid_connected=False, energized_source_ids=["GFM", "GFL"], open_switches=["TIE"]),
        ],
    )
    rep = gate_forming(site)
    coupled = [f.severity for f in rep.findings if f.scope.startswith("COUPLED")]
    split = [f.severity for f in rep.findings if f.scope.startswith("SPLIT")]
    assert coupled == [Severity.PASS]
    assert sorted(v.value for v in split) == ["fail", "pass"]


def test_energisation_gate_fails_whole_section_and_passes_one_at_a_time():
    """Five 3 MVA transformers closed together exceed six 5.5 MVA PCS headroom; one at a time fits."""
    import json
    from mgtwin.gates import gate_energisation, Severity
    site = Site.model_validate_json((Path(__file__).resolve().parent.parent / "examples" / "c003_revC_site.json").read_text())
    rep = gate_energisation(site)
    by = {str(f.scope).split("/")[0]: f.severity for f in rep.findings}
    assert by["FAST_TRANSFER_WHOLE_SECTION"] == Severity.FAIL
    assert by["SLOW_TRANSFER_ONE_GROUP"] in (Severity.PASS, Severity.WARN)


def _revc_example():
    return Site.model_validate_json((Path(__file__).resolve().parent.parent / "examples" / "c003_revC_site.json").read_text())


def test_loading_gate_fails_a_black_start_carrying_the_halls():
    """Rev C's BLACKSTART_GEN leaves the cord-A hall feeders closed on the 3 MVA genset: the grounding, forming and
    energisation gates pass it; the loading gate does not. Its later black-start states, load breakers open, pass."""
    from mgtwin.gates import gate_loading, Severity
    rep = gate_loading(_revc_example())
    worst = {}
    for f in rep.findings:
        isl = str(f.scope).split("/")[0]
        rank = {Severity.PASS: 0, Severity.WARN: 1, Severity.FAIL: 2}
        if rank[f.severity] >= rank[worst.get(isl, Severity.PASS)]:
            worst[isl] = f.severity
    assert worst["BLACKSTART_GEN"] == Severity.FAIL
    assert worst["NORMAL"] == Severity.PASS
    assert worst["BLACKSTART_GEN_FIRST_PCS_TR"] == Severity.PASS
    assert any("exceeds the rating of every source" in f.message for f in rep.findings
               if str(f.scope).startswith("BLACKSTART_GEN/"))


def test_loading_gate_warns_when_storage_carries_the_load():
    """PCS 1-2 forming BESS-1 with Aux A closed and no SOFC running: carried, but on the batteries."""
    from mgtwin.gates import gate_loading, Severity
    rep = gate_loading(_revc_example())
    assert [f.severity for f in rep.findings if str(f.scope).startswith("BLACKSTART_BESS1_SOFC_PAIR/")] == [Severity.WARN]


def test_loading_gate_fails_a_transformer_feeding_more_than_its_rating():
    """A 3.5 MVA transformer feeding a 12.5 MW hall cord with no source behind it (Site v10's TR_IT_A): FAIL, in
    every state that closes the cord's feeder."""
    from mgtwin.gates import gate_loading, Severity
    from mgtwin.ontology import Bus, Load, Transformer
    site = _revc_example()
    hall = next(ld for ld in site.loads if ld.p_mw > 5)
    site.buses.append(Bus(id="HALL_LV", v_nom_kv=0.4))
    site.transformers.append(Transformer(id="TR_HALL", hv_bus=hall.bus, lv_bus="HALL_LV", s_rated_mva=3.5,
                                         hv_kv=33.0, lv_kv=0.4, x_pu=0.08, r_pu=0.006,
                                         hv_config="delta", lv_config="wye_grounded"))
    site.loads = [ld if ld.id != hall.id else Load(**{**ld.model_dump(), "bus": "HALL_LV"}) for ld in site.loads]
    rep = gate_loading(site)
    tr = [f for f in rep.findings if str(f.scope).endswith("/TR_HALL")]
    assert tr and all(f.severity == Severity.FAIL for f in tr)
    assert str(tr[0].scope).split("/")[0] == "NORMAL" or any(str(f.scope).startswith("NORMAL/") for f in tr)


def test_units_gate_fails_transformer_reactance_on_the_system_base():
    """Transformer.x_pu is on the transformer's own rating; a Site that gives it on the 50 MVA system base (x 1.0
    for 6 % on 3 MVA) is caught, and one entered on its own rating is not."""
    from mgtwin.gates import gate_units, Severity
    site = _revc_example()
    fails = [f for f in gate_units(site).findings if f.severity == Severity.FAIL and "transformer reactance" in f.message]
    assert fails and "TR_SOFC01 1" in fails[0].message
    for tx in site.transformers:
        tx.x_pu = 0.06
    assert not [f for f in gate_units(site).findings if "transformer reactance" in f.message]


def test_loading_gate_keeps_normally_open_switches_open():
    """A normally open tie or alternate leg is open unless the state closes it, so it does not join the boards it
    ties; marked normally closed, it does."""
    from mgtwin.gates import _sections
    site = _revc_example()
    apart = [(isl, sw) for isl in site.islands for sw in site.switches if not sw.normally_closed
             and _sections(site, isl)(sw.from_bus) != _sections(site, isl)(sw.to_bus)]
    assert apart
    isl, sw = apart[0]
    sw.normally_closed = True
    find = _sections(site, isl)
    assert find(sw.from_bus) == find(sw.to_bus)


def test_a_state_closes_a_normally_open_leg():
    """Rev C's transfers re-close groups 1-5 onto SOFC-2A through their normally open second legs: listed in the
    state's closed_switches, the legs conduct and the energisation gate weighs the step's inrush. Left out, the
    gate says the step energises nothing - it never passes a dead transformer."""
    from mgtwin.gates import gate_energisation, Severity
    site = _revc_example()
    isl = next(i for i in site.islands if i.id == "FAST_TRANSFER_WHOLE_SECTION")
    assert "CB_SOFC01_2" in isl.closed_switches and "CB_SOFC01_2" not in site.open_switch_ids(isl)
    fast = [f for f in gate_energisation(site).findings if str(f.scope).startswith("FAST_TRANSFER_WHOLE_SECTION/")]
    assert len(fast) == 1 and "5 transformer(s)" in fast[0].message and fast[0].severity == Severity.FAIL
    isl.closed_switches = []
    fast = [f for f in gate_energisation(site).findings if str(f.scope).startswith("FAST_TRANSFER_WHOLE_SECTION/")]
    assert len(fast) == 5 and all("no energised source reaches" in f.message for f in fast)


def test_a_state_cannot_open_and_close_one_switch():
    import pytest
    from mgtwin.ontology import Site
    raw = _revc_example().model_dump(mode="json")
    st = next(i for i in raw["islands"] if i["id"] == "FAST_TRANSFER_WHOLE_SECTION")
    st["open_switches"].append("CB_SOFC01_2")
    with pytest.raises(ValueError, match="both opens and closes"):
        Site.model_validate(raw)
    st["open_switches"].remove("CB_SOFC01_2")
    st["closed_switches"].append("CB_NOPE")
    with pytest.raises(ValueError, match="unknown switch"):
        Site.model_validate(raw)


# ---------------------------------------------------------------------------
# Stage 6: the validity envelope
# ---------------------------------------------------------------------------


def _revc_rebased():
    site = _revc_example()
    for tx in site.transformers:
        tx.x_pu = 0.06
    return site


def test_envelope_a_site_wide_fail_puts_everything_outside():
    from mgtwin.envelope import Inside, build_envelope
    env = build_envelope(_revc_example())
    assert env.site_wide_fails and "transformer reactance" in env.site_wide_fails[0]
    verdict, why = env.check("NORMAL", "loading", load_mva=30.0)
    assert verdict == Inside.OUTSIDE and "site-wide" in why[0]


def test_envelope_states_gates_and_tier():
    """The gates bound the envelope state by state; phenomena beyond the tier and states the Site does not
    enumerate are outside; storage-carried load is provisional; assumed parameters make frequency provisional."""
    from mgtwin.envelope import Inside, build_envelope
    env = build_envelope(_revc_rebased())
    assert not env.site_wide_fails
    assert env.check("NORMAL", "loading", load_mva=37.0)[0] == Inside.INSIDE
    assert env.check("NORMAL", "loading", load_mva=60.0)[0] == Inside.PROVISIONAL   # above 44 MVA sustained
    assert env.check("NORMAL", "loading", load_mva=80.0)[0] == Inside.OUTSIDE       # above 77 MVA in all
    v, why = env.check("BLACKSTART_GEN", "loading")
    assert v == Inside.OUTSIDE and "loading" in why[0]
    assert env.check("FAST_TRANSFER_WHOLE_SECTION", "energisation")[0] == Inside.OUTSIDE
    assert env.check("SLOW_TRANSFER_ONE_GROUP", "energisation")[0] == Inside.PROVISIONAL  # assumed inrush
    assert env.check("NORMAL", "emt")[0] == Inside.OUTSIDE
    assert env.check("NORMAL", "voltage")[0] == Inside.OUTSIDE
    assert env.check("NOT_A_STATE", "loading")[0] == Inside.OUTSIDE
    v, why = env.check("NORMAL", "frequency", step_pu=0.3)
    assert v == Inside.PROVISIONAL and any("assumed" in w for w in why)
    assert any("ranges over which the dynamic models hold" in u for u in env.undeclared)


def test_envelope_identified_amplitudes_bound_a_step():
    """With every forming source identified over 0.1-0.5 pu steps, a 0.3 pu step is inside and a 0.65 pu step is
    an extrapolation; protection stays provisional while ride-through is assumed."""
    import json as _json
    import pytest
    from mgtwin.envelope import IdentificationRecord, Inside, build_envelope
    from mgtwin.ontology import ConverterControl
    site = _revc_rebased()
    gfm = [c.id for c in site.converters if c.control != ConverterControl.GFL] + [m.id for m in site.machines]
    recs = [IdentificationRecord(s, "identified", (0.1, 0.5), {"rp": 0.04}) for s in gfm]
    env = build_envelope(site, recs)
    assert env.check("NORMAL", "frequency", step_pu=0.3)[0] == Inside.INSIDE
    v, why = env.check("NORMAL", "frequency", step_pu=0.65)
    assert v == Inside.PROVISIONAL and "extrapolation" in why[0]
    assert env.check("NORMAL", "protection")[0] == Inside.PROVISIONAL
    recs[0].verdict = "structural_mismatch"
    assert env.check("NORMAL", "frequency", step_pu=0.3)[0] == Inside.PROVISIONAL
    d = _json.loads(build_envelope(site, recs).to_json())
    assert d["states"]["NORMAL"]["rating_mva"] == 77.0 and d["phenomena"]["emt"]["support"] == "excluded"
    assert "# Validity envelope" in env.render()
    with pytest.raises(ValueError, match="unknown sources"):
        build_envelope(site, [IdentificationRecord("NOPE", "identified", (0.1, 0.5))])


# ---------------------------------------------------------------------------
# Stage 1: tag ingest
# ---------------------------------------------------------------------------


def _revc_rules():
    from mgtwin.ingest import TagRule
    return [
        TagRule(r"^(?P<id>PCS\d+)\.(?P<meas>\w+)$", "converter", {"P_MW": "P", "Q_MVAR": "Q", "F": "F", "SOC": "SOC"}),
        TagRule(r"^(?P<id>SOFC\d\d)_(?P<meas>KW|KVAR|HZ)$", "converter", {"KW": "P", "KVAR": "Q", "HZ": "F"}),
        TagRule(r"^(?P<id>CB_\w+)\.POS$", "switch", measurement="POS"),
        TagRule(r"^GEN\.(?P<meas>\w+)$", "machine", {"P": "P", "F": "F"}, id_template="GEN_BS"),
        TagRule(r"^(?P<id>PCS1)\.P_MW$", "converter", measurement="Q"),   # a deliberately conflicting rule
    ]


def test_ingest_maps_scales_and_flags_tags():
    from mgtwin.ingest import Tag, TagStatus, ingest
    tags = [Tag("PCS2.P_MW", "MW", sample_period_s=0.02), Tag("PCS2.F", "Hz", sample_period_s=0.02),
            Tag("SOFC01_KW", "kW", sample_period_s=1.0), Tag("SOFC01_HZ", "Hz", sample_period_s=1.0),
            Tag("SOFC02_KW", "A"),                    # wrong unit for power
            Tag("PCS3.TEMP_C", "degC"),               # code not in the map
            Tag("PCS1.P_MW", "MW"),                   # two rules disagree
            Tag("CB_TIE.POS", ""), Tag("CB_TIE.POS", ""),   # a duplicate
            Tag("HALL7.P", "MW"),                     # no rule
            Tag("GEN.P", "kW", sample_period_s=0.1), Tag("GEN.F", "Hz", sample_period_s=0.1)]
    rep = ingest(tags, _revc_rules())
    st = {r.tag.tag: r.status for r in rep.results}
    assert st["PCS2.P_MW"] == st["SOFC01_KW"] == st["GEN.P"] == TagStatus.MATCHED
    assert st["PCS3.TEMP_C"] == st["HALL7.P"] == TagStatus.UNMATCHED
    assert st["PCS1.P_MW"] == TagStatus.AMBIGUOUS
    assert [r.status for r in rep.results if r.tag.tag == "CB_TIE.POS"] == [TagStatus.MATCHED, TagStatus.DUPLICATE]
    sofc = rep.entities[("converter", "SOFC01")]
    assert sofc.measurements["P"].scale == 1e-3
    bad = next(r for r in rep.results if r.tag.tag == "SOFC02_KW")
    assert any("is current, but P is power" in n for n in bad.notes)
    assert rep.entities[("machine", "GEN_BS")].measurements["P"].scale == 1e-3
    assert "## Flagged tags" in rep.render() and rep.to_dict()["counts"]["ambiguous"] == 1


def test_ingest_against_the_site():
    """Against Rev C: telemetry for equipment the Site lacks, equipment no tag observes, and which sources can be
    identified from operating data at a PMU-grade period."""
    from mgtwin.ingest import Tag, compare_with_site, ingest
    tags = [Tag("PCS2.P_MW", "MW", sample_period_s=0.02), Tag("PCS2.F", "Hz", sample_period_s=0.02),
            Tag("SOFC01_KW", "kW", sample_period_s=1.0), Tag("SOFC01_HZ", "Hz", sample_period_s=1.0),
            Tag("PCS9.P_MW", "MW"), Tag("PCS3.P_MW", "MW")]
    rep = ingest(tags, _revc_rules())
    out = {(f.id, f.message.split(":")[0]) for f in compare_with_site(rep, _revc_example(), identification_period_s=0.05)}
    assert ("PCS9", "telemetry for equipment the Site does not hold") in out
    assert ("PCS1", "in the Site, but no tag observes it") in out
    assert ("PCS3", "cannot be identified from operating data") in out
    assert ("SOFC01", "sampled too slowly to identify (needs <= 0.05 s)") in out
    assert not any(i == "PCS2" for i, _ in out)


def test_ingest_rules_are_checked():
    import pytest
    from mgtwin.ingest import TagRule
    with pytest.raises(ValueError, match="unknown kind"):
        TagRule(r"^(?P<id>X)$", "widget", measurement="P")
    with pytest.raises(ValueError, match="named group 'id'"):
        TagRule(r"^X\.(?P<meas>P)$", "converter", {"P": "P"})
    with pytest.raises(ValueError, match="unknown measurement"):
        TagRule(r"^(?P<id>X)\.(?P<meas>P)$", "converter", {"P": "POWER"})


@banshee
def test_banshee_transformers_paired_by_impedance_on_their_own_rating():
    """The .m's transformer branches pair with the .mdl's instances by impedance, not by order: XTX_3_4 (x 1.96 on
    20 MVA) is a 500 kVA unit, not the 2500 kVA it was given in turn. Every reactance, rebased to its unit's rating,
    is then a transformer's (the units gate passes); the three branches no instance matches are named."""
    from examples.banshee_rung1 import CASE_V5, MDL, RELAYS
    from mgtwin.banshee import compose_banshee, transformer_pairing
    from mgtwin.gates import gate_units, Severity
    pr = transformer_pairing(CASE_V5, MDL)
    assert pr["TX_3_4"]["kva"] == 500.0 and pr["TX_3_4"]["mismatch"] == 0.0
    assert sorted(k for k, v in pr.items() if v["mismatch"] > 0.05) == ["TX_30_40", "TX_50_60", "TX_80_110"]
    site = compose_banshee(CASE_V5, relay_dir=RELAYS, mdl_path=MDL)
    tx = {t.id: t for t in site.transformers}
    assert tx["XTX_3_4"].s_rated_mva == 0.5 and abs(tx["XTX_3_4"].x_pu - 0.049) < 1e-3
    assert not [f for f in gate_units(site).findings if f.severity == Severity.FAIL and "reactance" in f.message]


# ---------------------------------------------------------------------------
# v0.6.8: protection functions, settings groups, and the screening checks
# ---------------------------------------------------------------------------


def _two_bess_site(functions=None, pickup_a=None, groups=("A",), state_group="A"):
    """Two 5 MVA grid-forming BESS on B1 and B2, tied by breaker SW; its relay carries `functions`."""
    from mgtwin.ontology import ProtectionFunction  # noqa: F401
    bess = lambda i, b: Converter(id=i, bus=b, s_rated_mva=5.0, control=ConverterControl.GFM_DROOP,  # noqa: E731
                                  params=ConverterParams(p_f_droop_pu=0.05, i_limit_transient_pu=1.2))
    return _minimal_site(
        buses=[Bus(id="B1", v_nom_kv=11.0), Bus(id="B2", v_nom_kv=11.0)],
        converters=[bess("BESS1", "B1"), bess("BESS2", "B2")],
        switches=[Switch(id="SW", from_bus="B1", to_bus="B2")],
        relays=[OvercurrentRelay(id="R", switch_id="SW", ct_ratio=600.0, pickup_a=pickup_a,
                                 functions=list(functions or []))],
        islands=[IslandConfig(id="I1", grid_connected=False, settings_group=state_group,
                              energized_source_ids=["BESS1", "BESS2"])],
    )


def test_protection_functions_are_validated():
    from mgtwin.ontology import ProtectionFunction
    with pytest.raises(ValueError, match="unknown ANSI"):
        ProtectionFunction(ansi="99X")
    with pytest.raises(ValueError, match="needs a direction"):
        ProtectionFunction(ansi="67N", pickup_a=50.0)
    with pytest.raises(ValueError, match="zone's buses"):
        ProtectionFunction(ansi="87L")
    with pytest.raises(ValueError, match="declares no protection element"):
        OvercurrentRelay(id="R", switch_id="SW", ct_ratio=600.0)
    with pytest.raises(ValueError, match="unknown bus"):
        _two_bess_site([ProtectionFunction(ansi="87L", zone_buses=["B1", "NOPE"])])


def test_a_legacy_relay_reads_as_one_phase_element():
    r = OvercurrentRelay(id="R", switch_id="SW", ct_ratio=600.0, pickup_a=300.0, time_dial=0.3)
    (e,) = r.elements()
    assert e.ansi == "51" and e.pickup_a == 300.0 and e.active_in("BLACKSTART")
    assert OvercurrentRelay(id="R", switch_id="SW", ct_ratio=600.0, pickup_a=300.0, directional=True
                            ).elements()[0].ansi == "67"


def test_protection_gate_fails_a_state_whose_settings_group_has_nothing_active():
    """The black-start state runs group BS; a relay set only for the grid-connected group G leaves the breaker
    unprotected there. Give it a low-set 51 in BS and the state is screened again."""
    from mgtwin.ontology import ProtectionFunction
    grid_only = [ProtectionFunction(ansi="51", pickup_a=1200.0, groups=["G"])]
    rep = gate_protection(_two_bess_site(grid_only, state_group="BS"))
    assert any("no element of relay R is active in settings group 'BS'" in f.message for f in rep.by_severity(Severity.FAIL))
    both = grid_only + [ProtectionFunction(ansi="51", pickup_a=200.0, groups=["BS"])]
    rep = gate_protection(_two_bess_site(both, state_group="BS"))
    assert not rep.failed
    assert [f.scope for f in rep.findings if "pickup available" in f.message] == ["I1/R/51"]


def test_protection_gate_names_magnitude_only_breakers_and_passes_unit_protection():
    from mgtwin.ontology import ProtectionFunction
    rep = gate_protection(_two_bess_site(pickup_a=150.0))
    warn = [f for f in rep.by_severity(Severity.WARN) if "current-limited converters" in f.message]
    assert warn and "rests on magnitude at R" in warn[0].message
    rep = gate_protection(_two_bess_site([ProtectionFunction(ansi="87L", zone_buses=["B1", "B2"])], pickup_a=150.0))
    assert not [f for f in rep.by_severity(Severity.WARN) if "current-limited converters" in f.message]
    assert any("does not rest on magnitude alone" in f.message for f in rep.by_severity(Severity.PASS))


def test_protection_gate_wants_direction_on_a_breaker_fed_from_both_sides():
    """BESS on each side of SW: a non-directional 51 sees through-faults; a 67 or an 87L does not need the WARN."""
    from mgtwin.ontology import ProtectionFunction
    both_sides = lambda rep: [f for f in rep.by_severity(Severity.WARN) if "from both sides" in f.message]  # noqa: E731
    assert both_sides(gate_protection(_two_bess_site(pickup_a=150.0)))
    assert not both_sides(gate_protection(_two_bess_site(
        [ProtectionFunction(ansi="67", pickup_a=150.0, direction="forward")])))
    assert not both_sides(gate_protection(_two_bess_site(
        [ProtectionFunction(ansi="87L", zone_buses=["B1", "B2"])], pickup_a=150.0)))


def test_envelope_takes_protection_studies(tmp_path):
    """A study supersedes the screening for the state it covers: failed -> outside; passed on assumed converter
    models -> provisional, naming the study; on identified models -> inside. States it does not cover keep the
    screening's provisional verdict."""
    import json as _json
    from mgtwin.envelope import Inside, ProtectionStudyRecord, build_envelope, load_protection_studies
    site = _revc_rebased()
    path = tmp_path / "studies.json"
    path.write_text(_json.dumps([
        {"state_id": "NORMAL", "tool": "Electrisim", "passed": True, "reference": "C003 report Rev 6, s. 9"},
        {"state_id": "N1_PCS", "tool": "Electrisim", "passed": True, "models_identified": True},
        {"state_id": "MVA_BUS_FAULT", "tool": "Electrisim", "passed": False}]))
    env = build_envelope(site, protection_studies=load_protection_studies(path))
    v, why = env.check("NORMAL", "protection")
    assert v == Inside.PROVISIONAL and why == ["studied in Electrisim (C003 report Rev 6, s. 9) on assumed converter "
                                               "current limits and ride-through"]
    assert env.check("N1_PCS", "protection") == (Inside.INSIDE, [])
    v, why = env.check("MVA_BUS_FAULT", "protection")
    assert v == Inside.OUTSIDE and "Electrisim protection study" in why[0]
    v, why = env.check("N1_SOFC", "protection")
    assert v == Inside.PROVISIONAL and any("ride-through assumed" in w for w in why)
    assert "## Protection studies" in env.render() and env.to_dict()["protection_studies"]["N1_PCS"]["passed"]
    with pytest.raises(ValueError, match="does not enumerate"):
        build_envelope(site, protection_studies=[ProtectionStudyRecord("NOPE", "Electrisim", True)])
