"""
End-to-end demonstration.

Part 1  A composed site containing two defects that survive human review:
        the grounding reference sits outside the microgrid, and relay pickup was
        carried over from the grid-connected study. Both are caught structurally.

Part 2  The same site remediated.

Part 3  Identification against known truth, three cases:
          A  well-posed
          B  structurally non-identifiable parameter set
          C  model-form mismatch, where tight intervals would be a lie

Part 3C is the case that matters. Everything else is bookkeeping.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mgtwin import (  # noqa: E402
    Base,
    Branch,
    Bus,
    Converter,
    ConverterControl,
    ConverterParams,
    GroundingKind,
    GroundingSource,
    IslandConfig,
    Load,
    OvercurrentRelay,
    Site,
    Switch,
    SynchronousGen,
    Transformer,
    WindingConfig,
    identify,
    load_step,
    load_suite,
    run_all,
    simulate_gfm,
)

RULE = "=" * 78
SUB = "-" * 78


# ---------------------------------------------------------------------------
# Site construction
# ---------------------------------------------------------------------------


def build_site(remediated: bool = False) -> Site:
    base = Base(s_base_mva=10.0, v_base_kv=11.0, f_base_hz=50.0)

    buses = [
        Bus(id="BUS_UTIL", v_nom_kv=33.0, is_poi=True),
        Bus(id="BUS_MV", v_nom_kv=11.0),
        Bus(id="BUS_LOAD", v_nom_kv=11.0),
    ]

    transformers = [
        Transformer(
            id="TX1",
            hv_bus="BUS_UTIL",
            lv_bus="BUS_MV",
            s_rated_mva=15.0,
            hv_kv=33.0,
            lv_kv=11.0,
            x_pu=0.08,
            r_pu=0.006,
            hv_config=WindingConfig.WYE_GROUNDED,
            # Delta on the microgrid side: no zero-sequence path inside the
            # microgrid. Utterly standard, and the reason the grounding gate
            # exists.
            lv_config=WindingConfig.DELTA,
        )
    ]

    branches = [
        dict(
            id="FDR1",
            from_bus="BUS_MV",
            to_bus="BUS_LOAD",
            r_pu=0.012,
            x_pu=0.028,
            s_base_mva=10.0,
            v_base_kv=11.0,
            rating_mva=12.0,
        )
    ]

    switches = [
        Switch(id="SW_PCC", from_bus="BUS_UTIL", to_bus="BUS_MV", is_pcc=True),
        Switch(id="SW_MV", from_bus="BUS_MV", to_bus="BUS_LOAD"),
    ]

    converters = [
        Converter(
            id="BESS_1",
            bus="BUS_MV",
            s_rated_mva=5.0,
            control=ConverterControl.GFM_DROOP,
            params=ConverterParams(
                p_f_droop_pu=0.05,
                q_v_droop_pu=0.05,
                h_sec=2.0,
                tp_sec=0.20,
                i_limit_transient_pu=1.20,
                i_limit_steady_pu=1.10 if remediated else None,
                ride_through_identified=remediated,
            ),
            vendor="(vendor)",
            firmware="4.2.1" if remediated else "",
            energy_mwh=10.0,
        ),
        Converter(
            id="PV_1",
            bus="BUS_LOAD",
            s_rated_mva=3.0,
            control=ConverterControl.GFL,
            params=ConverterParams(
                tp_sec=0.05,
                i_limit_transient_pu=1.20,
                i_limit_steady_pu=1.10 if remediated else None,
                ride_through_identified=remediated,
            ),
            firmware="2.8.0" if remediated else "",
        ),
    ]

    machines = [
        SynchronousGen(
            id="DG_1",
            bus="BUS_MV",
            s_rated_mva=2.0,
            h_sec=1.2,
            xd_subtransient_pu=0.18,
            droop_pu=0.04,
            neutral_grounded=remediated,
        )
    ]

    loads = [
        Load(id="LD_CRIT", bus="BUS_LOAD", p_mw=2.5, q_mvar=0.8, critical=True),
        Load(id="LD_GEN", bus="BUS_LOAD", p_mw=3.5, q_mvar=1.1),
    ]

    grounding = [
        GroundingSource(
            id="G_UTIL",
            bus="BUS_UTIL",
            kind=GroundingKind.SOLID,
            available_when_islanded=False,
            notes="Utility substation neutral, upstream of the PCC.",
        )
    ]
    if remediated:
        grounding.append(
            GroundingSource(
                id="G_ZIGZAG",
                bus="BUS_MV",
                kind=GroundingKind.LOW_RESISTANCE,
                z0_ohm=12.0,
                available_when_islanded=True,
                notes="Zig-zag grounding transformer added inside the microgrid.",
            )
        )

    # Pickup originally set from grid-connected fault duty, where the utility
    # contributes several thousand amperes. Remediation adds an islanded
    # settings group.
    relays = [
        OvercurrentRelay(
            id="R_MV",
            switch_id="SW_MV",
            ct_ratio=600.0,
            pickup_a=300.0 if remediated else 600.0,
            time_dial=0.30,
            notes="islanded settings group" if remediated else "grid-connected settings",
        )
    ]

    islands = [
        IslandConfig(
            id="ISL_0",
            description="Grid-connected, normal",
            grid_connected=True,
            energized_source_ids=["BESS_1", "PV_1", "DG_1"],
        ),
        IslandConfig(
            id="ISL_1",
            description="Islanded, all local sources available",
            open_switches=["SW_PCC"],
            grid_connected=False,
            energized_source_ids=["BESS_1", "PV_1", "DG_1"],
        ),
        IslandConfig(
            id="ISL_2",
            description="Islanded, genset on maintenance outage (converters only)",
            open_switches=["SW_PCC"],
            grid_connected=False,
            energized_source_ids=["BESS_1", "PV_1"],
        ),
    ]

    return Site(
        id="DEMO_MG",
        name="Demonstration microgrid",
        base=base,
        buses=buses,
        branches=[Branch(**b) for b in branches],
        transformers=transformers,
        switches=switches,
        relays=relays,
        converters=converters,
        machines=machines,
        loads=loads,
        grounding=grounding,
        islands=islands,
    )


# ---------------------------------------------------------------------------
# Part 3: identification against known truth
# ---------------------------------------------------------------------------

RP_TRUE, TP_TRUE, H_TRUE = 0.05, 0.20, 2.0

#: Two instrumentation classes, deliberately contrasted.
PMU = dict(dt_meas=0.02, t_end=8.0, sigma=2.0e-4)  # ~0.01 Hz resolution at 50 Hz
HISTORIAN = dict(dt_meas=1.0, t_end=20.0, sigma=1.0e-3)  # ~0.05 Hz, 1 s scan


def make_measurement(
    dt_meas: float,
    t_end: float,
    sigma: float,
    ramp_limit: float | None = None,
    seed: int = 7,
):
    t, p = load_step(t_end=t_end, dt=0.002)
    truth = simulate_gfm(t, p, RP_TRUE, TP_TRUE, H_TRUE, ramp_limit_pu_s=ramp_limit)
    idx = np.arange(0, t.size, int(round(dt_meas / 0.002)))
    rng = np.random.default_rng(seed)
    return t, p, idx, truth[idx, 0] + rng.normal(0.0, sigma, idx.size)


def predictor(t, p, idx, fixed: dict[str, float]):
    def predict(theta: dict[str, float]) -> np.ndarray:
        return simulate_gfm(
            t,
            p,
            theta.get("rp_pu", fixed.get("rp_pu")),
            theta.get("tp_sec", fixed.get("tp_sec")),
            theta.get("h_sec", fixed.get("h_sec")),
        )[idx, 0]

    return predict


def report_case(label: str, blurb: str, result, truth: dict[str, float], n: int) -> None:
    print(f"\n{SUB}\n{label}\n{blurb}\n")
    print(result.render())
    print(f"  n = {n} samples")
    print("  truth: " + ", ".join(f"{k}={v:g}" for k, v in truth.items()))
    for name, est in zip(result.names, result.theta):
        if truth.get(name):
            err = 100.0 * (est - truth[name]) / truth[name]
            print(f"    {name:<10} {err:+8.1f}% from truth")


def run_identification() -> dict[str, object]:
    results: dict[str, object] = {}

    # -- A: PMU-rate, inertia taken from datasheet -------------------------
    t, p, idx, y = make_measurement(**PMU)
    res = identify(
        predictor(t, p, idx, fixed={"h_sec": H_TRUE}),
        y,
        theta0={"rp_pu": 0.03, "tp_sec": 0.40},
        bounds={"rp_pu": (0.005, 0.20), "tp_sec": (0.01, 2.0)},
    )
    report_case(
        "CASE A  PMU-rate data, two parameters, correct model form",
        "Virtual inertia fixed at its datasheet value. Baseline for the rest.",
        res,
        {"rp_pu": RP_TRUE, "tp_sec": TP_TRUE},
        idx.size,
    )
    results["A"] = res

    # -- B: PMU-rate, inertia also estimated -------------------------------
    res = identify(
        predictor(t, p, idx, fixed={}),
        y,
        theta0={"rp_pu": 0.03, "tp_sec": 0.40, "h_sec": 1.0},
        bounds={"rp_pu": (0.005, 0.20), "tp_sec": (0.01, 2.0), "h_sec": (0.05, 20.0)},
    )
    report_case(
        "CASE B  same data, virtual inertia estimated as well",
        "The transfer function is omega/P = -rp / [(1 + tp*s)(1 + 2*H*rp*s)], so the\n"
        "two lags enter symmetrically and at 2*H*rp = tp = 0.2 s they are nearly\n"
        "interchangeable. The least-determined direction below is that trade, and it\n"
        "shows up with the right shape: tp and h_sec with opposite signs. At PMU rate\n"
        "there is still enough information to separate them, so the honest verdict is\n"
        "IDENTIFIED with the trade-off named -- not a refusal.",
        res,
        {"rp_pu": RP_TRUE, "tp_sec": TP_TRUE, "h_sec": H_TRUE},
        idx.size,
    )
    results["B"] = res

    # -- C: same plant, historian-grade data -------------------------------
    t, p, idx, y = make_measurement(**HISTORIAN)
    res = identify(
        predictor(t, p, idx, fixed={}),
        y,
        theta0={"rp_pu": 0.03, "tp_sec": 0.40, "h_sec": 1.0},
        bounds={"rp_pu": (0.005, 0.20), "tp_sec": (0.01, 2.0), "h_sec": (0.05, 20.0)},
    )
    report_case(
        "CASE C  identical plant and model, historian-grade measurement",
        "1 s scan rate, 0.05 Hz resolution -- i.e. what actually comes out of a SCADA\n"
        "historian. The 0.2 s transient falls between samples. Droop is still\n"
        "recoverable because it is a steady-state gain; the dynamics are not\n"
        "recoverable at all, and the optimiser happily drives tp onto its bound.\n"
        "The verdict is a property of the EXPERIMENT, not of the model. This is the\n"
        "gate doing the one job that matters: returning an instrumentation\n"
        "requirement instead of a number.",
        res,
        {"rp_pu": RP_TRUE, "tp_sec": TP_TRUE, "h_sec": H_TRUE},
        idx.size,
    )
    results["C"] = res

    # -- D: model-form mismatch --------------------------------------------
    t, p, idx, y = make_measurement(**PMU, ramp_limit=0.30, seed=11)
    res = identify(
        predictor(t, p, idx, fixed={"h_sec": H_TRUE}),
        y,
        theta0={"rp_pu": 0.03, "tp_sec": 0.40},
        bounds={"rp_pu": (0.005, 0.20), "tp_sec": (0.01, 2.0)},
    )
    report_case(
        "CASE D  truth has a ramp-rate limit the library model does not contain",
        "A real BESS ramp limit of 0.30 pu/s against a 0.30 pu step. The fitted model\n"
        "is the same well-conditioned two-parameter form as case A, so the Fisher\n"
        "intervals come back tight -- and they are worthless. Only the residual\n"
        "structure reveals it, which is why whiteness is tested first and overrides\n"
        "the intervals. An identifier that reported 1.5% on tp here would have\n"
        "converted an unknown into a false certainty.",
        res,
        {"rp_pu": RP_TRUE, "tp_sec": TP_TRUE},
        idx.size,
    )
    results["D"] = res

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    print(RULE)
    print("PART 1  Composed site, as an engineer would first hand it over")
    print(RULE)
    site = build_site(remediated=False)
    rep = run_all(site)
    print(rep.render())
    print(f"\ngate result: {'BLOCKED' if rep.failed else 'passed'}")

    print(f"\n{RULE}")
    print("PART 2  Same site after remediation")
    print(RULE)
    fixed_site = build_site(remediated=True)
    rep2 = run_all(fixed_site)
    print(rep2.render())
    print(f"\ngate result: {'BLOCKED' if rep2.failed else 'passed'}")
    print(
        "\nThe converter-dominance warning survives remediation, correctly: the\n"
        "arithmetic passes but magnitude-based grading is still the wrong scheme."
    )

    print(f"\n{RULE}")
    print("PART 3  Identification against known truth")
    print(RULE)
    ident = run_identification()

    print(f"\n{RULE}")
    print("PART 4  Test suite carried as versioned data, not code")
    print(RULE)
    suite = load_suite("ieee2030_8_2018")
    print(
        f"edition {suite.edition}: {len(suite.cases)} cases, "
        f"{len(suite.emt_cases())} of which cannot be answered at RMS timescale"
    )
    for c in suite.cases:
        emt = " [EMT]" if c.requires_emt else ""
        print(f"  {c.id:<10} {c.function.value:<22}{emt}")
    unset = sum(1 for c in suite.cases for cr in c.criteria if cr.limit is None)
    print(
        f"\n{unset} criteria carry no numeric limit. They are project or "
        f"interconnection-agreement\nvalues; inventing them would be the most "
        f"dangerous thing this harness could do."
    )

    print(f"\n{RULE}")
    print("SUMMARY")
    print(RULE)
    print(f"  gates, as handed over   : {'BLOCKED' if rep.failed else 'passed'}")
    print(f"  gates, after remediation: {'BLOCKED' if rep2.failed else 'passed'}")
    for k in ("A", "B", "C", "D"):
        print(f"  identification {k}        : {ident[k].verdict.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
