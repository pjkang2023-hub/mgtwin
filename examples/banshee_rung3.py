"""
Eval rung 3: run the controller test suite against Banshee's diesel island.

Demonstrates the runner's full honesty taxonomy on one island:

  EXECUTED / PASS           -- planned islanding, steady state (with the
                               conditional-on-assumed-parameters note)
  EXECUTED / FAIL           -- unplanned islanding at full import: droop-only
                               4 MVA diesel takes an 0.86 pu step; no load
                               shed exists in the model. The number is the
                               engineering to-do list.
  EXECUTED / INDETERMINATE  -- protection screening after grounding is
                               remediated: measured values reported, criteria
                               left unevaluated because no project limits
                               exist for them, EMT confirmation noted.
  BLOCKED                   -- protection before remediation: the grounding
                               gate's finding IS the answer, attached.
  DEFERRED                  -- dispatch (no EMS), reconnection (no
                               synchroniser), black start and PQ (EMT tier).

The project limits below are EXAMPLES, clearly labelled: they stand in for
the interconnection agreement. The runner never invents one -- criteria
without a supplied limit stay UNEVALUATED with the measured value shown.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mgtwin.banshee import compose_banshee  # noqa: E402
from mgtwin.ontology import GroundingKind, GroundingSource  # noqa: E402
from mgtwin.runner import run_suite  # noqa: E402
from mgtwin.testspec import load_suite  # noqa: E402

DATA = Path(
    os.environ.get(
        "BANSHEE_DATA",
        "/home/claude/banshee_raw/Supporting Data for Banshee Benchmark Paper/"
        "00 models and data for system verification",
    )
)
RULE = "=" * 78

#: EXAMPLE project limits -- surrogate for the interconnection agreement.
EXAMPLE_LIMITS = {
    ("ISL-P-01", "frequency_nadir_deviation"): 0.5,
    ("ISL-P-01", "critical_load_interrupted"): 0.0,
    ("ISL-U-01", "frequency_nadir_deviation"): 2.0,
    ("ISL-U-01", "rocof_peak"): 3.0,
    ("ISL-U-01", "critical_load_interrupted"): 0.0,
    ("SS-01", "steady_state_frequency_band"): 2.5,
}


def main() -> int:
    site = compose_banshee(
        DATA / "Matpower/case_banshee_feeder_all_v5.m",
        relay_dir=DATA / "OPAL-RT/relay_settings",
        mdl_path=(DATA / "OPAL-RT/BANSHEE.mdl")
        if (DATA / "OPAL-RT/BANSHEE.mdl").exists()
        else None,
    )
    suite = load_suite("ieee2030_8_2018")

    print(RULE)
    print("RUNG 3a  Suite vs ISL_F1 as released")
    print(RULE)
    print("(project limits are EXAMPLES standing in for the interconnection "
          "agreement)\n")
    rep = run_suite(site, suite, "ISL_F1", project_limits=EXAMPLE_LIMITS)
    print(rep.render())

    print(f"\n{RULE}")
    print("RUNG 3b  Same island after grounding remediation (12-ohm zig-zag)")
    print(RULE)
    site2 = site.model_copy(deep=True)
    site2.grounding.append(
        GroundingSource(
            id="G_ZZ_F1",
            bus="B2",
            kind=GroundingKind.LOW_RESISTANCE,
            z0_ohm=12.0,
            available_when_islanded=True,
            notes="remediation from the rung-2 SLG study",
        )
    )
    rep2 = run_suite(site2, suite, "ISL_F1", project_limits=EXAMPLE_LIMITS)
    prot = next(r for r in rep2.results if r.case.id == "PROT-01")
    print(prot.render())

    print(f"\n{RULE}")
    print("READING THE RESULT")
    print(RULE)
    print(
        "The report is the deliverable, and its shape is the point:\n"
        "  * The unplanned-islanding FAIL is real engineering output -- a\n"
        "    droop-only diesel cannot absorb a 0.86 pu step inside a 2 Hz\n"
        "    nadir. The fix is a load-shed scheme or storage, and the harness\n"
        "    will re-judge the same case when the model contains one.\n"
        "  * PROT-01 is BLOCKED as released and becomes an INDETERMINATE\n"
        "    screening after the zig-zag is added -- with the 1.88 pu\n"
        "    overvoltage from rung 2 reappearing in the metrics, because the\n"
        "    machinery is the same. Note it never becomes PASS here: the\n"
        "    limits belong to the project, and the case demands EMT.\n"
        "  * Every PASS on assumed parameters carries its condition. The\n"
        "    identification pipeline (rung 4) is what discharges it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
