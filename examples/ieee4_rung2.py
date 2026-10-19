"""
Eval rung 2: verify the unbalanced solver against the IEEE 4-node test feeder.

The 4-node feeder exists for exactly one purpose: verifying transformer
connection models in unbalanced power flow. Two connections are exercised
here, chosen for what the microgrid work needs:

  * grounded-wye / grounded-wye -- the zero-sequence-continuous case
  * delta / grounded-wye       -- Banshee's connection, zero-seq blocking

Published references are the IEEE test feeder solutions (unbalanced load,
step-down), as reproduced in the LBNL Modelica Buildings library validation
suite (V2/V3/V4 magnitudes in volts, angles in degrees; magnitudes are
published rounded to 1 V and angles to 0.1 deg, which sets the achievable
agreement floor).

Feeder: 12.47 kV infinite bus -- 2000 ft line -- 6 MVA 12.47/4.16 kV
transformer (R=1%, X=6%) -- 2500 ft line -- unbalanced wye load
(A: 1275 kW @ 0.85, B: 1800 kW @ 0.90, C: 2375 kW @ 0.95, lagging).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mgtwin.unbalanced import (  # noqa: E402
    LineSegment,
    RadialFeeder,
    TransformerDGY,
    TransformerGYGY,
    balanced_source_ln,
    polar,
)

RULE = "=" * 78

# 4-wire grounded-wye phase impedance matrix, ohms/mile (IEEE 4-node doc).
Z_Y = np.array(
    [
        [0.4576 + 1.0780j, 0.1559 + 0.5017j, 0.1535 + 0.3849j],
        [0.1559 + 0.5017j, 0.4666 + 1.0482j, 0.1580 + 0.4236j],
        [0.1535 + 0.3849j, 0.1580 + 0.4236j, 0.4615 + 1.0651j],
    ]
)

LOAD_KW = (1275.0, 1800.0, 2375.0)
LOAD_PF = (0.85, 0.90, 0.95)

#: Published IEEE solutions, unbalanced step-down (LN volts / degrees except
#: node 2 of the DGY case, which the doc publishes line-to-line on the delta).
REFS = {
    "GrY-GrY": dict(
        v3=[(2305, -2.3), (2255, -123.6), (2203, 114.8)],
        v4=[(2175, -4.1), (1930, -126.8), (1833, 102.8)],
    ),
    "D-GrY": dict(
        v3=[(2290, -32.4), (2261, -153.8), (2214, 85.2)],
        v4=[(2157, -34.2), (1936, -157.0), (1849, 73.4)],
    ),
}

#: Published rounding is 1 V / 0.1 deg; allow 2 V and 0.15 deg.
TOL_V = 2.0
TOL_DEG = 0.15


def build(connection: str) -> RadialFeeder:
    tx_cls = {"GrY-GrY": TransformerGYGY, "D-GrY": TransformerDGY}[connection]
    return RadialFeeder(
        v_source_ln=balanced_source_ln(12.47),
        elements=[
            LineSegment(Z_Y, 2000.0),
            tx_cls(kvll_hi=12.47, kvll_lo=4.16, s_mva=6.0, r_pct=1.0, x_pct=6.0),
            LineSegment(Z_Y, 2500.0),
        ],
        load_kw=LOAD_KW,
        load_pf=LOAD_PF,
    )


def check_case(connection: str) -> bool:
    feeder = build(connection)
    v_nodes, i_load = feeder.solve()
    got = {"v3": polar(v_nodes[2]), "v4": polar(v_nodes[3])}
    ref = REFS[connection]

    print(f"\n{connection}  (transformer low-side and load-node LN voltages)")
    print(f"  {'node':<5}{'ph':<4}{'published':>16}{'computed':>18}"
          f"{'d|V|':>8}{'d_ang':>8}")
    ok = True
    for node in ("v3", "v4"):
        for ph, ((mr, ar), (mc, ac)) in enumerate(zip(ref[node], got[node])):
            dm, da = mc - mr, ac - ar
            flag = ""
            if abs(dm) > TOL_V or abs(da) > TOL_DEG:
                ok, flag = False, "  <-- OUT OF TOLERANCE"
            print(
                f"  {node:<5}{'abc'[ph]:<4}{mr:>9.0f} /_{ar:>6.1f}"
                f"{mc:>11.1f} /_{ac:>6.2f}{dm:>8.2f}{da:>8.3f}{flag}"
            )
    print(f"  load currents: " + "  ".join(
        f"{m:.1f} A /_{a:.1f}" for m, a in polar(i_load)))
    return ok


def main() -> int:
    print(RULE)
    print("RUNG 2  Unbalanced solver vs the IEEE 4-node test feeder")
    print(RULE)
    print(
        "Published magnitudes are rounded to 1 V and angles to 0.1 deg, so\n"
        f"agreement within {TOL_V:.0f} V / {TOL_DEG} deg is the achievable floor."
    )

    results = {c: check_case(c) for c in ("GrY-GrY", "D-GrY")}

    print(f"\n{RULE}")
    all_ok = all(results.values())
    for c, ok in results.items():
        print(f"  {c:<10} {'PASS' if ok else 'FAIL'}")
    print(
        "\nWhat this buys: the transformer connection algebra (including the\n"
        "30-degree delta/grounded-wye shift and its zero-sequence blocking)\n"
        "is now verified against the case the IEEE built for that purpose --\n"
        "so when the same matrices quantify SLG overvoltages on Banshee's\n"
        "islands, the machinery has a pedigree rather than a promise."
        if all_ok
        else "\nDo not build on this until the deviations are explained."
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
