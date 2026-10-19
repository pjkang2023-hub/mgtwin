"""
Quantify the rung-1 grounding finding: SLG faults on Banshee's islanded
13.8 kV zones under three grounding designs.

The grounding gate established structurally that every islanded MV zone has
no zero-sequence source. This example puts numbers on it for ISL_F1 (feeder 1
islanded on the 4 MVA diesel), where a machine island has a classical
sequence Thevenin: Z1 ~= Xd'' (0.15 pu on 4 MVA) plus the tie line, on the
20 MVA system base.

Three designs are compared, because the first "obvious" remediation teaches
the wrong lesson if presented alone:

  1. As released (ungrounded)      -- sqrt(3) overvoltage, zero fault current
  2. Low-resistance zig-zag (Rn)   -- detectable current, damage limited, but
                                      the temporary overvoltage is NOT fixed:
                                      a resistive Z0 against a reactive Z1
                                      rotates the unfaulted phasors and can
                                      slightly EXCEED sqrt(3)
  3. Effective grounding (X0-dom.) -- overvoltage held (~1.3-1.4 pu, so
                                      arresters and insulation relax), paid
                                      for with the highest fault current

That is the standard grounding design triangle: overvoltage vs fault current
vs arc damage. Pick two.

The converter-only island (ISL_F2) is stated qualitatively: a GFM converter
is not a voltage source behind reactance, so a sequence Thevenin
misrepresents it. The sqrt(3) conclusion for the ungrounded case holds
regardless (it is a property of the missing zero-sequence path, not of the
source); the post-remediation fault current would be converter-limited and
needs the ride-through data the honesty gate keeps flagging.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mgtwin.fault import slg_fault, zigzag_z0_pu  # noqa: E402
from mgtwin.units import Base  # noqa: E402

RULE = "=" * 78

BASE = Base(s_base_mva=20.0, v_base_kv=13.8, f_base_hz=60.0)
V_LN = 13.8e3 / 3**0.5

# Diesel 4 MVA, Xd'' = 0.15 pu own base -> 20 MVA base, plus 2000 ft of
# 13.8 kV feeder (positive-sequence approximation from the case data).
Z1 = 1j * 0.15 * (20.0 / 4.0) + complex(0.0128, 0.0214)


def show(title: str, consequence: str, **kw) -> None:
    print(f"\n{title}")
    res = slg_fault(z1_pu=Z1, **kw)
    print(res.render(i_base_a=BASE.i_base_a, v_base_ln=V_LN))
    print(consequence)


def main() -> int:
    print(RULE)
    print("SLG fault on the islanded 13.8 kV zone (ISL_F1, diesel island)")
    print(RULE)
    print(f"Z1 = Z2 = {Z1.real:.3f} + j{Z1.imag:.3f} pu on 20 MVA / 13.8 kV")

    show(
        "1. As released -- no zero-sequence source in the zone:",
        "   consequence   : phase-ground insulation and arresters sized near\n"
        "                   1.0 pu see 1.73 pu for as long as the fault\n"
        "                   persists -- and it persists, because there is no\n"
        "                   neutral current for a ground relay to measure.",
        z0_pu=None,
    )

    rn_ohm = 12.0
    z0_lrg = 3.0 * zigzag_z0_pu(rn_ohm, 13.8, 20.0) + 1j * 0.10
    show(
        f"2. Low-resistance grounding -- zig-zag with {rn_ohm:.0f}-ohm neutral resistor:",
        "   consequence   : the fault is now DETECTABLE (a 51G element has\n"
        "                   hundreds of amperes to work with) and arc damage is\n"
        "                   limited -- but note the overvoltage: a resistive Z0\n"
        "                   against a reactive Z1 rotates the unfaulted phasors\n"
        "                   and the worst phase slightly EXCEEDS sqrt(3). LRG\n"
        "                   buys detection and damage limitation, not\n"
        "                   overvoltage relief; insulation stays full line-line\n"
        "                   rated. This is why 'add a grounding bank' is not\n"
        "                   the end of the design conversation.",
        z0_pu=z0_lrg,
    )

    z0_eff = 0.05 + 1j * 1.6  # X0/X1 ~= 2.0, R0/X1 << 1: effectively grounded
    show(
        "3. Effective grounding -- low-impedance bank, X0/X1 ~= 2:",
        "   consequence   : overvoltage held to ~1.2 pu (COG well under 0.8;\n"
        "                   reduced-rating arresters are admissible), paid for\n"
        "                   with the largest ground-fault current of the three\n"
        "                   -- which the 4 MVA diesel must survive and the\n"
        "                   switchgear must interrupt.",
        z0_pu=z0_eff,
    )

    print(f"\n{RULE}")
    print(
        "The design triangle in one table: overvoltage vs fault current vs\n"
        "damage -- pick two. The released benchmark sits at the worst corner\n"
        "for an islanded system: full overvoltage AND an undetectable fault.\n"
        "\n"
        "ISL_F2 (PV-only island): conclusion 1 is identical -- sqrt(3) is a\n"
        "property of the missing zero-sequence path, not of the source. But\n"
        "quantifying designs 2 and 3 there requires the converter's current-\n"
        "limited fault model, which the release does not contain. That is the\n"
        "same gap gate_model_honesty flags, now with an engineering\n"
        "consequence attached."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
