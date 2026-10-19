"""
Single-line-to-ground fault analysis by symmetrical components.

This is the quantitative face of gate_grounding: where the gate says "this
zone has no zero-sequence source", this module says what that costs --
unfaulted-phase overvoltage approaching sqrt(3) pu and a ground-fault current
approaching zero (so nothing current-based can detect the fault).

Sequence networks for an SLG fault on phase a through fault impedance Zf:

    I1 = I2 = I0 = Vf / (Z1 + Z2 + Z0 + 3 Zf)
    V1 = Vf - Z1 I1      V2 = -Z2 I2      V0 = -Z0 I0
    Va = V0 + V1 + V2
    Vb = V0 + a^2 V1 + a V2
    Vc = V0 + a  V1 + a^2 V2

The ungrounded case (Z0 -> infinity) is handled as an exact limit, not a big
number: I -> 0, V0 -> -Vf, so Vb = Vf (a^2 - 1) and Vc = Vf (a - 1), both of
magnitude sqrt(3) Vf. All quantities per-unit on a common base.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_A = np.exp(2j * np.pi / 3)


@dataclass(frozen=True)
class SLGResult:
    i_fault_pu: complex        # phase-a fault current
    v_abc_pu: tuple            # phase voltages at the fault bus
    v_max_unfaulted_pu: float  # the number the insulation cares about
    grounded: bool

    def render(self, i_base_a: float | None = None, v_base_ln: float | None = None) -> str:
        mags = [abs(v) for v in self.v_abc_pu]
        lines = [
            f"  fault current  : "
            + (
                f"{abs(self.i_fault_pu):.3f} pu"
                + (f" = {abs(self.i_fault_pu) * i_base_a:,.0f} A" if i_base_a else "")
                if self.grounded
                else "~0 (capacitive charging only -- below any overcurrent element)"
            ),
            f"  phase voltages : "
            + "  ".join(f"{'abc'[k]}={m:.3f} pu" for k, m in enumerate(mags)),
            f"  max unfaulted  : {self.v_max_unfaulted_pu:.3f} pu"
            + (
                f" = {self.v_max_unfaulted_pu * v_base_ln / 1e3:.2f} kV LN"
                if v_base_ln
                else ""
            ),
        ]
        return "\n".join(lines)


def slg_fault(
    z1_pu: complex,
    z2_pu: complex | None = None,
    z0_pu: complex | None = None,
    zf_pu: complex = 0.0,
    v_prefault_pu: complex = 1.0,
) -> SLGResult:
    """
    z0_pu=None means NO zero-sequence path (ungrounded zone): exact limit.
    z2_pu defaults to z1_pu. All impedances Thevenin at the fault bus.
    """
    z1 = complex(z1_pu)
    z2 = complex(z2_pu) if z2_pu is not None else z1
    vf = complex(v_prefault_pu)

    if z0_pu is None:
        # Z0 -> infinity limit.
        vb = vf * (_A**2 - 1.0)
        vc = vf * (_A - 1.0)
        return SLGResult(
            i_fault_pu=0.0,
            v_abc_pu=(0.0 + 0.0j, vb, vc),
            v_max_unfaulted_pu=float(max(abs(vb), abs(vc))),
            grounded=False,
        )

    z0 = complex(z0_pu)
    i_seq = vf / (z1 + z2 + z0 + 3.0 * complex(zf_pu))
    v1 = vf - z1 * i_seq
    v2 = -z2 * i_seq
    v0 = -z0 * i_seq
    va = v0 + v1 + v2
    vb = v0 + _A**2 * v1 + _A * v2
    vc = v0 + _A * v1 + _A**2 * v2
    return SLGResult(
        i_fault_pu=3.0 * i_seq,
        v_abc_pu=(va, vb, vc),
        v_max_unfaulted_pu=float(max(abs(vb), abs(vc))),
        grounded=True,
    )


def zigzag_z0_pu(z0_ohm_per_phase: float, v_kv_ll: float, s_base_mva: float) -> float:
    """Zero-sequence impedance of a zig-zag grounding bank, in pu."""
    return z0_ohm_per_phase / (v_kv_ll**2 / s_base_mva)
