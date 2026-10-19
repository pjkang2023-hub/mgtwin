"""
Per-unit base system and sign conventions.

Rationale
---------
Most defects in composed power-system models are not wrong equations. They are
(a) a factor of 1000 / sqrt(3), or (b) a flipped sign convention. Both are
mechanically checkable, so they are checked here rather than left to review.

Every quantity that enters the ontology carries an explicit unit string. Nothing
is stored "in pu" without also storing the base it is referred to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Sign conventions
# ---------------------------------------------------------------------------


class PowerSign(str, Enum):
    """
    Which direction is positive real/reactive power at a terminal.

    GENERATOR: positive = injection into the network (machines, converters).
    LOAD:      positive = withdrawal from the network (loads).

    Declared per-component and checked at assembly. A BESS declared LOAD-positive
    while its droop gain assumes GENERATOR-positive produces a model that is
    unstable for reasons no one can find.
    """

    GENERATOR = "generator"
    LOAD = "load"


class AngleRef(str, Enum):
    """Angle reference for phasor quantities."""

    COSINE = "cosine"  # v = V*cos(wt + th)
    SINE = "sine"  # v = V*sin(wt + th)


# ---------------------------------------------------------------------------
# Base system
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Base:
    """
    Three-phase per-unit base at one voltage level.

    s_base_mva : three-phase apparent power base
    v_base_kv  : line-to-line RMS voltage base
    f_base_hz  : nominal frequency
    """

    s_base_mva: float
    v_base_kv: float
    f_base_hz: float = 50.0

    def __post_init__(self) -> None:
        if self.s_base_mva <= 0:
            raise ValueError("s_base_mva must be positive")
        if self.v_base_kv <= 0:
            raise ValueError("v_base_kv must be positive")
        if self.f_base_hz not in (50.0, 60.0):
            raise ValueError("f_base_hz must be 50 or 60")

    @property
    def z_base_ohm(self) -> float:
        """Z_base = V_LL^2 / S_3ph."""
        return (self.v_base_kv**2) / self.s_base_mva

    @property
    def i_base_a(self) -> float:
        """I_base = S_3ph / (sqrt(3) * V_LL), in amperes."""
        return (self.s_base_mva * 1e6) / (math.sqrt(3.0) * self.v_base_kv * 1e3)

    def amps(self, i_pu: float) -> float:
        return i_pu * self.i_base_a

    def pu_current(self, i_amp: float) -> float:
        return i_amp / self.i_base_a

    def ohms(self, z_pu: float) -> float:
        return z_pu * self.z_base_ohm

    def pu_impedance(self, z_ohm: float) -> float:
        return z_ohm / self.z_base_ohm

    def rebase_impedance(self, z_pu: float, old: "Base") -> float:
        """Convert a pu impedance from another base to this one."""
        return z_pu * (old.z_base_ohm / self.z_base_ohm)


# ---------------------------------------------------------------------------
# Plausibility ranges, used by the unit-sanity gate
# ---------------------------------------------------------------------------

#: (low, high) physically plausible ranges for distribution-class microgrids.
#: Deliberately wide -- these catch order-of-magnitude errors, not tuning errors.
PLAUSIBLE = {
    "v_base_kv": (0.2, 138.0),
    "s_base_mva": (0.01, 500.0),
    "x_over_r": (0.1, 50.0),
    "z_pu": (1e-5, 5.0),
    "droop_pu": (0.005, 0.20),  # 0.5%-20% P-f droop
    "h_sec": (0.0, 20.0),  # virtual/real inertia constant
    "tp_sec": (1e-3, 5.0),  # power measurement filter
    "i_limit_pu": (0.8, 3.0),  # converter current limit on own rating
}


def check_plausible(name: str, value: float) -> str | None:
    """Return a message if `value` falls outside its plausible band, else None."""
    band = PLAUSIBLE.get(name)
    if band is None:
        return None
    lo, hi = band
    if not (lo <= value <= hi):
        return (
            f"{name}={value:g} is outside the plausible range [{lo:g}, {hi:g}]. "
            f"This usually indicates a unit error rather than an unusual design."
        )
    return None
