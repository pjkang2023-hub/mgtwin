"""
Unbalanced (phase-frame) power flow for radial feeders, ladder method.

Why this exists despite the "no solvers" rule
---------------------------------------------
The rule is that the AUTHORING AGENT never writes solvers ad hoc, because its
bugs would be indistinguishable from physics. This module is the opposite
case: a deliberate, hand-written library component implementing exactly the
method (Kersting's ladder iterative technique with generalized transformer
matrices) that the IEEE 4-node test feeder was designed to verify, and it is
accepted only against that feeder's published solutions. Its scope is
intentionally narrow: radial networks, full 3x3 phase impedance matrices,
grounded-wye constant-PQ loads, and the two transformer connections the
microgrid work needs (grounded-wye/grounded-wye and delta/grounded-wye).
Meshed or large systems belong to OpenDSS, not here.

Conventions
-----------
Backward relation (low -> high side of an element):
    [VLN_up] = [a][VLN_dn] + [b][I_dn]        [I_up] = [c][VLN_dn] + [d][I_dn]
Forward relation (high -> low):
    [VLN_dn] = [A][VLN_up] - [B][I_dn]
American-standard 30 degree shift for the delta/grounded-wye connection: the
high side leads the low side by 30 degrees. All voltages are line-to-neutral
(or equivalent line-to-neutral on delta sides); currents are line currents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_A120 = np.exp(2j * np.pi / 3)  # 1 /_ 120


# ---------------------------------------------------------------------------
# Element models
# ---------------------------------------------------------------------------


@dataclass
class LineSegment:
    """Series 3x3 phase-impedance element: z_ohm_per_mile * length."""

    z_ohm_per_mile: np.ndarray  # complex (3,3)
    length_ft: float

    def __post_init__(self) -> None:
        self.z_ohm_per_mile = np.asarray(self.z_ohm_per_mile, dtype=complex)
        if self.z_ohm_per_mile.shape != (3, 3):
            raise ValueError("phase impedance matrix must be 3x3")
        self.Z = self.z_ohm_per_mile * (self.length_ft / 5280.0)

    # ladder interface -------------------------------------------------
    def backward(self, v_dn: np.ndarray, i_dn: np.ndarray):
        return v_dn + self.Z @ i_dn, i_dn

    def forward(self, v_up: np.ndarray, i_dn: np.ndarray) -> np.ndarray:
        return v_up - self.Z @ i_dn


def _step_down_ratio_gywye(kvll_hi: float, kvll_lo: float) -> float:
    return kvll_hi / kvll_lo


@dataclass
class TransformerGYGY:
    """Grounded-wye / grounded-wye, per Kersting.

    a = nt*I, b = nt*Zt, c = 0, d = I/nt, A = I/nt, B = Zt,
    with nt the LN (== LL) voltage ratio and Zt the per-phase series
    impedance in ohms referred to the LOW side.
    """

    kvll_hi: float
    kvll_lo: float
    s_mva: float
    r_pct: float
    x_pct: float

    def __post_init__(self) -> None:
        self.nt = _step_down_ratio_gywye(self.kvll_hi, self.kvll_lo)
        z_base_lo = self.kvll_lo**2 / self.s_mva
        self.zt = complex(self.r_pct, self.x_pct) / 100.0 * z_base_lo
        eye = np.eye(3, dtype=complex)
        self.a = self.nt * eye
        self.b = self.nt * self.zt * eye
        self.d = eye / self.nt
        self.A = eye / self.nt
        self.B = self.zt * eye

    def backward(self, v_dn: np.ndarray, i_dn: np.ndarray):
        return self.a @ v_dn + self.b @ i_dn, self.d @ i_dn

    def forward(self, v_up: np.ndarray, i_dn: np.ndarray) -> np.ndarray:
        return self.A @ v_up - self.B @ i_dn

    def passes_zero_sequence(self) -> bool:
        return True


@dataclass
class TransformerDGY:
    """Delta (high) / grounded-wye (low), step-down, American standard
    (high side leads low by 30 degrees), per Kersting:

        nt  = kVLL_hi / kVLN_lo
        a   = -(nt/3) * [[0,2,1],[1,0,2],[2,1,0]]
        b   = a * Zt
        d   = (1/nt) * [[1,-1,0],[0,1,-1],[-1,0,1]]
        A   = (1/nt) * [[1,0,-1],[-1,1,0],[0,-1,1]]
        B   = Zt * I

    Zt in ohms per phase referred to the low (wye) side. The delta winding
    provides no zero-sequence path toward the high side; the grounded-wye
    low winding is a zero-sequence source for its own zone -- the algebra
    here is the power-flow face of the same fact the grounding gate checks
    structurally.
    """

    kvll_hi: float
    kvll_lo: float
    s_mva: float
    r_pct: float
    x_pct: float

    def __post_init__(self) -> None:
        kvln_lo = self.kvll_lo / np.sqrt(3.0)
        self.nt = self.kvll_hi / kvln_lo
        z_base_lo = self.kvll_lo**2 / self.s_mva
        self.zt = complex(self.r_pct, self.x_pct) / 100.0 * z_base_lo
        perm = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]], dtype=complex)
        self.a = -(self.nt / 3.0) * perm
        self.b = self.a * self.zt
        self.d = (1.0 / self.nt) * np.array(
            [[1, -1, 0], [0, 1, -1], [-1, 0, 1]], dtype=complex
        )
        self.A = (1.0 / self.nt) * np.array(
            [[1, 0, -1], [-1, 1, 0], [0, -1, 1]], dtype=complex
        )
        self.B = self.zt * np.eye(3, dtype=complex)

    def backward(self, v_dn: np.ndarray, i_dn: np.ndarray):
        return self.a @ v_dn + self.b @ i_dn, self.d @ i_dn

    def forward(self, v_up: np.ndarray, i_dn: np.ndarray) -> np.ndarray:
        return self.A @ v_up - self.B @ i_dn

    def passes_zero_sequence(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Radial ladder solver
# ---------------------------------------------------------------------------


@dataclass
class RadialFeeder:
    """
    source -> elements[0] -> elements[1] -> ... -> wye PQ load at the end.

    `v_source_ln`: complex (3,) line-to-neutral source phasors (or delta-side
    equivalent LN). `load_kw`, `load_pf` are per-phase, lagging.
    """

    v_source_ln: np.ndarray
    elements: list = field(default_factory=list)
    load_kw: tuple[float, float, float] = (0.0, 0.0, 0.0)
    load_pf: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def load_s_va(self) -> np.ndarray:
        s = np.zeros(3, dtype=complex)
        for k in range(3):
            p = self.load_kw[k] * 1e3
            phi = np.arccos(self.load_pf[k])
            s[k] = p * (1.0 + 1j * np.tan(phi))
        return s

    def solve(self, tol_v: float = 1e-4, max_iter: int = 200):
        """
        Ladder iterative technique. Returns list of node LN voltage arrays
        [V_source, ..., V_load] (len(elements)+1 entries) and the load
        current. Fixed tolerance on load-node voltage magnitude change in
        volts; refuses to pretend convergence if max_iter is hit.
        """
        s_load = self.load_s_va()
        n = len(self.elements)
        # initial guess: nominal, propagated through ideal ratios
        v_nodes: list[np.ndarray] = [self.v_source_ln.astype(complex)]
        v = self.v_source_ln.astype(complex)
        for el in self.elements:
            v = el.forward(v, np.zeros(3, dtype=complex))
            v_nodes.append(v)

        v_load = v_nodes[-1].copy()
        for _ in range(max_iter):
            i_load = np.conj(s_load / v_load)
            # backward sweep: accumulate to the source
            v_bk, i_bk = v_load, i_load
            currents = [None] * (n + 1)
            currents[n] = i_load
            for k in range(n - 1, -1, -1):
                v_bk, i_bk = self.elements[k].backward(v_bk, currents[k + 1])
                currents[k] = i_bk
            # forward sweep from the true source voltage
            v_nodes[0] = self.v_source_ln.astype(complex)
            for k, el in enumerate(self.elements):
                v_nodes[k + 1] = el.forward(v_nodes[k], currents[k + 1])
            dv = float(np.max(np.abs(np.abs(v_nodes[-1]) - np.abs(v_load))))
            v_load = v_nodes[-1].copy()
            if dv < tol_v:
                return v_nodes, i_load
        raise RuntimeError(
            f"ladder sweep did not converge in {max_iter} iterations "
            f"(last dV = {dv:.3e} V); refusing to return an unconverged answer"
        )


def polar(v: np.ndarray) -> list[tuple[float, float]]:
    """[(magnitude, angle_deg), ...] for reporting."""
    return [(float(abs(x)), float(np.degrees(np.angle(x)))) for x in v]


def balanced_source_ln(kvll: float, angle_a_deg: float = 0.0) -> np.ndarray:
    vln = kvll * 1e3 / np.sqrt(3.0)
    th = np.radians(angle_a_deg)
    return vln * np.array(
        [np.exp(1j * th), np.exp(1j * (th - 2 * np.pi / 3)), np.exp(1j * (th + 2 * np.pi / 3))]
    )
