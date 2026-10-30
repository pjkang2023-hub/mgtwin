"""
Canonical microgrid ontology.

Design rules
------------
1. Grounding is a first-class entity, not an attribute buried in a transformer.
   It changes on islanding and positive-sequence models are blind to it.
2. Converter parameters follow the shape of the WECC generic library
   (REGFM_A1 droop GFM, REGFM_B1 VSM GFM, REGCA/REECA GFL) so that composed
   models can be exported to tools that reviewers already accept.
3. Every island configuration is enumerable. The topology space of a microgrid
   is small -- bus, branch, transformer, source, load, switch -- which is what
   makes automated composition tractable here and not in a process plant.
4. Nothing carries an implicit base. Impedances declare the base they refer to.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .units import Base, PowerSign

# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


class GroundingKind(str, Enum):
    UNGROUNDED = "ungrounded"
    SOLID = "solidly_grounded"
    LOW_RESISTANCE = "low_resistance_grounded"
    HIGH_RESISTANCE = "high_resistance_grounded"
    REACTANCE = "reactance_grounded"
    RESONANT = "resonant_grounded"


class GroundingSource(BaseModel):
    """
    A zero-sequence source: a grounded transformer neutral, a dedicated
    grounding transformer (zig-zag or grounded-wye/delta), or a machine neutral.

    `available_when_islanded` is the field that matters. A microgrid whose only
    grounding reference sits on the utility side of the point of interconnection
    becomes ungrounded the moment it islands, with SLG overvoltages up to ~sqrt(3)
    on the unfaulted phases. That is a safety finding, not an accuracy finding.
    """

    id: str
    bus: str
    kind: GroundingKind
    z0_ohm: float = Field(0.0, ge=0.0, description="Zero-sequence grounding impedance")
    available_when_islanded: bool
    notes: str = ""

    @model_validator(mode="after")
    def _impedance_consistent(self) -> "GroundingSource":
        if self.kind == GroundingKind.SOLID and self.z0_ohm > 1.0:
            raise ValueError(
                f"grounding source {self.id!r} declares solid grounding but "
                f"z0_ohm={self.z0_ohm}; declare LOW_RESISTANCE or REACTANCE instead"
            )
        if self.kind == GroundingKind.UNGROUNDED and self.z0_ohm != 0.0:
            raise ValueError(
                f"grounding source {self.id!r} is UNGROUNDED; z0_ohm is meaningless"
            )
        return self


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class Bus(BaseModel):
    id: str
    v_nom_kv: float = Field(gt=0.0)
    is_poi: bool = False


class Branch(BaseModel):
    id: str
    from_bus: str
    to_bus: str
    r_pu: float = Field(ge=0.0)
    x_pu: float = Field(gt=0.0)
    b_pu: float = 0.0
    s_base_mva: float = Field(gt=0.0, description="Base the pu impedance refers to")
    v_base_kv: float = Field(gt=0.0, description="Base the pu impedance refers to")
    rating_mva: float = Field(gt=0.0)


class WindingConfig(str, Enum):
    DELTA = "delta"
    WYE_GROUNDED = "wye_grounded"
    WYE_UNGROUNDED = "wye_ungrounded"


class Transformer(BaseModel):
    id: str
    hv_bus: str
    lv_bus: str
    s_rated_mva: float = Field(gt=0.0)
    hv_kv: float = Field(gt=0.0)
    lv_kv: float = Field(gt=0.0)
    x_pu: float = Field(gt=0.0, description="On own rating")
    r_pu: float = Field(0.0, ge=0.0, description="On own rating")
    hv_config: WindingConfig
    lv_config: WindingConfig
    inrush_pu: float = Field(
        10.0, gt=0.0,
        description="Peak magnetising inrush as a multiple of rated rms current on first "
                    "energisation (typ. 8-12 for distribution units). Default is ASSUMED; "
                    "set inrush_identified=True once taken from a test or the nameplate.")
    inrush_identified: bool = False

    def provides_zero_sequence_path_to(self, side: Literal["hv", "lv"]) -> bool:
        """A grounded-wye winding provides a zero-sequence path to its own side."""
        cfg = self.hv_config if side == "hv" else self.lv_config
        return cfg == WindingConfig.WYE_GROUNDED


class Switch(BaseModel):
    """Breaker, recloser, or sectionalizer. `is_pcc` marks the utility interface."""

    id: str
    from_bus: str
    to_bus: str
    normally_closed: bool = True
    is_pcc: bool = False


# ---------------------------------------------------------------------------
# Protection
# ---------------------------------------------------------------------------


#: ANSI device functions a relay may declare, and what each is.
ANSI_FUNCTIONS = {
    "50": "phase instantaneous overcurrent",
    "51": "phase time overcurrent",
    "50N": "residual instantaneous overcurrent",
    "51N": "residual time overcurrent",
    "50G": "ground instantaneous overcurrent",
    "51G": "ground time overcurrent",
    "67": "directional phase overcurrent",
    "67N": "directional earth-fault overcurrent",
    "51V": "voltage-restrained / -controlled overcurrent",
    "21": "distance",
    "87L": "line differential",
    "87B": "bus differential",
    "87T": "transformer differential",
    "27": "undervoltage",
    "59": "overvoltage",
    "59N": "neutral overvoltage",
    "81U": "underfrequency",
    "81O": "overfrequency",
    "50BF": "breaker failure",
}
#: Phase elements whose operation rests on fault-current magnitude.
PHASE_OC = {"50", "51", "67"}
#: Earth-fault elements: their magnitude is the zero-sequence path's, which the gates do not compute.
EARTH_OC = {"50N", "51N", "50G", "51G", "67N"}
#: The note on the element a relay's legacy fields declare.
LEGACY_ELEMENT = "from the relay's pickup_a / curve / time_dial / directional"
#: Elements whose discrimination does not rest on current magnitude alone.
UNIT = {"87L", "87B", "87T"}
NOT_MAGNITUDE = UNIT | {"51V", "21"}


class ProtectionFunction(BaseModel):
    """
    One protection element on a breaker's relay. Pickup is primary amperes, as it is set in the
    field; None for elements set otherwise (a differential's bias, a voltage or frequency element).
    `groups` names the settings groups the element is active in - empty means every group - and a
    state selects one (IslandConfig.settings_group). A unit element's zone is the buses it covers.
    """

    ansi: str
    pickup_a: float | None = Field(None, gt=0.0, description="Primary amperes")
    direction: Literal["none", "forward", "reverse"] = "none"
    curve: str = ""
    time_dial: float | None = Field(None, gt=0.0)
    groups: list[str] = Field(default_factory=list)
    zone_buses: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> "ProtectionFunction":
        if self.ansi not in ANSI_FUNCTIONS:
            raise ValueError(f"unknown ANSI function {self.ansi!r} (one of {sorted(ANSI_FUNCTIONS)})")
        if self.ansi.startswith("67") and self.direction == "none":
            raise ValueError(f"a {self.ansi} element needs a direction")
        if self.ansi in UNIT and not self.zone_buses:
            raise ValueError(f"a {self.ansi} element needs its zone's buses")
        return self

    @property
    def directional(self) -> bool:
        return self.direction != "none" or self.ansi in UNIT

    def active_in(self, group: str) -> bool:
        return not self.groups or group in self.groups


class OvercurrentRelay(BaseModel):
    """
    The relay on one breaker. Its protection elements are `functions`; the legacy fields
    (pickup_a, curve, time_dial, directional) declare one phase overcurrent element active in
    every settings group - a 67 forward when `directional`, else a 51 - so a Site written before
    v0.6.8 reads as it did. Pickup is stored in amperes, deliberately, because that is how it is
    set in the field and converting it is where errors enter.
    """

    id: str
    switch_id: str
    ct_ratio: float | None = Field(None, gt=0.0, description="e.g. 600.0 for 600:5; None until the design declares it")
    pickup_a: float | None = Field(None, gt=0.0, description="Primary amperes")
    curve: str = "IEC-VI"
    time_dial: float | None = Field(None, gt=0.0)
    directional: bool = False
    functions: list[ProtectionFunction] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _has_an_element(self) -> "OvercurrentRelay":
        if self.pickup_a is None and not self.functions:
            raise ValueError(f"relay {self.id!r} declares no protection element (pickup_a or functions)")
        return self

    def legacy_element(self) -> ProtectionFunction | None:
        if self.pickup_a is None:
            return None
        return ProtectionFunction(ansi="67" if self.directional else "51", pickup_a=self.pickup_a,
                                  direction="forward" if self.directional else "none",
                                  curve=self.curve, time_dial=self.time_dial, notes=LEGACY_ELEMENT)

    def elements(self) -> list[ProtectionFunction]:
        leg = self.legacy_element()
        return ([leg] if leg else []) + list(self.functions)

    def active(self, group: str) -> list[ProtectionFunction]:
        return [e for e in self.elements() if e.active_in(group)]


#: The relay on a breaker, with the protection functions it carries.
Relay = OvercurrentRelay


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class ConverterControl(str, Enum):
    """
    Control class. GFM_DROOP / GFM_VSM map onto the WECC generic library shapes
    (REGFM_A1 and REGFM_B1 respectively); GFL maps onto REGCA/REECA.
    """

    GFL = "grid_following"
    GFM_DROOP = "grid_forming_droop"
    GFM_VSM = "grid_forming_vsm"
    GFM_HYBRID = "grid_forming_hybrid"  # REGFM_C1 shape


class ConverterParams(BaseModel):
    """
    Parameter vector for a converter's RMS-timescale representation.

    Known gap, carried explicitly: the WECC generic GFM specification for
    REGFM_A1 states that steady-state current limiting and advanced voltage
    ride-through are NOT included and are deferred to a later generic model.
    Those two behaviours are exactly what sets islanded fault current, so they
    are represented here as separate fields that must be either identified from
    response data or declared unknown. `ride_through_identified` is the honesty
    flag: False means any protection conclusion drawn from this model is
    provisional.
    """

    p_f_droop_pu: float | None = Field(
        None, gt=0.0, description="P-f droop, pu on own rating (GFM only)"
    )
    q_v_droop_pu: float | None = Field(None, gt=0.0)
    h_sec: float | None = Field(None, ge=0.0, description="Virtual inertia constant")
    tp_sec: float = Field(0.05, gt=0.0, description="Power measurement filter")
    tv_sec: float = Field(0.02, gt=0.0, description="Voltage measurement filter")

    # Current limiting -- the part the public generics omit.
    i_limit_transient_pu: float = Field(
        1.2, gt=0.0, description="Transient current limit, pu on own rating"
    )
    i_limit_steady_pu: float | None = Field(
        None, gt=0.0, description="Steady-state limit; None = not identified"
    )
    ride_through_identified: bool = Field(
        False,
        description="True only if ride-through/current-limit behaviour was "
        "identified from measured response, not assumed",
    )

    @model_validator(mode="after")
    def _steady_le_transient(self) -> "ConverterParams":
        if (
            self.i_limit_steady_pu is not None
            and self.i_limit_steady_pu > self.i_limit_transient_pu
        ):
            raise ValueError(
                "steady-state current limit exceeds transient limit; "
                "check which is which"
            )
        return self


class Converter(BaseModel):
    """Inverter-based resource: PV, BESS, wind, or a hybrid plant."""

    id: str
    bus: str
    s_rated_mva: float = Field(gt=0.0)
    control: ConverterControl
    params: ConverterParams
    sign: PowerSign = PowerSign.GENERATOR
    vendor: str = ""
    firmware: str = Field(
        "", description="Firmware version. Ride-through behaviour is firmware-bound."
    )
    energy_mwh: float | None = Field(None, gt=0.0, description="BESS only")

    @model_validator(mode="after")
    def _control_params_consistent(self) -> "Converter":
        is_gfm = self.control in (
            ConverterControl.GFM_DROOP,
            ConverterControl.GFM_VSM,
            ConverterControl.GFM_HYBRID,
        )
        if is_gfm and self.params.p_f_droop_pu is None:
            raise ValueError(
                f"converter {self.id!r} declares grid-forming control but no "
                f"P-f droop gain"
            )
        if self.control == ConverterControl.GFM_VSM and self.params.h_sec is None:
            raise ValueError(
                f"converter {self.id!r} declares VSM control but no inertia constant"
            )
        return self

    def fault_contribution_a(self, base: Base) -> float:
        """
        Symmetrical fault current contribution in amperes, islanded.

        A converter is a current-limited source, not a voltage source behind
        subtransient reactance. Contribution is its own rated current times its
        transient limit -- typically 1.1-1.5 pu, not the 5-10 pu a synchronous
        machine would deliver.
        """
        i_rated_a = (self.s_rated_mva * 1e6) / (3**0.5 * base.v_base_kv * 1e3)
        return i_rated_a * self.params.i_limit_transient_pu


class SynchronousGen(BaseModel):
    """Diesel/gas genset or turbine. Contributes real short-circuit current."""

    id: str
    bus: str
    s_rated_mva: float = Field(gt=0.0)
    h_sec: float = Field(gt=0.0)
    xd_subtransient_pu: float = Field(gt=0.0, le=1.0)
    droop_pu: float = Field(0.04, gt=0.0)
    isochronous_capable: bool = True
    neutral_grounded: bool = False

    def fault_contribution_a(self, base: Base) -> float:
        """I_f ~= 1.0 / X''_d, on own rating."""
        i_rated_a = (self.s_rated_mva * 1e6) / (3**0.5 * base.v_base_kv * 1e3)
        return i_rated_a / self.xd_subtransient_pu


class LoadModel(str, Enum):
    ZIP = "zip"
    CONSTANT_POWER = "constant_power"
    MOTOR = "induction_motor"


class Load(BaseModel):
    id: str
    bus: str
    p_mw: float
    q_mvar: float = 0.0
    model: LoadModel = LoadModel.ZIP
    zip_p: tuple[float, float, float] = (0.0, 0.0, 1.0)  # Z, I, P fractions
    zip_q: tuple[float, float, float] = (0.0, 0.0, 1.0)
    critical: bool = False
    sign: PowerSign = PowerSign.LOAD

    @field_validator("zip_p", "zip_q")
    @classmethod
    def _fractions_sum_to_one(cls, v: tuple[float, float, float]):
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError(f"ZIP fractions must sum to 1.0, got {sum(v):g}")
        return v


# ---------------------------------------------------------------------------
# Island configurations and the site itself
# ---------------------------------------------------------------------------


class IslandConfig(BaseModel):
    """
    One reachable operating configuration. Enumerating these is the whole point:
    a gate that passes in grid-connected mode and fails in island 2 is the
    finding you want before energization, not after.
    """

    id: str
    description: str = ""
    open_switches: list[str] = Field(default_factory=list)
    closed_switches: list[str] = Field(
        default_factory=list,
        description="Normally open switches this state closes (a tie, an alternate leg after a "
                    "transfer). A normally open switch not listed here is open.")
    grid_connected: bool
    load_factors: dict[str, float] = Field(
        default_factory=dict,
        description="Load id -> the factor on its p and q in this state: a load transfer a Site's fixed loads "
                    "cannot express - the dual-corded halls' cord-A load doubling when board B is lost "
                    "({'IT_cordA': 2.0}). Loads not listed run at 1.")
    settings_group: str = Field(
        "A", description="The protection settings group the relays run in this state (black start, "
                         "islanded and grid-connected states typically differ).")
    energized_source_ids: list[str] = Field(default_factory=list)
    energise_transformer_ids: list[str] = Field(
        default_factory=list,
        description="Transformers switched in SIMULTANEOUSLY in this state (one ATS step, one "
                    "black-start step). The energisation gate checks their summed inrush "
                    "against the grid-forming current headroom of the section they land in.")
    energise_mode: Literal["switch", "ramp"] = Field(
        "switch",
        description="'switch': energise_transformer_ids are switched in from the live side (inrush against the "
                    "forming headroom). 'ramp': they are connected dead and the section's forming sources ramp "
                    "the voltage from zero - no inrush; the check is magnetising + charging + connected load "
                    "against the forming rating in the section they ramp.")


class Site(BaseModel):
    """A complete canonical microgrid model."""

    id: str
    name: str = ""
    base: Base
    buses: list[Bus] = Field(default_factory=list)
    branches: list[Branch] = Field(default_factory=list)
    transformers: list[Transformer] = Field(default_factory=list)
    switches: list[Switch] = Field(default_factory=list)
    relays: list[OvercurrentRelay] = Field(default_factory=list)
    converters: list[Converter] = Field(default_factory=list)
    machines: list[SynchronousGen] = Field(default_factory=list)
    loads: list[Load] = Field(default_factory=list)
    grounding: list[GroundingSource] = Field(default_factory=list)
    islands: list[IslandConfig] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    # -- lookups -----------------------------------------------------------

    def bus_ids(self) -> set[str]:
        return {b.id for b in self.buses}

    def source_ids(self) -> set[str]:
        return {c.id for c in self.converters} | {m.id for m in self.machines}

    def state_loads(self, isl: "IslandConfig") -> list[tuple["Load", float, float]]:
        """Each load with its p and q in a state: its own, times the state's load factor for it."""
        return [(ld, ld.p_mw * isl.load_factors.get(ld.id, 1.0), ld.q_mvar * isl.load_factors.get(ld.id, 1.0))
                for ld in self.loads]

    def open_switch_ids(self, isl: "IslandConfig") -> set[str]:
        """The switches open in a state: those it opens, and the normally open ones it does not close."""
        closes = set(isl.closed_switches)
        return set(isl.open_switches) | {sw.id for sw in self.switches if not sw.normally_closed and sw.id not in closes}

    def source(self, sid: str) -> Converter | SynchronousGen:
        for c in self.converters:
            if c.id == sid:
                return c
        for m in self.machines:
            if m.id == sid:
                return m
        raise KeyError(sid)

    def relay_for_switch(self, switch_id: str) -> OvercurrentRelay | None:
        for r in self.relays:
            if r.switch_id == switch_id:
                return r
        return None

    @model_validator(mode="after")
    def _referential_integrity(self) -> "Site":
        buses = self.bus_ids()
        problems: list[str] = []

        def need(bus: str, owner: str) -> None:
            if bus not in buses:
                problems.append(f"{owner} references unknown bus {bus!r}")

        for br in self.branches:
            need(br.from_bus, f"branch {br.id!r}")
            need(br.to_bus, f"branch {br.id!r}")
        for tx in self.transformers:
            need(tx.hv_bus, f"transformer {tx.id!r}")
            need(tx.lv_bus, f"transformer {tx.id!r}")
        for sw in self.switches:
            need(sw.from_bus, f"switch {sw.id!r}")
            need(sw.to_bus, f"switch {sw.id!r}")
        for c in self.converters:
            need(c.bus, f"converter {c.id!r}")
        for m in self.machines:
            need(m.bus, f"machine {m.id!r}")
        for ld in self.loads:
            need(ld.bus, f"load {ld.id!r}")
        for g in self.grounding:
            need(g.bus, f"grounding source {g.id!r}")

        switch_ids = {s.id for s in self.switches}
        for r in self.relays:
            if r.switch_id not in switch_ids:
                problems.append(f"relay {r.id!r} references unknown switch {r.switch_id!r}")
            for fn in r.functions:
                for b in fn.zone_buses:
                    need(b, f"relay {r.id!r} {fn.ansi} zone")

        srcs = self.source_ids()
        for isl in self.islands:
            for sid in isl.energized_source_ids:
                if sid not in srcs:
                    problems.append(f"island {isl.id!r} references unknown source {sid!r}")
            for swid in list(isl.open_switches) + list(isl.closed_switches):
                if swid not in switch_ids:
                    problems.append(f"island {isl.id!r} references unknown switch {swid!r}")
            load_ids = {ld.id for ld in self.loads}
            for lid, k in isl.load_factors.items():
                if lid not in load_ids:
                    problems.append(f"island {isl.id!r} scales unknown load {lid!r}")
                if k < 0:
                    problems.append(f"island {isl.id!r} scales load {lid!r} by a negative factor {k:g}")
            both = set(isl.open_switches) & set(isl.closed_switches)
            if both:
                problems.append(f"island {isl.id!r} both opens and closes {sorted(both)}")

        if problems:
            raise ValueError("; ".join(problems))
        return self
