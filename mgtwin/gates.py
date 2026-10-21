"""
Verification gates.

A gate is a mechanical check that runs before anyone is allowed to believe a
model. Gates are deliberately conservative and deliberately dumb: they exist to
catch the defect classes that reliably survive human review, not to replace
engineering judgement.

Gates FAIL rather than warn when the finding is a safety issue. Everything else
warns. Nothing here is a substitute for a coordination study or a protection
review -- these are screening checks whose job is to stop a model reaching a
customer with a known-fatal flaw in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .ontology import (
    Converter,
    ConverterControl,
    EARTH_OC,
    LEGACY_ELEMENT,
    NOT_MAGNITUDE,
    PHASE_OC,
    UNIT,
    GroundingKind,
    Site,
    SynchronousGen,
    WindingConfig,
)
from .units import PLAUSIBLE, check_plausible


class Severity(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class Finding:
    gate: str
    severity: Severity
    message: str
    scope: str = ""  # island id, component id, etc.
    remedy: str = ""


@dataclass
class GateReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(f.severity == Severity.FAIL for f in self.findings)

    @property
    def warned(self) -> bool:
        return any(f.severity == Severity.WARN for f in self.findings)

    def by_severity(self, sev: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity == sev]

    def extend(self, other: "GateReport") -> "GateReport":
        self.findings.extend(other.findings)
        return self

    def render(self) -> str:
        if not self.findings:
            return "no findings"
        order = {Severity.FAIL: 0, Severity.WARN: 1, Severity.PASS: 2}
        lines = []
        for f in sorted(self.findings, key=lambda x: order[x.severity]):
            tag = f.severity.value.upper()
            scope = f" [{f.scope}]" if f.scope else ""
            lines.append(f"{tag:<4} {f.gate}{scope}: {f.message}")
            if f.remedy:
                lines.append(f"       -> {f.remedy}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gate 1: unit and per-unit plausibility
# ---------------------------------------------------------------------------


def gate_units(site: Site) -> GateReport:
    """
    Order-of-magnitude screening on declared quantities, plus per-unit base
    consistency across branches.

    Catches the two commonest classes: an impedance entered in ohms into a pu
    field, and a branch whose pu impedance refers to a base nobody rebased from.
    """
    rep = GateReport()

    for name, val in (
        ("s_base_mva", site.base.s_base_mva),
        ("v_base_kv", site.base.v_base_kv),
    ):
        msg = check_plausible(name, val)
        if msg:
            rep.findings.append(Finding("units", Severity.FAIL, msg, site.id))

    for br in site.branches:
        msg = check_plausible("z_pu", br.x_pu)
        if msg:
            rep.findings.append(
                Finding(
                    "units",
                    Severity.FAIL,
                    f"branch reactance: {msg}",
                    br.id,
                    "Confirm whether the value was entered in ohms rather than pu.",
                )
            )
        if br.r_pu > 0:
            xr = br.x_pu / br.r_pu
            msg = check_plausible("x_over_r", xr)
            if msg:
                rep.findings.append(
                    Finding(
                        "units",
                        Severity.WARN,
                        f"X/R = {xr:.2f} is unusual for this voltage class",
                        br.id,
                        "Check for a transposed R and X.",
                    )
                )
        # Per-unit base consistency: a branch declaring a base other than the
        # site base must be rebased before it is used.
        if (
            abs(br.s_base_mva - site.base.s_base_mva) > 1e-9
            or abs(br.v_base_kv - site.base.v_base_kv) > 1e-9
        ):
            rep.findings.append(
                Finding(
                    "units",
                    Severity.WARN,
                    f"pu impedance refers to {br.s_base_mva:g} MVA / "
                    f"{br.v_base_kv:g} kV, not the site base "
                    f"({site.base.s_base_mva:g} MVA / {site.base.v_base_kv:g} kV)",
                    br.id,
                    "Rebase with Base.rebase_impedance before assembling the network.",
                )
            )

    for c in site.converters:
        for name, val in (
            ("droop_pu", c.params.p_f_droop_pu),
            ("h_sec", c.params.h_sec),
            ("tp_sec", c.params.tp_sec),
            ("i_limit_pu", c.params.i_limit_transient_pu),
        ):
            if val is None:
                continue
            msg = check_plausible(name, val)
            if msg:
                rep.findings.append(Finding("units", Severity.FAIL, msg, c.id))

    # Transformer reactance: on its own rating a power transformer's is some 4-15 %. A value far
    # outside that was entered on another base - the site's, typically - and scales every fault
    # level and voltage drop through it.
    off = [tx for tx in site.transformers if check_plausible("tx_x_pu", tx.x_pu)]
    if off:
        lo, hi = PLAUSIBLE["tx_x_pu"]
        rep.findings.append(
            Finding(
                "units",
                Severity.FAIL,
                f"{len(off)} transformer reactance(s) outside [{lo:g}, {hi:g}] pu on their own rating: "
                + ", ".join(f"{tx.id} {tx.x_pu:g}" for tx in off[:8])
                + (f" and {len(off) - 8} more" if len(off) > 8 else ""),
                site.id,
                "Transformer.x_pu is on the transformer's own rating; rebase a system-base value "
                "(x_own = x_sys * s_rated / s_base).",
            )
        )

    # Transformer ratio vs declared bus voltages.
    bus_kv = {b.id: b.v_nom_kv for b in site.buses}
    for tx in site.transformers:
        for side, kv, bus in (
            ("hv", tx.hv_kv, tx.hv_bus),
            ("lv", tx.lv_kv, tx.lv_bus),
        ):
            nom = bus_kv.get(bus)
            if nom is None:
                continue
            if abs(kv - nom) / nom > 0.10:
                rep.findings.append(
                    Finding(
                        "units",
                        Severity.FAIL,
                        f"{side} winding rated {kv:g} kV but bus {bus!r} is "
                        f"nominally {nom:g} kV",
                        tx.id,
                        "A mismatched ratio silently scales every downstream current.",
                    )
                )

    if not rep.findings:
        rep.findings.append(
            Finding("units", Severity.PASS, "declared quantities are dimensionally plausible")
        )
    return rep


# ---------------------------------------------------------------------------
# Gate 2: grounding continuity across island transitions
# ---------------------------------------------------------------------------


def _zero_sequence_zones(site: Site, isl) -> tuple[list[set[str]], dict]:
    """
    Partition an island's buses into zero-sequence zones.

    Electrical connectivity: branches, transformer windings, and closed
    switches all connect buses. An OPEN switch between (a, b) severs EVERY
    edge between a and b -- breakers are in series with whatever they switch,
    and composed models often carry the switch and the impedance as parallel
    edges between the same pair.

    Zero-sequence connectivity is the same graph MINUS all transformer edges:
    a transformer never passes zero sequence between its sides in a way that
    can be assumed (delta blocks it outright; wye-wye depends on both neutrals
    and the core, which is exactly the sort of fact that must be declared, not
    inferred). Grounding then enters per zone:

      * a GroundingSource at a bus grounds that bus's zone, subject to
        available_when_islanded;
      * an energized transformer winding declared WYE_GROUNDED grounds the
        zone of the bus that winding connects to.

    Returns (zones_needing_check, info) where each zone is a set of bus ids
    containing at least one energized source or load, and info maps zone
    index -> dict(grounded_by=set, kinds=set, buses=set).
    """
    open_pairs: set[frozenset[str]] = set()
    sw_by_id = {sw.id: sw for sw in site.switches}
    opened = site.open_switch_ids(isl)
    for swid in opened:
        sw = sw_by_id[swid]
        open_pairs.add(frozenset((sw.from_bus, sw.to_bus)))

    def severed(a: str, b: str) -> bool:
        return frozenset((a, b)) in open_pairs

    # A composed model may carry the same physical transformer twice: as a
    # Branch (power-flow-exact impedance) and as a Transformer (winding
    # facts). For zero-sequence purposes any branch that parallels a declared
    # transformer IS that transformer and must not conduct in the zone graph.
    tx_pairs = {
        frozenset((tx.hv_bus, tx.lv_bus)) for tx in site.transformers
    }

    parent: dict[str, str] = {b.id: b.id for b in site.buses}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # -- electrical islands: everything conducts ---------------------------
    elec = dict(parent)

    def efind(x: str) -> str:
        while elec[x] != x:
            elec[x] = elec[elec[x]]
            x = elec[x]
        return x

    def eunion(a: str, b: str) -> None:
        ra, rb = efind(a), efind(b)
        if ra != rb:
            elec[ra] = rb

    for br in site.branches:
        if not severed(br.from_bus, br.to_bus):
            eunion(br.from_bus, br.to_bus)
    for tx in site.transformers:
        if not severed(tx.hv_bus, tx.lv_bus):
            eunion(tx.hv_bus, tx.lv_bus)
    for sw in site.switches:
        if sw.id not in opened and not severed(sw.from_bus, sw.to_bus):
            eunion(sw.from_bus, sw.to_bus)

    src_buses = {site.source(sid).bus for sid in isl.energized_source_ids}
    island_roots = {efind(b) for b in src_buses}

    # -- zero-sequence zones: transformers do not conduct -------------------
    for br in site.branches:
        if frozenset((br.from_bus, br.to_bus)) in tx_pairs:
            continue
        if not severed(br.from_bus, br.to_bus):
            union(br.from_bus, br.to_bus)
    for sw in site.switches:
        if sw.id not in opened and not severed(sw.from_bus, sw.to_bus):
            union(sw.from_bus, sw.to_bus)

    zones: dict[str, set[str]] = {}
    for b in site.buses:
        if efind(b.id) not in island_roots:
            continue  # outside the energized island entirely
        zones.setdefault(find(b.id), set()).add(b.id)

    load_buses = {ld.bus for ld in site.loads}
    grounded: dict[str, set[str]] = {r: set() for r in zones}
    kinds: dict[str, set[str]] = {r: set() for r in zones}

    declared_it: dict[str, set[str]] = {r: set() for r in zones}
    for g in site.grounding:
        if g.kind == GroundingKind.UNGROUNDED:
            # A declared IT (isolated) system: not a zero-sequence source, but a
            # design decision that must travel with a ground detector.
            r = find(g.bus)
            if r in declared_it:
                declared_it[r].add(g.id)
            continue
        if isl.grid_connected or g.available_when_islanded:
            r = find(g.bus)
            if r in grounded:
                grounded[r].add(g.id)
                kinds[r].add(g.kind.value)
    for tx in site.transformers:
        if severed(tx.hv_bus, tx.lv_bus):
            continue
        if efind(tx.hv_bus) not in island_roots:
            continue  # de-energized transformer grounds nothing
        if tx.lv_config == WindingConfig.WYE_GROUNDED:
            r = find(tx.lv_bus)
            if r in grounded:
                grounded[r].add(f"{tx.id}:lv_Yg")
                kinds[r].add("transformer_wye_grounded")
        if tx.hv_config == WindingConfig.WYE_GROUNDED:
            r = find(tx.hv_bus)
            if r in grounded:
                grounded[r].add(f"{tx.id}:hv_Yg")
                kinds[r].add("transformer_wye_grounded")

    # Every zone here is part of an energized electrical island (the bus set
    # was restricted to island_roots above), so every zone is energized
    # conductor and needs a grounding verdict -- including pure transit zones
    # with no local source or load, which is exactly the case of an MV network
    # backfed through a DYg service transformer.
    _ = load_buses  # retained for future severity shading
    checkable, info = [], {}
    for r, buses in sorted(zones.items()):
        checkable.append(buses)
        info[len(checkable) - 1] = dict(
            grounded_by=grounded[r], kinds=kinds[r], buses=buses,
            declared_it=declared_it[r],
        )
    return checkable, info


def gate_grounding(site: Site) -> GateReport:
    """
    Every zero-sequence ZONE of every island that carries sources or load must
    retain a zero-sequence source. Zones, not islands: delta windings partition
    a network into independent zero-sequence regions, so an island can be
    perfectly grounded at 480 V while its 13.8 kV level floats -- and a
    site-level check would miss exactly that.

    Requires topology. If neither grounding sources nor transformers are
    declared, the site cannot be assessed and the gate FAILS on absence.
    """
    rep = GateReport()

    if not site.grounding and not site.transformers:
        rep.findings.append(
            Finding(
                "grounding",
                Severity.FAIL,
                "no grounding sources and no transformer winding data declared; "
                "zero-sequence behaviour cannot be assessed",
                site.id,
                "Grounding is not optional metadata. Declare winding "
                "configurations and zero-sequence sources.",
            )
        )
        return rep

    for isl in site.islands:
        if not isl.grid_connected and not isl.energized_source_ids:
            rep.findings.append(
                Finding(
                    "grounding",
                    Severity.WARN,
                    "island declares no energized sources; nothing to assess "
                    "(if this island can really form, its sources are missing "
                    "from the declaration)",
                    isl.id,
                )
            )
            continue
        zones, info = _zero_sequence_zones(site, isl)
        for i, buses in enumerate(zones):
            meta = info[i]
            label = f"{isl.id}/zone({_zone_name(site, buses)})"
            if meta["grounded_by"]:
                k = ", ".join(sorted(meta["kinds"]))
                rep.findings.append(
                    Finding("grounding", Severity.PASS,
                            f"zero-sequence source present ({k})", label)
                )
                if meta["kinds"] == {"high_resistance_grounded"}:
                    rep.findings.append(
                        Finding(
                            "grounding",
                            Severity.WARN,
                            "only high-resistance grounding in this zone; "
                            "ground-fault current may be below relay sensitivity",
                            label,
                        )
                    )
            elif meta.get("declared_it"):
                ids = ", ".join(sorted(meta["declared_it"]))
                rep.findings.append(
                    Finding(
                        "grounding",
                        Severity.WARN,
                        f"ungrounded by DECLARED design (IT system: {ids}); first "
                        "earth fault produces no current and 1.73 pu on healthy "
                        "phases -- acceptable only with a ground detector (IMD or "
                        "59N), line-to-line insulation/arrester ratings, and a "
                        "first-fault operating rule",
                        label,
                        "Confirm the detector and the first-fault rule are in the "
                        "grounding design; otherwise ground the zone.",
                    )
                )
            else:
                sev = Severity.WARN if isl.grid_connected else Severity.FAIL
                rep.findings.append(
                    Finding(
                        "grounding",
                        sev,
                        "no zero-sequence source in this zone; SLG faults will "
                        "drive unfaulted phases toward ~1.73 pu and neutral-"
                        "current ground-fault detection will not operate",
                        label,
                        "Add a grounding transformer in this zone, ground a "
                        "machine neutral, or use a grounded-wye winding on the "
                        "in-zone side of a transformer.",
                    )
                )

    # Structural note unchanged: delta-delta passes zero sequence nowhere.
    for tx in site.transformers:
        if tx.hv_config == WindingConfig.DELTA and tx.lv_config == WindingConfig.DELTA:
            rep.findings.append(
                Finding(
                    "grounding",
                    Severity.WARN,
                    "delta-delta transformer blocks zero sequence in both "
                    "directions; grounding must be established independently "
                    "on each side",
                    tx.id,
                )
            )
    return rep


def _zone_name(site: Site, buses: set[str]) -> str:
    kvs = sorted({b.v_nom_kv for b in site.buses if b.id in buses})
    kv_s = "/".join(f"{k:g}kV" for k in kvs)
    if len(buses) <= 3:
        return f"{kv_s}:{','.join(sorted(buses))}"
    return f"{kv_s}:{len(buses)} buses"


# ---------------------------------------------------------------------------
# Gate 3: islanded protection adequacy
# ---------------------------------------------------------------------------

#: Dependability margin. A relay needs meaningfully more than pickup to operate
#: in a useful time on an inverse curve, not merely to exceed it.
PROTECTION_MARGIN = 1.5

#: Above this converter share of fault current, overcurrent grading is regarded
#: as structurally marginal regardless of the arithmetic.
CONVERTER_DOMINANCE = 0.80


def gate_protection(site: Site) -> GateReport:
    """
    Screening check: will the protection operate when islanded, in the settings
    group each state runs?

    Grid-connected, the utility contributes 5-10 pu of fault current and
    overcurrent grading works. Islanded, the sources are current-limited
    converters contributing perhaps 1.1-1.5 pu of their own rating. Relay
    settings carried over from the grid-connected study may simply never pick up.
    That is the finding that blocks energization, so it is FAIL, not WARN.

    Per islanded state, for the elements active in its settings group:
      * a closed breaker whose relay has no element active in the group is
        unprotected in that state: FAIL;
      * every phase overcurrent element with a pickup (50, 51, 67) against the
        available fault current, with the dependability margin;
      * converter-dominated states: the breakers whose primary protection rests
        on magnitude alone (no 21 or 51V element, and no unit element whose
        zones cover both sides of the breaker) are named in a WARN;
      * a breaker fed from both sides with only non-directional overcurrent sees
        the through-faults the far side can drive: phase, when that side's fault
        contribution exceeds the margin times its rating (a machine's, not a
        converter's 1.1-1.5 pu); earth, when each side has an earthing source.
        WARN, it needs direction (67 / 67N) or unit protection;
      * phase elements whose pickup is not yet set are named, not screened (WARN).
    Earth-fault elements are not screened: their current is the zero-sequence
    path's. Screening only - grading, settings and fault levels are a
    protection study's (Electrisim), whose results the envelope takes.
    """
    rep = GateReport()

    if not site.relays:
        rep.findings.append(
            Finding(
                "protection",
                Severity.WARN,
                "no relays declared; islanded protection cannot be screened",
                site.id,
            )
        )
        return rep

    for isl in site.islands:
        if isl.grid_connected:
            continue

        conv_a = 0.0
        mach_a = 0.0
        for sid in isl.energized_source_ids:
            src = site.source(sid)
            if isinstance(src, Converter):
                conv_a += src.fault_contribution_a(site.base)
            elif isinstance(src, SynchronousGen):
                mach_a += src.fault_contribution_a(site.base)

        total_a = conv_a + mach_a
        if total_a <= 0.0:
            rep.findings.append(
                Finding(
                    "protection",
                    Severity.FAIL,
                    "no energized sources declared for this island",
                    isl.id,
                )
            )
            continue

        open_sw = site.open_switch_ids(isl)
        in_service = [r for r in site.relays if r.switch_id not in open_sw]
        group = isl.settings_group

        oc_elements, unset = [], []
        for r in in_service:
            act = r.active(group)
            if not act:
                rep.findings.append(Finding("protection", Severity.FAIL,
                    f"no element of relay {r.id} is active in settings group {group!r}: breaker "
                    f"{r.switch_id} is unprotected in this state", f"{isl.id}/{r.id}",
                    "Give the relay an element in this group, or run the state in a group it has."))
                continue
            for e in act:
                if e.ansi in PHASE_OC and e.pickup_a is not None:
                    oc_elements.append((r, e))
                elif e.ansi in PHASE_OC:
                    unset.append(f"{r.id} {e.ansi}")
        if unset:
            rep.findings.append(Finding("protection", Severity.WARN,
                f"{len(unset)} phase overcurrent element(s) with no pickup set are not screened: "
                + ", ".join(unset[:8]) + (f" and {len(unset) - 8} more" if len(unset) > 8 else ""),
                f"{isl.id}/unset",
                "Declare the settings once the setting study has them (pickup in primary amperes)."))

        for r, e in oc_elements:
            scope = f"{isl.id}/{r.id}" if e.notes == LEGACY_ELEMENT else f"{isl.id}/{r.id}/{e.ansi}"
            ratio = total_a / e.pickup_a
            if ratio < 1.0:
                rep.findings.append(
                    Finding(
                        "protection",
                        Severity.FAIL,
                        f"available islanded fault current {total_a:,.0f} A is "
                        f"below pickup {e.pickup_a:,.0f} A "
                        f"(ratio {ratio:.2f}); this relay will never operate",
                        scope,
                        "Islanded settings group, voltage-restrained or "
                        "voltage-controlled overcurrent, differential, or "
                        "communication-assisted scheme per IEEE 2030.12.",
                    )
                )
            elif ratio < PROTECTION_MARGIN:
                rep.findings.append(
                    Finding(
                        "protection",
                        Severity.FAIL,
                        f"fault current {total_a:,.0f} A gives only "
                        f"{ratio:.2f}x pickup, below the {PROTECTION_MARGIN}x "
                        f"dependability margin; operating time will be very long",
                        scope,
                        "Lower the islanded pickup or change scheme; do not rely "
                        "on an inverse curve this close to pickup.",
                    )
                )
            else:
                rep.findings.append(
                    Finding(
                        "protection",
                        Severity.PASS,
                        f"{ratio:.1f}x pickup available when islanded",
                        scope,
                    )
                )

        protected = [r for r in in_service if r.active(group)]
        sw_ends = {sw.id: (sw.from_bus, sw.to_bus) for sw in site.switches}

        def unit_covers(r, act):
            """A unit element protects the breaker's circuit only if its zones take in both sides of the breaker
            (a feeder's 87L, a bus section between two 87B zones) - not a board's 87B on a breaker leaving it."""
            zone = {b for e in act if e.ansi in UNIT for b in e.zone_buses}
            return set(sw_ends[r.switch_id]) <= zone

        magnitude_only = [r.id for r in protected
                          if not unit_covers(r, r.active(group))
                          and not any(e.ansi in NOT_MAGNITUDE - UNIT for e in r.active(group))]
        share = conv_a / total_a
        if share >= CONVERTER_DOMINANCE and (magnitude_only or not protected):
            rep.findings.append(
                Finding(
                    "protection",
                    Severity.WARN,
                    f"{share:.0%} of islanded fault current comes from "
                    f"current-limited converters; magnitude-based grading is "
                    f"structurally marginal here even where the arithmetic passes"
                    + (f" - primary protection rests on magnitude at {', '.join(magnitude_only[:8])}"
                       + (f" and {len(magnitude_only) - 8} more" if len(magnitude_only) > 8 else "")
                       if magnitude_only else ""),
                    isl.id,
                    "Treat overcurrent as backup only; primary protection should "
                    "not depend on fault-current magnitude (87L/87B, or 51V).",
                )
            )
        elif share >= CONVERTER_DOMINANCE:
            rep.findings.append(Finding("protection", Severity.PASS,
                f"{share:.0%} converter fault current, and every breaker in service has protection "
                f"that does not rest on magnitude alone", isl.id))

        # Breakers fed from both sides: non-directional overcurrent sees a through-fault the far side can drive.
        # Phase: the far side's fault contribution above PROTECTION_MARGIN x its rated current (a synchronous
        # machine's 1/x''; a converter's 1.1-1.5 pu is not - a pickup above its load with margin does not see
        # it). Earth: an earthing source on each side at the breaker's voltage. Unless a 67 / 67N or a unit
        # element covering the breaker's circuit is there.
        sw_by_id = {sw.id: sw for sw in site.switches}
        kv = {b.id: b.v_nom_kv for b in site.buses}
        srcs = [site.source(s_) for s_ in isl.energized_source_ids]
        earths = [g.bus for g in site.grounding
                  if g.kind != GroundingKind.UNGROUNDED and g.available_when_islanded]
        earths += [tx.hv_bus if side == "hv" else tx.lv_bus for tx in site.transformers for side in ("hv", "lv")
                   if tx.provides_zero_sequence_path_to(side)]
        low_ratio = []
        for r in protected:
            act = r.active(group)
            unit = unit_covers(r, act)
            blind = [kind for kind, oc in (("phase", PHASE_OC), ("earth", EARTH_OC))
                     if not unit and any(e.ansi in oc and not e.directional for e in act)
                     and not any(e.ansi in oc and e.directional for e in act)]
            if not blind:
                continue
            sw = sw_by_id[r.switch_id]
            if sw.from_bus == sw.to_bus:
                continue
            cut = _sections(site, isl, open_extra=frozenset({sw.id}))
            ends = (cut(sw.from_bus), cut(sw.to_bus))
            if ends[0] == ends[1]:
                continue
            seen = []
            if "phase" in blind:
                ratios = []
                for e_ in ends:
                    side = [x for x in srcs if cut(x.bus) == e_]
                    rated = sum(x.s_rated_mva for x in side) * 1e6 / (3 ** 0.5 * site.base.v_base_kv * 1e3)
                    ratios.append(sum(x.fault_contribution_a(site.base) for x in side) / rated if rated else 0.0)
                if min(ratios) > 0 and max(ratios) > PROTECTION_MARGIN:
                    seen.append(f"phase (the far side gives {max(ratios):.1f}x its rated current)")
                elif min(ratios) > 0:
                    low_ratio.append(r.switch_id)
            if "earth" in blind:
                at_kv = [b for b in earths if abs(kv.get(b, 0) - kv[sw.from_bus]) < 1e-6]
                if all(any(cut(b) == e_ for b in at_kv) for e_ in ends):
                    seen.append("earth (an earthing source on each side)")
            if seen:
                kinds = [k.split(" ")[0] for k in seen]
                rep.findings.append(Finding("protection", Severity.WARN,
                    f"breaker {r.switch_id} is fed from both sides; relay {r.id}'s non-directional overcurrent "
                    f"sees through-faults: {'; '.join(seen)}",
                    f"{isl.id}/{r.id}",
                    "Add a directional element (" + " / ".join({"phase": "67", "earth": "67N"}[k] for k in kinds)
                    + ") or unit protection (87L) on this breaker."))
        if low_ratio:
            rep.findings.append(Finding("protection", Severity.PASS,
                f"{len(low_ratio)} breaker(s) fed from both sides by converters only, whose fault current is within "
                f"{PROTECTION_MARGIN}x their rating: a phase pickup above load with margin does not see the "
                f"through-fault (" + ", ".join(low_ratio[:6]) + (" ..." if len(low_ratio) > 6 else "") + ")",
                f"{isl.id}/through-faults"))

    return rep


# ---------------------------------------------------------------------------
# Gate 4: model honesty
# ---------------------------------------------------------------------------


def gate_model_honesty(site: Site) -> GateReport:
    """
    Refuse to let an assumed converter model support a protection conclusion.

    The public WECC generic GFM specification documents that steady-state current
    limiting and advanced voltage ride-through are not represented. If a
    converter's ride-through was assumed rather than identified from measured
    response, every fault-current number downstream is a placeholder, and the
    report must say so.
    """
    rep = GateReport()

    for c in site.converters:
        if not c.params.ride_through_identified:
            rep.findings.append(
                Finding(
                    "honesty",
                    Severity.WARN,
                    "current-limit / ride-through behaviour was assumed, not "
                    "identified from measured response; fault-current results "
                    "are provisional",
                    c.id,
                    "Identify from a measured step or fault response, or state "
                    "the assumption explicitly in the validity envelope.",
                )
            )
        if c.params.i_limit_steady_pu is None:
            rep.findings.append(
                Finding(
                    "honesty",
                    Severity.WARN,
                    "no steady-state current limit declared; sustained-fault "
                    "behaviour is unmodelled (a known gap in the public generics)",
                    c.id,
                )
            )
        if not c.firmware:
            rep.findings.append(
                Finding(
                    "honesty",
                    Severity.WARN,
                    "no firmware version recorded; ride-through behaviour is "
                    "firmware-bound and a vendor update silently invalidates "
                    "this model",
                    c.id,
                    "Record firmware at identification time and re-identify after "
                    "any update.",
                )
            )

    return rep


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------




def gate_forming(site: Site) -> GateReport:
    """
    Every electrically connected section of every islanded configuration must
    contain at least one grid-forming source (synchronous machine, or a
    converter not in grid-following control). A section of grid-following
    converters alone has no voltage/frequency reference and collapses the
    moment it is sectionalised -- the gate a topology study needs when the
    plant has few grid-forming units (CAMPAIGN-003: 4 GFM among 22).
    """
    from .ontology import ConverterControl

    rep = GateReport()
    gfm_ids = {m.id for m in site.machines} | {
        c.id for c in site.converters if c.control != ConverterControl.GFL
    }
    sw_by_id = {sw.id: sw for sw in site.switches}
    for isl in site.islands:
        if isl.grid_connected or not isl.energized_source_ids:
            continue
        opened = site.open_switch_ids(isl)
        open_pairs = {frozenset((sw_by_id[i].from_bus, sw_by_id[i].to_bus)) for i in opened}
        parent = {b.id: b.id for b in site.buses}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            if frozenset((a, b)) in open_pairs:
                return
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for br in site.branches:
            union(br.from_bus, br.to_bus)
        for tx in site.transformers:
            union(tx.hv_bus, tx.lv_bus)
        for sw in site.switches:
            if sw.id not in opened:
                union(sw.from_bus, sw.to_bus)
        energized = set(isl.energized_source_ids)
        src_bus = {s_.id: s_.bus for s_ in list(site.machines) + list(site.converters)}
        sections: dict = {}
        for sid in energized:
            r = find(src_bus[sid])
            d = sections.setdefault(r, {"gfm": set(), "gfl": set()})
            (d["gfm"] if sid in gfm_ids else d["gfl"]).add(sid)
        for r, d in sections.items():
            buses = {b.id for b in site.buses if find(b.id) == r}
            kv = sorted({b.v_nom_kv for b in site.buses if b.id in buses})
            label = f"{isl.id}/section({_zone_name(site, buses)})"
            mw = sum(ld.p_mw for ld in site.loads if ld.bus in buses)
            if d["gfm"]:
                rep.findings.append(Finding("forming", Severity.PASS,
                    f"{len(d['gfm'])} grid-forming source(s) {sorted(d['gfm'])} for "
                    f"{len(d['gfl'])} grid-following, {mw:.1f} MW load", label))
            else:
                rep.findings.append(Finding("forming", Severity.FAIL,
                    f"no grid-forming source in this section ({len(d['gfl'])} grid-following "
                    f"sources, {mw:.1f} MW load): no voltage/frequency reference -- the section "
                    f"collapses when sectionalised", label,
                    "Keep the section coupled to one containing a grid-forming unit, add a "
                    "grid-forming source to it, or make its sectionalising breaker a "
                    "sources-off interlock."))
    return rep



def gate_energisation(site: Site) -> GateReport:
    """
    Energisation gate: can the grid-forming sources of a section absorb the
    magnetising inrush of the transformers switched in together?

    Converter-formed islands have no stiff source: the inrush of a transformer
    is served by the current-limited PCS (and any machine) in the section, less
    the current already carrying the section's load. A step whose summed peak
    inrush exceeds that headroom drives the forming units into current limit,
    collapses section voltage and risks a cascade trip of grid-following units.
    The multiple `inrush_pu` is ASSUMED unless `inrush_identified` is set; the
    verdict says so. Static check only: remanence, point-on-wave and
    sympathetic inrush need an EMT study, which this gate names when it warns.
    """
    rep = GateReport()
    sw_by_id = {sw.id: sw for sw in site.switches}
    tx_by_id = {t.id: t for t in site.transformers}
    v_kv = site.base.v_base_kv
    for isl in site.islands:
        if not isl.energise_transformer_ids:
            continue
        opened = site.open_switch_ids(isl)
        open_pairs = {frozenset((sw_by_id[i].from_bus, sw_by_id[i].to_bus)) for i in opened}
        parent = {b.id: b.id for b in site.buses}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            if frozenset((a, b)) in open_pairs:
                return
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for br in site.branches:
            union(br.from_bus, br.to_bus)
        for tx in site.transformers:
            if tx.id not in isl.energise_transformer_ids:
                union(tx.hv_bus, tx.lv_bus)
        for sw in site.switches:
            if sw.id not in opened:
                union(sw.from_bus, sw.to_bus)
        energized = set(isl.energized_source_ids)
        txs = [tx_by_id[t] for t in isl.energise_transformer_ids if t in tx_by_id]
        roots = {find(tx.hv_bus) for tx in txs}
        for r in roots:
            buses = {b.id for b in site.buses if find(b.id) == r}
            step = [tx for tx in txs if find(tx.hv_bus) == r]
            if not any(s_.bus in buses for s_ in list(site.machines) + list(site.converters) if s_.id in energized):
                rep.findings.append(Finding("energisation", Severity.FAIL,
                    f"{', '.join(tx.id for tx in step)}: no energised source reaches the HV side in this "
                    f"state - the step energises nothing", f"{isl.id}/energise({_zone_name(site, buses)})",
                    "Close the switch that connects it (a normally open leg goes in the state's "
                    "closed_switches), or correct the state's open switches."))
                continue
            gfm_mva = sum(c.s_rated_mva * c.params.i_limit_transient_pu for c in site.converters
                          if c.id in energized and c.bus in buses and c.control != ConverterControl.GFL)
            mach_mva = sum(m.s_rated_mva / max(m.xd_subtransient_pu, 0.05) for m in site.machines
                           if m.id in energized and m.bus in buses)
            load_mva = sum((ld.p_mw ** 2 + ld.q_mvar ** 2) ** 0.5 for ld in site.loads if ld.bus in buses)
            gfl_mva = sum(c.s_rated_mva for c in site.converters
                          if c.id in energized and c.bus in buses and c.control == ConverterControl.GFL)
            # the forming units already carry whatever the grid-following fleet does not;
            # only the remainder of their current limit is available for inrush
            deficit_mva = max(load_mva - gfl_mva, 0.0)
            to_a = 1e6 / (3 ** 0.5 * v_kv * 1e3)
            head_pk = max(gfm_mva + mach_mva - deficit_mva, 0.0) * to_a * 2 ** 0.5
            inrush_pk = sum(tx.inrush_pu * tx.s_rated_mva for tx in step) * to_a
            assumed = any(not tx.inrush_identified for tx in step)
            label = f"{isl.id}/energise({_zone_name(site, buses)})"
            prov = "ASSUMED inrush multiple" if assumed else "IDENTIFIED inrush"
            names = ", ".join(tx.id for tx in step)
            ratio = inrush_pk / head_pk if head_pk > 0 else float("inf")
            msg = (f"{len(step)} transformer(s) [{names}] switched together: summed peak inrush "
                   f"{inrush_pk:.0f} A pk against forming headroom {head_pk:.0f} A pk "
                   f"(ratio {ratio:.2f}; {prov})")
            if ratio > 1.0:
                rep.findings.append(Finding("energisation", Severity.FAIL, msg, label,
                    "Sequence the closings (one transformer per step), use point-on-wave "
                    "closing with remanence tracking, pre-magnetise from the LV side, or add "
                    "forming capacity. Confirm with an EMT study."))
            elif ratio > 0.8 or assumed:
                rep.findings.append(Finding("energisation", Severity.WARN, msg, label,
                    "Within headroom on the assumed multiple; identify the inrush multiple "
                    "and check remanence / point-on-wave in an EMT study."))
            else:
                rep.findings.append(Finding("energisation", Severity.PASS, msg, label))
    return rep


def _sections(site: Site, isl, without: frozenset = frozenset(), open_extra: frozenset = frozenset()):
    """Union-find over an island's closed switches (less `open_extra`), branches and transformers (less `without`)."""
    sw_by_id = {sw.id: sw for sw in site.switches}
    opened = site.open_switch_ids(isl) | set(open_extra)
    open_pairs = {frozenset((sw_by_id[i].from_bus, sw_by_id[i].to_bus)) for i in opened if i in sw_by_id}
    parent = {b.id: b.id for b in site.buses}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if frozenset((a, b)) in open_pairs:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for br in site.branches:
        union(br.from_bus, br.to_bus)
    for tx in site.transformers:
        if tx.id not in without:
            union(tx.hv_bus, tx.lv_bus)
    for sw in site.switches:
        if sw.id not in opened:
            union(sw.from_bus, sw.to_bus)
    return find


def gate_loading(site: Site) -> GateReport:
    """
    Loading gate: can the sources a state energises carry the load it leaves connected?

    The other gates check that a section has a voltage reference, a zero-sequence path and
    room for its inrush - not that its sources can carry its load. A black-start state that
    leaves the halls' feeders closed on a 3 MVA genset passes all of them. Per energised
    section of each islanded state: FAIL when the load's apparent power exceeds the rating
    of every source energised in it, storage included; WARN when it exceeds the sustained
    sources' (machines and converters without storage) - the section then runs on its
    batteries; and FAIL for a transformer feeding a side with no source of its own whose load
    exceeds its rating. Ratings, not a load flow: losses, voltage and temperature derating are not
    counted, so a PASS is necessary, not sufficient.
    """
    rep = GateReport()
    tx_by_id = {t.id: t for t in site.transformers}
    srcs = {s_.id: s_ for s_ in list(site.machines) + list(site.converters)}

    def s_load(buses):
        p = sum(ld.p_mw for ld in site.loads if ld.bus in buses)
        q = sum(ld.q_mvar for ld in site.loads if ld.bus in buses)
        return (p * p + q * q) ** 0.5

    for isl in site.islands:
        if isl.grid_connected or not isl.energized_source_ids:
            continue
        find = _sections(site, isl)
        energized = [srcs[i] for i in isl.energized_source_ids if i in srcs]
        roots = {find(s_.bus) for s_ in energized}
        for r in roots:
            buses = {b.id for b in site.buses if find(b.id) == r}
            here = [s_ for s_ in energized if s_.bus in buses]
            total = sum(s_.s_rated_mva for s_ in here)
            sustained = sum(s_.s_rated_mva for s_ in here if not getattr(s_, "energy_mwh", None))
            load = s_load(buses)
            label = f"{isl.id}/loading({_zone_name(site, buses)})"
            if load > total * 1.0 + 1e-9:
                rep.findings.append(Finding("loading", Severity.FAIL,
                    f"the section's load {load:.2f} MVA exceeds the rating of every source energised in it, "
                    f"{total:.2f} MVA ({len(here)} source(s)): it cannot be carried", label,
                    "Open the feeders of the loads the state does not intend to carry (as a black-start "
                    "step should), or energise more sources."))
            elif load > sustained + 1e-9 and total > sustained:
                rep.findings.append(Finding("loading", Severity.WARN,
                    f"the section's load {load:.2f} MVA exceeds its sustained sources' {sustained:.2f} MVA: "
                    f"it runs on storage ({total - sustained:.2f} MVA of it)", label,
                    "Check the storage's energy against how long the state lasts."))
            else:
                rep.findings.append(Finding("loading", Severity.PASS,
                    f"load {load:.2f} MVA within {sustained:.2f} MVA sustained, {total:.2f} MVA in all", label))
            # Transformers carrying load with no source behind them.
            for tx in site.transformers:
                if tx.hv_bus not in buses:
                    continue
                cut = _sections(site, isl, frozenset({tx.id}))
                for end in (tx.hv_bus, tx.lv_bus):
                    side = {b.id for b in site.buses if cut(b.id) == cut(end)}
                    if side == buses or any(s_.bus in side for s_ in here):
                        continue
                    behind = s_load(side)
                    if behind > tx.s_rated_mva * 1.0 + 1e-9:
                        rep.findings.append(Finding("loading", Severity.FAIL,
                            f"{tx.id} ({tx.s_rated_mva:g} MVA) feeds {behind:.2f} MVA of load on a side with "
                            f"no source of its own ({100 * behind / tx.s_rated_mva:.0f} %)", f"{isl.id}/{tx.id}",
                            "Size the transformer (or its bank) for the load it feeds - for dual-corded "
                            "loads, the whole load when the other cord is lost."))
    return rep


ALL_GATES = (gate_forming, gate_units, gate_grounding, gate_protection, gate_model_honesty, gate_energisation,
             gate_loading)


def run_all(site: Site) -> GateReport:
    rep = GateReport()
    for g in ALL_GATES:
        rep.extend(g(site))
    return rep
