"""
Stage 6: the declared validity envelope.

A twin that passes its gates is still only valid somewhere: in the states it
enumerates, for the phenomena its tier represents, within the amplitudes its
parameters were identified at. The envelope is the reviewable artifact that
says where - and `check` answers a question with INSIDE, PROVISIONAL or
OUTSIDE plus the reasons, so a result can be quoted together with its limits.

Rules the envelope keeps:

  * No invented limits. A bound the Site and the identification records do not
    supply is UNDECLARED, never a typical value.
  * The gates bound the envelope. A state that FAILs a gate is OUTSIDE for the
    phenomena that gate speaks to; a site-wide FAIL (units) puts everything
    OUTSIDE until it clears.
  * Tier limits are named. Phenomena this package does not represent (EMT,
    network load flow, dispatch, synchronising) are OUTSIDE with the tool that
    answers them, as the runner defers them.
  * Assumed parameters make a result PROVISIONAL, not invalid; an amplitude
    beyond the identification experiments is an extrapolation and says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from .gates import GateReport, Severity, run_all
from .ontology import ConverterControl, Site


class Phenomenon(str, Enum):
    LOADING = "loading"                    # power balance against source ratings
    FREQUENCY = "frequency"                # RMS frequency response to load steps
    PROTECTION = "protection"              # islanded fault duty against relay pickup
    GROUNDING = "grounding"                # zero-sequence paths, SLG overvoltage (radial)
    ENERGISATION = "energisation"          # static summed inrush against forming headroom
    VOLTAGE = "voltage"                    # bus voltages, losses: a network load flow
    EMT = "emt"                            # waveforms, point-on-wave, harmonics, sub-cycle limiting
    DISPATCH = "dispatch"                  # EMS dispatch, black-start sequencing in time
    RECONNECTION = "reconnection"          # synchronising and reconnection


class Support(str, Enum):
    SUPPORTED = "supported"
    PROVISIONAL = "provisional"
    EXCLUDED = "excluded"


class Inside(str, Enum):
    INSIDE = "inside"
    PROVISIONAL = "provisional"
    OUTSIDE = "outside"


#: Phenomena outside this package's tier, and what answers them.
EXCLUDED_PHENOMENA = {
    Phenomenon.VOLTAGE: "no network solver here by design: pandapower or OpenDSS",
    Phenomenon.EMT: "an EMT study (point-on-wave, remanence, harmonics, sub-cycle current limiting)",
    Phenomenon.DISPATCH: "an EMS dispatch model (the test runner defers DISP cases)",
    Phenomenon.RECONNECTION: "a synchroniser model (the test runner defers REC cases)",
}

#: Which gate's FAIL takes a state outside which phenomena. A forming FAIL takes it out of all.
GATE_PHENOMENA = {
    "loading": {Phenomenon.LOADING, Phenomenon.FREQUENCY},
    "protection": {Phenomenon.PROTECTION},
    "grounding": {Phenomenon.GROUNDING, Phenomenon.PROTECTION},
    "energisation": {Phenomenon.ENERGISATION},
    "forming": set(Phenomenon),
}


@dataclass
class IdentificationRecord:
    """
    What identify_multi established for one source, as the envelope needs it:
    its verdict and the load-step amplitudes (pu on the source's rating) the
    experiments spanned. A step beyond them is an extrapolation.
    """

    source_id: str
    verdict: str                       # an identify.Verdict value
    step_range_pu: tuple[float, float]
    parameters: dict[str, float] = field(default_factory=dict)

    @property
    def identified(self) -> bool:
        return self.verdict == "identified"


@dataclass
class ProtectionStudyRecord:
    """
    A protection study's verdict on one state, from the tool that computes fault currents
    (Electrisim: IEC 60909 and EMT fault levels at each relay, grading, settings). `passed` is
    the study's own conclusion; `models_identified` says whether its converter current limits and
    ride-through were identified from measured response or the vendors' assumed models.
    """

    state_id: str
    tool: str
    passed: bool
    models_identified: bool = False
    reference: str = ""
    notes: str = ""


def load_protection_studies(path) -> list[ProtectionStudyRecord]:
    """Records from a JSON list of {state_id, tool, passed, models_identified?, reference?, notes?}."""
    with open(path, encoding="utf-8") as f:
        return [ProtectionStudyRecord(**r) for r in json.load(f)]


@dataclass
class StateEnvelope:
    state_id: str
    grid_connected: bool
    sources: list[str]
    rating_mva: float                  # every energised source
    sustained_mva: float               # less storage
    excluded: dict[str, list[str]]     # phenomenon -> gate findings that exclude it


@dataclass
class Envelope:
    site_id: str
    version: str
    phenomena: dict[Phenomenon, tuple[Support, str]]
    states: dict[str, StateEnvelope]
    site_wide_fails: list[str]
    identifications: dict[str, IdentificationRecord]
    assumed_sources: list[str]          # grid-forming sources with no identification record
    ride_through_assumed: list[str]
    undeclared: list[str]
    protection_studies: dict[str, ProtectionStudyRecord] = field(default_factory=dict)

    # -- the question ----------------------------------------------------------
    def check(self, state_id: str, phenomenon: Phenomenon | str, load_mva: float | None = None,
              step_pu: float | None = None) -> tuple[Inside, list[str]]:
        """Is a result for this state and phenomenon (at this load, this step) inside the envelope?"""
        ph = Phenomenon(phenomenon)
        out, prov = [], []
        if self.site_wide_fails:
            out.append(f"site-wide gate FAILs must clear first: {'; '.join(self.site_wide_fails[:3])}")
        support, why = self.phenomena[ph]
        if support == Support.EXCLUDED:
            out.append(f"{ph.value} is outside this tier - {why}")
        st = self.states.get(state_id)
        if st is None:
            out.append(f"state {state_id!r} is not enumerated in the Site: the twin says nothing about it")
        else:
            excl = st.excluded.get(ph.value, [])
            passed_study = self.protection_studies.get(state_id)
            if ph == Phenomenon.PROTECTION and passed_study is not None and passed_study.passed:
                # the study computed the fault currents the screening only bounds from ratings
                excl = [x for x in excl if not x.startswith("protection:")]
            if excl:
                out.append(f"the gates FAIL this state for {ph.value}: {'; '.join(excl[:3])}")
            if load_mva is not None and not st.grid_connected:
                if load_mva > st.rating_mva + 1e-9:
                    out.append(f"load {load_mva:g} MVA exceeds the state's source rating {st.rating_mva:g} MVA")
                elif load_mva > st.sustained_mva + 1e-9:
                    prov.append(f"load {load_mva:g} MVA exceeds the sustained sources' {st.sustained_mva:g} MVA: "
                                f"valid only as long as the storage lasts")
            if ph in (Phenomenon.FREQUENCY, Phenomenon.PROTECTION):
                for sid in st.sources:
                    rec = self.identifications.get(sid)
                    if sid in self.assumed_sources:
                        prov.append(f"{sid}: dynamic parameters assumed, not identified")
                    elif rec is not None and not rec.identified:
                        prov.append(f"{sid}: identification verdict {rec.verdict}")
                    elif rec is not None and step_pu is not None and not (
                            rec.step_range_pu[0] - 1e-9 <= abs(step_pu) <= rec.step_range_pu[1] + 1e-9):
                        prov.append(f"{sid}: step {step_pu:g} pu outside the identified "
                                    f"{rec.step_range_pu[0]:g}-{rec.step_range_pu[1]:g} pu - an extrapolation")
            study = self.protection_studies.get(state_id) if ph == Phenomenon.PROTECTION else None
            if study is not None:
                # A study supersedes the screening: its verdict, its models' provenance.
                prov = [w for w in prov if "dynamic parameters" not in w and "identification verdict" not in w
                        and "extrapolation" not in w]
                if not study.passed:
                    out.append(f"the {study.tool} protection study finds this state's protection fails"
                               + (f" ({study.reference})" if study.reference else ""))
                elif not study.models_identified:
                    prov.append(f"studied in {study.tool}"
                                + (f" ({study.reference})" if study.reference else "")
                                + " on assumed converter current limits and ride-through")
            elif ph == Phenomenon.PROTECTION:
                assumed = [s for s in st.sources if s in self.ride_through_assumed]
                if assumed:
                    prov.append(f"current-limit / ride-through assumed for {', '.join(assumed)}")
        studied = ph == Phenomenon.PROTECTION and state_id in self.protection_studies
        if support == Support.PROVISIONAL and not out and not studied:
            prov.append(why)
        if out:
            return Inside.OUTSIDE, out
        if prov:
            return Inside.PROVISIONAL, prov
        return Inside.INSIDE, []

    # -- the artifact ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "site_id": self.site_id,
            "mgtwin_version": self.version,
            "phenomena": {p.value: {"support": s.value, "note": n} for p, (s, n) in self.phenomena.items()},
            "site_wide_fails": self.site_wide_fails,
            "states": {k: {"grid_connected": v.grid_connected, "sources": v.sources,
                           "rating_mva": round(v.rating_mva, 4), "sustained_mva": round(v.sustained_mva, 4),
                           "excluded": v.excluded} for k, v in self.states.items()},
            "identifications": {k: {"verdict": r.verdict, "step_range_pu": list(r.step_range_pu),
                                    "parameters": r.parameters} for k, r in self.identifications.items()},
            "assumed_sources": self.assumed_sources,
            "ride_through_assumed": self.ride_through_assumed,
            "undeclared": self.undeclared,
            "protection_studies": {k: {"tool": r.tool, "passed": r.passed, "models_identified": r.models_identified,
                                       "reference": r.reference} for k, r in self.protection_studies.items()},
        }

    def to_json(self, path=None) -> str:
        txt = json.dumps(self.to_dict(), indent=1)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
        return txt

    def render(self) -> str:
        """Markdown: the envelope as a reviewer reads it."""
        L = [f"# Validity envelope: {self.site_id}", "", f"mgtwin {self.version}", "", "## Phenomena", "",
             "| Phenomenon | Support | Note |", "|---|---|---|"]
        L += [f"| {p.value} | {s.value} | {n} |" for p, (s, n) in self.phenomena.items()]
        if self.site_wide_fails:
            L += ["", "## Site-wide FAILs (everything outside until they clear)", ""]
            L += [f"- {f}" for f in self.site_wide_fails]
        L += ["", "## States", "", "| State | Sources | Rating MVA | Sustained MVA | Outside for |", "|---|---|---|---|---|"]
        for s in self.states.values():
            ex = ", ".join(sorted(s.excluded)) or "-"
            rat = "grid" if s.grid_connected else f"{s.rating_mva:.2f}"
            sus = "grid" if s.grid_connected else f"{s.sustained_mva:.2f}"
            L.append(f"| {s.state_id} | {len(s.sources)} | {rat} | {sus} | {ex} |")
        L += ["", "## Parameters", ""]
        for r in self.identifications.values():
            L.append(f"- {r.source_id}: {r.verdict}, steps {r.step_range_pu[0]:g}-{r.step_range_pu[1]:g} pu")
        if self.assumed_sources:
            L.append(f"- assumed (no identification): {', '.join(self.assumed_sources)}")
        if self.ride_through_assumed:
            L.append(f"- current limit / ride-through assumed: {', '.join(self.ride_through_assumed)}")
        if self.protection_studies:
            L += ["", "## Protection studies", "", "| State | Tool | Verdict | Models | Reference |", "|---|---|---|---|---|"]
            L += [f"| {r.state_id} | {r.tool} | {'passed' if r.passed else 'FAILS'} | "
                  f"{'identified' if r.models_identified else 'assumed'} | {r.reference} |"
                  for r in self.protection_studies.values()]
        if self.undeclared:
            L += ["", "## Undeclared bounds (not invented)", ""]
            L += [f"- {u}" for u in self.undeclared]
        return "\n".join(L) + "\n"


def build_envelope(site: Site, identifications: list[IdentificationRecord] | None = None,
                   report: GateReport | None = None,
                   protection_studies: list[ProtectionStudyRecord] | None = None) -> Envelope:
    """
    Declare the envelope of a Site from its gates, its identification records and the protection
    studies run on its states. A studied state's protection is judged by the study: OUTSIDE if it
    fails, INSIDE if it passed on identified models, PROVISIONAL if on assumed ones; a state no
    study covers keeps the gates' screening, PROVISIONAL at best.
    """
    from . import __version__

    report = report if report is not None else run_all(site)
    ids = {r.source_id: r for r in (identifications or [])}
    unknown = sorted(set(ids) - set(site.source_ids()))
    if unknown:
        raise ValueError(f"identification records for unknown sources {unknown}")
    island_ids = {i.id for i in site.islands}
    studies = {r.state_id: r for r in (protection_studies or [])}
    unknown = sorted(set(studies) - island_ids)
    if unknown:
        raise ValueError(f"protection studies for states the Site does not enumerate: {unknown}")

    site_wide, per_state = [], {}
    for f in report.findings:
        if f.severity != Severity.FAIL:
            continue
        state = str(f.scope).split("/")[0]
        if state in island_ids:
            per_state.setdefault(state, []).append(f)
        elif f.gate in ("units", "protection") or state == site.id:
            site_wide.append(f"{f.gate}: {f.message}")

    srcs = {s.id: s for s in list(site.machines) + list(site.converters)}
    states = {}
    for isl in site.islands:
        here = [srcs[i] for i in isl.energized_source_ids if i in srcs]
        excluded: dict[str, list[str]] = {}
        for f in per_state.get(isl.id, []):
            for ph in GATE_PHENOMENA.get(f.gate, set()):
                excluded.setdefault(ph.value, []).append(f"{f.gate}: {f.message}")
        states[isl.id] = StateEnvelope(
            isl.id, isl.grid_connected, [s.id for s in here],
            sum(s.s_rated_mva for s in here),
            sum(s.s_rated_mva for s in here if not getattr(s, "energy_mwh", None)),
            excluded)

    gfm = [c for c in site.converters if c.control != ConverterControl.GFL] + list(site.machines)
    assumed = sorted(s.id for s in gfm if s.id not in ids)
    rt_assumed = sorted(c.id for c in site.converters if not c.params.ride_through_identified)
    inrush_assumed = any(not t.inrush_identified for t in site.transformers)

    any_assumed = bool(assumed) or any(not r.identified for r in ids.values())
    phenomena = {
        Phenomenon.LOADING: (Support.SUPPORTED, "ratings, not a load flow: a pass is necessary, not sufficient"),
        Phenomenon.FREQUENCY: ((Support.PROVISIONAL, "aggregate RMS model on assumed dynamic parameters")
                               if any_assumed else (Support.SUPPORTED, "aggregate RMS model on identified parameters")),
        Phenomenon.PROTECTION: ((Support.PROVISIONAL, "screening on assumed current limits; settings need a "
                                 "protection study")
                                if rt_assumed else (Support.SUPPORTED, "screening; settings need a protection study"))
                               if not studies else
                               (Support.PROVISIONAL if rt_assumed else Support.SUPPORTED,
                                f"protection studies for {len(studies)} of {len(site.islands)} states; "
                                f"the rest screened only"),
        Phenomenon.GROUNDING: (Support.SUPPORTED, "zero-sequence topology; SLG overvoltage on radial feeders only"),
        Phenomenon.ENERGISATION: ((Support.PROVISIONAL, "static check on assumed inrush multiples")
                                  if inrush_assumed else (Support.SUPPORTED, "static check; waveforms need EMT")),
    }
    phenomena.update({p: (Support.EXCLUDED, why) for p, why in EXCLUDED_PHENOMENA.items()})

    undeclared = ["frequency and voltage ranges over which the dynamic models hold (no source records them)"]
    if not ids:
        undeclared.append("load-step amplitudes: no identification records, so no identified range")
    if any(isl.grid_connected for isl in site.islands):
        undeclared.append("grid strength at the point of interconnection (grid-connected states)")

    return Envelope(site.id, __version__, phenomena, states, site_wide, ids, assumed, rt_assumed, undeclared,
                    studies)
