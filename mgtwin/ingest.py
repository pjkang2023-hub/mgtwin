"""
Stage 1: ingest - a site's tag list to canonical entities.

A SCADA or historian export names its points by a site's own convention
("PCS1.P_MW", "SOFC01_KW", "CB_TIE.POS"). Ingest maps each tag to a canonical
entity (kind, id) and measurement (P, Q, V, I, F, SOC, POS) through rules the
engineer writes for that site - none are built in, because a naming
convention guessed is a mapping invented - and emits a reviewable report:

  * every tag MATCHED, UNMATCHED, AMBIGUOUS (two rules disagree) or DUPLICATE
    (a second tag for the same entity and measurement), with the reason;
  * every unit checked against the measurement's dimension and scaled to the
    canonical unit (MW, Mvar, kV, kA, Hz, pu); a kW tag declared as P is
    scaled, an A tag declared as P is flagged;
  * `compare_with_site` then sets the entities against a composed Site:
    telemetry for equipment the Site lacks, equipment with no telemetry, and
    sources that cannot be identified from operating data (no P or no F, or
    sampled slower than the caller's identification period).

Ingest does not compose topology: tags say what is measured, not what is
connected. Composition (stage 2) remains the engineer's, against the one-line.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

#: Canonical measurements and their dimension.
MEASUREMENTS = {
    "P": "power", "Q": "reactive", "S": "apparent", "V": "voltage", "I": "current",
    "F": "frequency", "SOC": "fraction", "POS": "state", "TEMP": "temperature",
}

#: unit (lower case) -> (dimension, scale to the canonical unit)
UNITS = {
    "w": ("power", 1e-6), "kw": ("power", 1e-3), "mw": ("power", 1.0),
    "var": ("reactive", 1e-6), "kvar": ("reactive", 1e-3), "mvar": ("reactive", 1.0),
    "va": ("apparent", 1e-6), "kva": ("apparent", 1e-3), "mva": ("apparent", 1.0),
    "v": ("voltage", 1e-3), "kv": ("voltage", 1.0),
    "a": ("current", 1e-3), "ka": ("current", 1.0),
    "hz": ("frequency", 1.0),
    "%": ("fraction", 0.01), "pu": ("fraction", 1.0),
    "degc": ("temperature", 1.0), "c": ("temperature", 1.0),
    "": ("state", 1.0), "bool": ("state", 1.0), "state": ("state", 1.0),
}
#: a voltage or current may be given per unit; it is then dimensionless but still that measurement
PU_ALLOWED = {"V", "I", "P", "Q", "S"}

#: Entity kinds, as the Site lists them.
KINDS = {"bus": "buses", "branch": "branches", "transformer": "transformers", "switch": "switches",
         "converter": "converters", "machine": "machines", "load": "loads"}


class TagStatus(str, Enum):
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    AMBIGUOUS = "ambiguous"
    DUPLICATE = "duplicate"


@dataclass
class TagRule:
    """
    A regular expression with a named group `id` (and `meas`, unless the rule
    fixes `measurement`), the entity kind it names, an optional `id_template`
    ("{id}" by default; a template without "{id}" names one entity and needs no
    group) and a map from the site's measurement codes to the canonical ones.
    """

    pattern: str
    kind: str
    meas_map: dict[str, str] = field(default_factory=dict)
    measurement: str | None = None
    id_template: str = "{id}"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"rule {self.pattern!r}: unknown kind {self.kind!r} (one of {sorted(KINDS)})")
        bad = sorted({m for m in list(self.meas_map.values()) + [self.measurement] if m} - set(MEASUREMENTS))
        if bad:
            raise ValueError(f"rule {self.pattern!r}: unknown measurement(s) {bad}")
        self.rx = re.compile(self.pattern)
        if "id" not in self.rx.groupindex and "{id}" in self.id_template:
            raise ValueError(f"rule {self.pattern!r} has no named group 'id' and its id_template needs one")
        if self.measurement is None and "meas" not in self.rx.groupindex:
            raise ValueError(f"rule {self.pattern!r} neither captures 'meas' nor fixes a measurement")

    def apply(self, tag: str):
        """(kind, id, measurement) or None; measurement None when the code is not in the map."""
        m = self.rx.match(tag)
        if not m:
            return None
        meas = self.measurement or self.meas_map.get(m.group("meas"))
        eid = self.id_template.format(id=m.group("id")) if "id" in self.rx.groupindex else self.id_template
        return self.kind, eid, meas, (m.group("meas") if "meas" in self.rx.groupindex else None)


@dataclass
class Tag:
    tag: str
    unit: str = ""
    description: str = ""
    sample_period_s: float | None = None


@dataclass
class TagResult:
    tag: Tag
    status: TagStatus
    kind: str | None = None
    entity_id: str | None = None
    measurement: str | None = None
    scale: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class Entity:
    kind: str
    id: str
    measurements: dict[str, TagResult] = field(default_factory=dict)


@dataclass
class IngestReport:
    results: list[TagResult]
    entities: dict[tuple[str, str], Entity]

    def by_status(self, status: TagStatus) -> list[TagResult]:
        return [r for r in self.results if r.status == status]

    def flagged(self) -> list[TagResult]:
        return [r for r in self.results if r.status != TagStatus.MATCHED or r.notes]

    def to_dict(self) -> dict:
        return {
            "counts": {s.value: len(self.by_status(s)) for s in TagStatus},
            "entities": [{"kind": e.kind, "id": e.id,
                          "measurements": {m: {"tag": r.tag.tag, "unit": r.tag.unit, "scale": r.scale,
                                               "sample_period_s": r.tag.sample_period_s}
                                           for m, r in sorted(e.measurements.items())}}
                         for e in sorted(self.entities.values(), key=lambda e: (e.kind, e.id))],
            "flagged": [{"tag": r.tag.tag, "status": r.status.value, "notes": r.notes} for r in self.flagged()],
        }

    def to_json(self, path=None) -> str:
        txt = json.dumps(self.to_dict(), indent=1)
        if path is not None:
            Path(path).write_text(txt, encoding="utf-8")
        return txt

    def render(self) -> str:
        c = self.to_dict()["counts"]
        L = ["# Tag ingest", "", ", ".join(f"{k} {v}" for k, v in c.items()), "", "## Entities", "",
             "| Kind | Id | Measurements |", "|---|---|---|"]
        for e in sorted(self.entities.values(), key=lambda e: (e.kind, e.id)):
            L.append(f"| {e.kind} | {e.id} | {', '.join(sorted(e.measurements))} |")
        fl = self.flagged()
        if fl:
            L += ["", "## Flagged tags", "", "| Tag | Status | Notes |", "|---|---|---|"]
            L += [f"| {r.tag.tag} | {r.status.value} | {'; '.join(r.notes)} |" for r in fl]
        return "\n".join(L) + "\n"


def load_rules(path: str | Path) -> list[TagRule]:
    """Rules from a JSON list of {pattern, kind, meas_map?, measurement?, id_template?}."""
    return [TagRule(**r) for r in json.loads(Path(path).read_text(encoding="utf-8"))]


def read_tags(path: str | Path) -> list[Tag]:
    """A tag list from CSV with a `tag` column and optional `unit`, `description`, `sample_period_s`."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if rows and "tag" not in rows[0]:
        raise ValueError(f"{path}: no 'tag' column (columns: {sorted(rows[0])})")
    return [Tag(r["tag"].strip(), (r.get("unit") or "").strip(), (r.get("description") or "").strip(),
                float(r["sample_period_s"]) if (r.get("sample_period_s") or "").strip() else None) for r in rows]


def _check_unit(meas: str, unit: str) -> tuple[float | None, list[str]]:
    u = unit.strip().lower().replace(" ", "")
    if u not in UNITS:
        return None, [f"unit {unit!r} not recognised"]
    dim, scale = UNITS[u]
    want = MEASUREMENTS[meas]
    if dim == want:
        return scale, []
    if u == "pu" and meas in PU_ALLOWED:
        return 1.0, ["per unit: the base must be stated before it is used"]
    return None, [f"unit {unit!r} is {dim}, but {meas} is {want}"]


def ingest(tags: list[Tag], rules: list[TagRule]) -> IngestReport:
    """Map tags to canonical entities by the rules; nothing unmatched is guessed."""
    results, entities = [], {}
    for t in tags:
        hits = [h for h in (r.apply(t.tag) for r in rules) if h is not None]
        distinct = {(k, i, m) for k, i, m, _ in hits}
        if not hits:
            results.append(TagResult(t, TagStatus.UNMATCHED, notes=["no rule matches"]))
            continue
        if len(distinct) > 1:
            results.append(TagResult(t, TagStatus.AMBIGUOUS,
                                     notes=[f"rules disagree: {sorted(distinct, key=str)}"]))
            continue
        kind, eid, meas, code = hits[0]
        if meas is None:
            results.append(TagResult(t, TagStatus.UNMATCHED, kind, eid,
                                     notes=[f"measurement code {code!r} is not in the rule's map"]))
            continue
        scale, notes = _check_unit(meas, t.unit)
        ent = entities.setdefault((kind, eid), Entity(kind, eid))
        if meas in ent.measurements:
            results.append(TagResult(t, TagStatus.DUPLICATE, kind, eid, meas, scale,
                                     notes + [f"{kind} {eid} {meas} already from {ent.measurements[meas].tag.tag}"]))
            continue
        res = TagResult(t, TagStatus.MATCHED, kind, eid, meas, scale, notes)
        ent.measurements[meas] = res
        results.append(res)
    return IngestReport(results, entities)


@dataclass
class IngestFinding:
    kind: str
    id: str
    message: str


def compare_with_site(report: IngestReport, site, identification_period_s: float | None = None) -> list[IngestFinding]:
    """
    Set the ingested entities against a composed Site: what the tags name that the Site lacks, what the Site
    holds that no tag observes, and which sources cannot be identified from operating data. A source needs P
    and F; `identification_period_s`, if the caller gives one, is the slowest sampling that still identifies
    its dynamics (rung 4: a historian's 1 s is not; a PMU's is) - no default is assumed.
    """
    out = []
    in_site = {kind: {x.id for x in getattr(site, attr)} for kind, attr in KINDS.items()}
    for (kind, eid) in sorted(report.entities):
        if eid not in in_site[kind]:
            out.append(IngestFinding(kind, eid, "telemetry for equipment the Site does not hold"))
    for kind in ("converter", "machine", "switch", "load"):
        for eid in sorted(in_site[kind]):
            if (kind, eid) not in report.entities:
                out.append(IngestFinding(kind, eid, "in the Site, but no tag observes it"))
    for kind in ("converter", "machine"):
        for eid in sorted(in_site[kind]):
            ent = report.entities.get((kind, eid))
            if ent is None:
                continue
            missing = [m for m in ("P", "F") if m not in ent.measurements]
            if missing:
                out.append(IngestFinding(kind, eid, f"cannot be identified from operating data: no {'/'.join(missing)} tag"))
            elif identification_period_s is not None:
                periods = {m: ent.measurements[m].tag.sample_period_s for m in ("P", "F")}
                slow = [f"{m} at {'an unknown period' if T is None else f'{T:g} s'}" for m, T in periods.items()
                        if T is None or T > identification_period_s]
                if slow:
                    out.append(IngestFinding(kind, eid, f"sampled too slowly to identify (needs <= "
                                             f"{identification_period_s:g} s): {', '.join(slow)}"))
    return out


#: The package exports ingest() under this name, so that `mgtwin.ingest` stays this module.
ingest_tags = ingest
