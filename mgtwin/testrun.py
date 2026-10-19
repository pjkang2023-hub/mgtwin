"""
Test result artifacts.

The runner's deliverable is the thing a customer eventually signs, so its
honesty rules are stricter than anywhere else in the pipeline:

1. A criterion with no numeric limit is UNEVALUATED, never pass/fail. The
   limits live in the interconnection agreement or the project spec; a
   harness that invents them is manufacturing compliance. Measured values
   are still reported -- the engineer can compare against the real limit
   the moment it exists.

2. A case whose prerequisites fail (an ungrounded zone, say) is BLOCKED,
   with the gate findings attached. Running it anyway would produce numbers
   with no meaning.

3. A case whose physics exceeds the available model tier (transformer
   inrush, fault coordination, harmonics) is DEFERRED with the tier named.
   An RMS harness that quietly "answers" an EMT question is lying twice.

4. Every executed case records the PROVENANCE of the dynamic parameters it
   used: identified from response data, or assumed. A PASS on assumed
   parameters is a conditional statement, and the artifact says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .gates import Finding
from .testspec import Criterion, TestCase


class CaseStatus(str, Enum):
    EXECUTED = "executed"
    BLOCKED = "blocked"      # prerequisites failed (gate findings attached)
    DEFERRED = "deferred"    # exceeds available model tier (reason attached)


class CaseVerdict(str, Enum):
    PASS = "pass"                    # every evaluated criterion passed
    FAIL = "fail"                    # at least one evaluated criterion failed
    INDETERMINATE = "indeterminate"  # measured, but no limits to compare
    NONE = "none"                    # not executed


class ParamProvenance(str, Enum):
    IDENTIFIED = "identified"  # from measured response, via identify()
    ASSUMED = "assumed"        # datasheet, typical value, or guess


@dataclass
class MetricValue:
    metric: str
    value: float
    unit: str
    note: str = ""


@dataclass
class CriterionResult:
    criterion: Criterion
    measured: float | None
    evaluated: bool
    passed: bool | None  # None when not evaluated

    def render(self) -> str:
        c = self.criterion
        meas = f"{self.measured:.4g} {c.unit}" if self.measured is not None else "-"
        if not self.evaluated:
            lim = (
                f"no limit set (source: {c.limit_source})"
                if c.limit is None
                else "not measured"
            )
            return f"      {c.metric:<32} measured {meas:<16} UNEVALUATED -- {lim}"
        mark = "PASS" if self.passed else "FAIL"
        return (
            f"      {c.metric:<32} measured {meas:<16} "
            f"{mark} vs {c.comparator} {c.limit:g} {c.unit} ({c.limit_source})"
        )


@dataclass
class TestResult:
    case: TestCase
    island_id: str
    status: CaseStatus
    verdict: CaseVerdict = CaseVerdict.NONE
    metrics: list[MetricValue] = field(default_factory=list)
    criteria: list[CriterionResult] = field(default_factory=list)
    blocking_findings: list[Finding] = field(default_factory=list)
    defer_reason: str = ""
    provenance: ParamProvenance = ParamProvenance.ASSUMED
    provenance_detail: str = ""
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        head = f"  {self.case.id:<10} [{self.island_id}] {self.case.title}"
        lines = [head]
        if self.status == CaseStatus.BLOCKED:
            lines.append(f"    status: BLOCKED -- prerequisites failed")
            for f in self.blocking_findings:
                lines.append(f"      gate {f.gate} [{f.scope}]: {f.message}")
            return "\n".join(lines)
        if self.status == CaseStatus.DEFERRED:
            lines.append(f"    status: DEFERRED -- {self.defer_reason}")
            return "\n".join(lines)

        lines.append(
            f"    status: EXECUTED   verdict: {self.verdict.value.upper()}"
            f"   parameters: {self.provenance.value}"
            + (f" ({self.provenance_detail})" if self.provenance_detail else "")
        )
        if self.provenance == ParamProvenance.ASSUMED and self.verdict == CaseVerdict.PASS:
            lines.append(
                "    NOTE: PASS is conditional on assumed dynamic parameters; "
                "identify from response data before relying on it."
            )
        for m in self.metrics:
            note = f"   ({m.note})" if m.note else ""
            lines.append(f"      {m.metric:<32} {m.value:.4g} {m.unit}{note}")
        if self.criteria:
            lines.append("    criteria:")
            for cr in self.criteria:
                lines.append(cr.render())
        for n in self.notes:
            lines.append(f"    note: {n}")
        return "\n".join(lines)


def judge(criteria_results: list[CriterionResult]) -> CaseVerdict:
    evaluated = [c for c in criteria_results if c.evaluated]
    if not evaluated:
        return CaseVerdict.INDETERMINATE
    if any(c.passed is False for c in evaluated):
        return CaseVerdict.FAIL
    return CaseVerdict.PASS


def evaluate_criterion(c: Criterion, measured: float | None) -> CriterionResult:
    if measured is None or c.limit is None:
        return CriterionResult(c, measured, evaluated=False, passed=None)
    ok = {
        "<=": measured <= c.limit,
        "<": measured < c.limit,
        ">=": measured >= c.limit,
        ">": measured > c.limit,
        "within": abs(measured) <= c.limit,
    }[c.comparator]
    return CriterionResult(c, measured, evaluated=True, passed=ok)


@dataclass
class RunReport:
    suite_edition: str
    site_id: str
    results: list[TestResult] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out = {"executed": 0, "blocked": 0, "deferred": 0,
               "pass": 0, "fail": 0, "indeterminate": 0}
        for r in self.results:
            out[r.status.value] += 1
            if r.status == CaseStatus.EXECUTED:
                out[r.verdict.value] += 1
        return out

    def render(self) -> str:
        lines = [
            f"Test run: suite edition {self.suite_edition} on site {self.site_id}",
            "-" * 78,
        ]
        for r in self.results:
            lines.append(r.render())
            lines.append("")
        c = self.counts()
        lines.append(
            f"summary: {c['executed']} executed "
            f"({c['pass']} pass, {c['fail']} fail, "
            f"{c['indeterminate']} indeterminate), "
            f"{c['blocked']} blocked, {c['deferred']} deferred"
        )
        return "\n".join(lines)
