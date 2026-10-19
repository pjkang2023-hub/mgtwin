"""
Test definitions as versioned data.

Why data and not code
---------------------
P2030.8 is under revision: the IEEE SA project page shows an Active PAR approved
2026-03-26 superseding 2030.8-2018. A harness that hard-codes the 2018 test
taxonomy rots the day the revision publishes. Test cases therefore live in YAML,
tagged with the edition they belong to, and multiple editions coexist so a
customer can be screened against the edition their contract names while
development tracks the revision.

Copyright
---------
No normative text from any IEEE standard is reproduced here. The `clause` field
is deliberately empty in the shipped specs and must be populated by the user
from their own licensed copy. Descriptions are original summaries written for
this harness, not paraphrases of standard text, and they are not a substitute
for reading the standard.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

SPEC_DIR = Path(__file__).parent / "specs"


class CoreFunction(str, Enum):
    """Controller function families a microgrid controller test suite covers."""

    DISPATCH = "dispatch"
    PLANNED_ISLANDING = "planned_islanding"
    UNPLANNED_ISLANDING = "unplanned_islanding"
    RECONNECTION = "reconnection"
    BLACK_START = "black_start"
    STEADY_STATE = "steady_state"
    POWER_QUALITY = "power_quality"
    PROTECTION = "protection"


class Criterion(BaseModel):
    """
    One pass/fail measurement.

    `limit` is intentionally optional: many limits are project-specific or come
    from the interconnection agreement rather than the standard, and inventing a
    number would be worse than leaving it to be filled in.
    """

    metric: str
    unit: str
    comparator: str = Field(pattern=r"^(<=|>=|<|>|within)$")
    limit: float | None = None
    limit_source: str = Field(
        "project", description="standard | interconnection_agreement | project"
    )
    notes: str = ""


class TestCase(BaseModel):
    id: str
    function: CoreFunction
    title: str
    edition: str
    clause: str = Field("", description="Fill from your licensed copy of the standard")
    initial_condition: str
    initiating_event: str
    measurements: list[str] = Field(default_factory=list)
    criteria: list[Criterion] = Field(default_factory=list)
    requires_island_config: bool = True
    requires_emt: bool = False
    notes: str = ""


class TestSuite(BaseModel):
    edition: str
    source: str
    disclaimer: str
    cases: list[TestCase]

    def by_function(self, fn: CoreFunction) -> list[TestCase]:
        return [c for c in self.cases if c.function == fn]

    def emt_cases(self) -> list[TestCase]:
        return [c for c in self.cases if c.requires_emt]


def load_suite(name: str = "ieee2030_8_2018") -> TestSuite:
    path = SPEC_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in SPEC_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"no spec {name!r}; available: {available}")
    with path.open() as fh:
        return TestSuite(**yaml.safe_load(fh))


def available_suites() -> list[str]:
    return sorted(p.stem for p in SPEC_DIR.glob("*.yaml"))
