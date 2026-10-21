"""
mgtwin -- staged, gated authoring of microgrid digital twins.

Pipeline stages, each emitting a reviewable artifact:

    1. ingest      tags -> canonical entities        ingest.py
    2. compose     canonical entities -> Site        ontology.py
    3. verify      Site -> GateReport                gates.py
    4. identify    response data -> parameters + honest confidence   identify.py
    5. test        Site + TestSuite -> pass/fail      testspec.py
    6. envelope    declared validity envelope         envelope.py

What this scaffold deliberately does NOT do: write its own solvers. Composition
targets OpenDSS / pandapower for unbalanced QSS and an external EMT tool for
faults and inrush. The RMS model in models.py exists only to exercise the
identification machinery against a case with known truth.
"""

from .units import Base, PowerSign, AngleRef
from .ontology import (
    Bus,
    Branch,
    Converter,
    ConverterControl,
    ConverterParams,
    GroundingKind,
    GroundingSource,
    IslandConfig,
    Load,
    LoadModel,
    OvercurrentRelay,
    ProtectionFunction,
    Relay,
    Site,
    Switch,
    SynchronousGen,
    Transformer,
    WindingConfig,
)
from .gates import (
    Finding,
    GateReport,
    Severity,
    gate_grounding,
    gate_model_honesty,
    gate_protection,
    gate_units,
    gate_energisation,
    gate_forming,
    gate_loading,
    run_all,
)
from .ingest import IngestReport, Tag, TagRule, TagStatus, compare_with_site, ingest_tags, load_rules, read_tags
from .envelope import (Envelope, IdentificationRecord, Inside, Phenomenon, ProtectionStudyRecord, Support,
                       build_envelope, load_protection_studies)
from .identify import (
    IdentificationResult,
    MultiIdentificationResult,
    Verdict,
    identify,
    identify_multi,
)
from .models import load_step, simulate_gfm
from .testspec import CoreFunction, TestSuite, available_suites, load_suite
from .unbalanced import LineSegment, RadialFeeder, TransformerDGY, TransformerGYGY
from .fault import SLGResult, slg_fault, zigzag_z0_pu
from .runner import run_suite
from .testrun import CaseStatus, CaseVerdict, ParamProvenance, RunReport

__version__ = "0.6.10"

__all__ = [
    "IngestReport",
    "Tag",
    "TagRule",
    "TagStatus",
    "compare_with_site",
    "ingest_tags",
    "load_rules",
    "read_tags",
    "Envelope",
    "IdentificationRecord",
    "Inside",
    "Phenomenon",
    "Support",
    "build_envelope",
    "gate_forming",
    "gate_loading",
    "Base",
    "PowerSign",
    "AngleRef",
    "Bus",
    "Branch",
    "Converter",
    "ConverterControl",
    "ConverterParams",
    "GroundingKind",
    "GroundingSource",
    "IslandConfig",
    "Load",
    "LoadModel",
    "OvercurrentRelay",
    "ProtectionFunction",
    "Relay",
    "ProtectionStudyRecord",
    "load_protection_studies",
    "Site",
    "Switch",
    "SynchronousGen",
    "Transformer",
    "WindingConfig",
    "Finding",
    "GateReport",
    "Severity",
    "gate_units",
    "gate_energisation",
    "gate_grounding",
    "gate_protection",
    "gate_model_honesty",
    "run_all",
    "identify",
    "identify_multi",
    "IdentificationResult",
    "MultiIdentificationResult",
    "Verdict",
    "simulate_gfm",
    "load_step",
    "load_suite",
    "LineSegment",
    "RadialFeeder",
    "TransformerDGY",
    "TransformerGYGY",
    "SLGResult",
    "slg_fault",
    "zigzag_z0_pu",
    "run_suite",
    "RunReport",
    "CaseStatus",
    "CaseVerdict",
    "ParamProvenance",
    "available_suites",
    "TestSuite",
    "CoreFunction",
    "__version__",
]
