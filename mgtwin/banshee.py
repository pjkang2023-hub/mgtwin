"""
Compose the Banshee benchmark into a canonical Site.

Sources, in decreasing order of authority:
  1. case_banshee_feeder_all_v5.m  -- machine-readable network (EPHCC release)
  2. F{1,2,3}/Substation relay spreadsheets -- real settings (same release)
  3. Paper-level documentation -- DER placement and ratings (Salcedo et al. 2019,
     J. Eng.; RTDS model description in the same release)

What each source CANNOT provide is recorded rather than papered over:

  * The MATPOWER case is positive-sequence with equal mutuals, machines and
    inverters disconnected, interties open. It carries NO winding
    configurations and NO grounding information. Grounding is therefore left
    UNDECLARED here, and gate_grounding is expected to FAIL on the islanded
    configurations. That failure is the correct output: it states, mechanically,
    that the benchmark's machine-readable artifact is insufficient for
    islanding studies and that the missing facts must come from the one-line.

  * DER control parameters (droop, inertia, current limits) are NOT in the
    release in machine-readable form. They are entered as paper-level
    assumptions with ride_through_identified=False, so gate_model_honesty
    flags every one of them. Again: correct, not cosmetic.

Bus numbering follows the case file: feeder 1 = 1..15, feeder 2 = 20..150
(tens), feeder 3 = 200..1200 (hundreds). Bus 1 is the 13.8 kV substation bus
(slack); bus 13 carries the diesel genset; bus 110 the PV inverter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl

from .matpower_io import parse_matpower_m
from .mdl_io import extract_transformers
from .ontology import (
    Branch,
    GroundingKind,
    GroundingSource,
    Transformer,
    WindingConfig,
    Bus,
    Converter,
    ConverterControl,
    ConverterParams,
    IslandConfig,
    Load,
    OvercurrentRelay,
    Site,
    Switch,
    SynchronousGen,
)
from .units import Base

# ---------------------------------------------------------------------------
# DER placement, from the paper and the RTDS description. Ratings are
# documentation-level facts; control parameters are assumptions and are
# flagged as such via ride_through_identified=False.
# ---------------------------------------------------------------------------

DIESEL_BUS = 13      # ".m" comment: "This is Diesel genset bus"
DIESEL_MVA = 4.0     # RTDS description: "4MW DIESEL GEN"
PV_BUS = 110         # ".m" comment: "PV inverter bus"
PV_MVA = 1.0         # assumed order of magnitude; not in machine-readable data
CHP_BUS = 800        # feeder-3 CHP per paper; 13.8 kV collector bus
CHP_MVA = 2.5        # RTDS description transformer rating near the CHP

FEEDER_HEAD_BUSES = {1: 2, 2: 20, 3: 200}  # first bus downstream of substation


def _feeder_of(bus_id: int) -> int:
    if bus_id >= 200:
        return 3
    if bus_id >= 20:
        return 2
    return 1


def _read_relays(xlsx: Path, feeder: str) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))
    header, out = rows[0], []
    col = {name: i for i, name in enumerate(header) if name}
    for r in rows[1:]:
        if r[0] is None:
            continue
        ct_primary = float(r[col["CT Primary : 5"]])
        pickup_sec = float(r[col["51P TOC Trip Pickup [A]"]]) if "51P TOC Trip Pickup [A]" in col else float(
            r[10]
        )
        out.append(
            dict(
                relay_no=int(r[0]),
                v_ll=float(r[1]),
                ct_ratio=ct_primary / 5.0,
                pickup_primary_a=pickup_sec * (ct_primary / 5.0),
                time_dial=float(r[col.get("51P TOC Time Dial", 12)]),
                feeder=feeder,
            )
        )
    return out


def compose_banshee(
    case_path: str | Path,
    relay_dir: str | Path | None = None,
    mdl_path: str | Path | None = None,
) -> Site:
    mpc = parse_matpower_m(case_path)
    bus_m, branch_m = mpc["bus"], mpc["branch"]
    base = Base(s_base_mva=mpc["baseMVA"], v_base_kv=13.8, f_base_hz=60.0)

    kv_of = {int(b[0]): float(b[9]) for b in bus_m}

    buses = [
        Bus(id=f"B{int(b[0])}", v_nom_kv=float(b[9]), is_poi=(int(b[0]) == 1))
        for b in bus_m
    ]

    loads = [
        Load(
            id=f"LD_B{int(b[0])}",
            bus=f"B{int(b[0])}",
            p_mw=float(b[2]),
            q_mvar=float(b[3]),
        )
        for b in bus_m
        if b[2] > 0 or b[3] != 0
    ]

    # Branch rows: every row becomes a Branch (power-flow-exact, pu on the
    # system base with the ratio absorbed by the differing bus voltage bases).
    # Rows spanning voltage levels ADDITIONALLY become Transformer entities
    # when winding data is available from the OPAL-RT model (mdl_path):
    # the .m carries no winding configurations, the .mdl carries them
    # explicitly ("XFR DYg-30 HIL" = delta primary / grounded-wye secondary).
    # Instance-to-branch pairing is by voltage class; every extracted physical
    # instance is DYg, so the grounding-relevant fact does not depend on
    # per-instance rating assignment, and ratings are class-matched
    # best-effort (they feed no gate).
    branches: list[Branch] = []
    for i, row in enumerate(branch_m):
        f, t = int(row[0]), int(row[1])
        is_tx = kv_of[f] != kv_of[t]
        branches.append(
            Branch(
                id=(f"TX_{f}_{t}" if is_tx else f"LN_{f}_{t}"),
                from_bus=f"B{f}",
                to_bus=f"B{t}",
                r_pu=float(row[2]),
                x_pu=float(row[3]),
                b_pu=float(row[4]),
                s_base_mva=mpc["baseMVA"],
                v_base_kv=13.8,
                rating_mva=float(row[5]) if row[5] > 0 else mpc["baseMVA"],
            )
        )

    transformers: list[Transformer] = []
    grounding: list[GroundingSource] = []
    if mdl_path is not None:
        insts = [
            t for t in extract_transformers(mdl_path) if not t.is_hil_artifact
        ]
        # class -> instances, for best-effort rating assignment
        by_class: dict[tuple[float, float], list] = {}
        for t in insts:
            by_class.setdefault((t.vp_v / 1e3, t.vs_v / 1e3), []).append(t)

        conn_map = {
            "delta": WindingConfig.DELTA,
            "wye_grounded": WindingConfig.WYE_GROUNDED,
        }
        for br in branches:
            if not br.id.startswith("TX_"):
                continue
            f, t = br.from_bus, br.to_bus
            kv_f, kv_t = kv_of[int(f[1:])], kv_of[int(t[1:])]
            hv_bus, lv_bus = (f, t) if kv_f >= kv_t else (t, f)
            hv_kv, lv_kv = max(kv_f, kv_t), min(kv_f, kv_t)
            pool = by_class.get((hv_kv, lv_kv), [])
            inst = pool.pop(0) if pool else None
            transformers.append(
                Transformer(
                    id=f"X{br.id}",
                    hv_bus=hv_bus,
                    lv_bus=lv_bus,
                    s_rated_mva=(inst.kva / 1e3) if inst else 1.0,
                    hv_kv=hv_kv,
                    lv_kv=lv_kv,
                    x_pu=br.x_pu,
                    r_pu=br.r_pu,
                    hv_config=conn_map[inst.hv_connection]
                    if inst
                    else WindingConfig.DELTA,
                    lv_config=conn_map[inst.lv_connection]
                    if inst
                    else WindingConfig.WYE_GROUNDED,
                )
            )
        # The three 50 MVA 115/13.8 kV substation banks are DYg with the Yg on
        # the 13.8 kV side -- but they sit UPSTREAM of the feeder-head
        # breakers, so their grounding is lost the moment a feeder islands.
        grounding.append(
            GroundingSource(
                id="G_SUBSTATION_Yg",
                bus="B1",
                kind=GroundingKind.SOLID,
                available_when_islanded=False,
                notes="50 MVA 115/13.8 kV DYg substation banks (from "
                "BANSHEE.mdl); Yg on the 13.8 kV side, upstream of the "
                "feeder-head breakers.",
            )
        )
        # Machine note: the released synchronous machine blocks provide no
        # zero-sequence path, and no grounding transformer exists anywhere in
        # the model -- so NO islanded MV grounding source is declared. That is
        # the finding, not an omission.

    # Feeder-head breakers (the reconfiguration/islanding points) plus the PCC.
    switches = [Switch(id="SW_PCC", from_bus="B1", to_bus="B1", is_pcc=True)]
    for fdr, head in FEEDER_HEAD_BUSES.items():
        switches.append(
            Switch(id=f"SW_F{fdr}", from_bus="B1", to_bus=f"B{head}")
        )

    # Relays from the release spreadsheets: the feeder-head 51P elements.
    relays: list[OvercurrentRelay] = []
    if relay_dir is not None:
        relay_dir = Path(relay_dir)
        for fdr in (1, 2, 3):
            xlsx = relay_dir / f"F{fdr}_relay_banshee.xlsx"
            if not xlsx.exists():
                continue
            rows = _read_relays(xlsx, f"F{fdr}")
            # Relay #1 in each sheet is the feeder-head device (13.8 kV,
            # CB initially closed).
            head = rows[0]
            relays.append(
                OvercurrentRelay(
                    id=f"R_F{fdr}_HEAD",
                    switch_id=f"SW_F{fdr}",
                    ct_ratio=head["ct_ratio"],
                    pickup_a=head["pickup_primary_a"],
                    curve="SEL-U1",
                    time_dial=head["time_dial"],
                    notes=f"51P from {xlsx.name}, relay #1, primary amperes",
                )
            )

    # DERs. Ratings documented; control parameters assumed and flagged.
    machines = [
        SynchronousGen(
            id="DG_DIESEL",
            bus=f"B{DIESEL_BUS}",
            s_rated_mva=DIESEL_MVA,
            h_sec=1.0,               # assumed
            xd_subtransient_pu=0.15, # assumed
            droop_pu=0.05,           # assumed
            neutral_grounded=False,  # UNKNOWN -- deliberately pessimistic
        ),
        SynchronousGen(
            id="DG_CHP",
            bus=f"B{CHP_BUS}",
            s_rated_mva=CHP_MVA,
            h_sec=1.5,
            xd_subtransient_pu=0.18,
            droop_pu=0.05,
            neutral_grounded=False,
        ),
    ]
    converters = [
        Converter(
            id="PV_F2",
            bus=f"B{PV_BUS}",
            s_rated_mva=PV_MVA,
            control=ConverterControl.GFL,
            params=ConverterParams(
                i_limit_transient_pu=1.2,       # assumed
                ride_through_identified=False,  # and said so
            ),
            vendor="",
            firmware="",
        )
    ]

    # Island configurations: grid-connected, whole-site island, and one island
    # per feeder (interties open in v5, so per-feeder islands are the natural
    # reachable set).
    islands = [
        IslandConfig(
            id="GRID",
            description="Grid-connected, all feeders on utility",
            grid_connected=True,
            energized_source_ids=["DG_DIESEL", "DG_CHP", "PV_F2"],
        ),
        IslandConfig(
            id="ISL_ALL",
            description="Whole site islanded at the PCC",
            open_switches=["SW_PCC"],
            grid_connected=False,
            energized_source_ids=["DG_DIESEL", "DG_CHP", "PV_F2"],
        ),
        IslandConfig(
            id="ISL_F1",
            description="Feeder 1 islanded on the diesel genset",
            open_switches=["SW_F1"],
            grid_connected=False,
            energized_source_ids=["DG_DIESEL"],
        ),
        IslandConfig(
            id="ISL_F2",
            description="Feeder 2 islanded on the PV inverter alone",
            open_switches=["SW_F2"],
            grid_connected=False,
            energized_source_ids=["PV_F2"],
        ),
        IslandConfig(
            id="ISL_F3",
            description="Feeder 3 islanded on the CHP",
            open_switches=["SW_F3"],
            grid_connected=False,
            energized_source_ids=["DG_CHP"],
        ),
    ]

    return Site(
        id="BANSHEE",
        name="Banshee distribution benchmark (EPHCC release, case v5)",
        base=base,
        buses=buses,
        branches=branches,
        transformers=transformers,
        switches=switches,
        relays=relays,
        converters=converters,
        machines=machines,
        loads=loads,
        grounding=grounding,
        islands=islands,
    )
