"""
Extract transformer instances from Simulink .mdl files (plain-text format).

Scope: the Banshee OPAL-RT models, where every transformer is a reference
block into transformer_lib with parameters Rating [kVA], Vp, Vs [V], Z [%],
XR, and a SourceBlock name encoding the connection ("XFR DYg-30 HIL" =
delta primary / grounded-wye secondary, 30 degree shift). Anything that is
not that pattern is returned raw for the caller to inspect, not guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_BLOCK_RE = re.compile(r"Block \{(.*?)\n\t\}", re.S)

#: SourceBlock name fragment -> (hv_connection, lv_connection)
CONNECTION_MAP = {
    "DYg": ("delta", "wye_grounded"),
    "YgYg": ("wye_grounded", "wye_grounded"),
    "YgD": ("wye_grounded", "delta"),
    "DD": ("delta", "delta"),
}


@dataclass(frozen=True)
class MdlTransformer:
    source_file: str
    name: str
    block_type: str
    kva: float
    vp_v: float
    vs_v: float
    z_pct: float
    xr: float
    hv_connection: str
    lv_connection: str

    @property
    def is_hil_artifact(self) -> bool:
        """Stub-line coupling transformers exist for HIL decomposition, not
        in the physical plant (Z = 0.1%, Vs = 500 V)."""
        return "stubline" in self.name.lower() or self.vs_v == 500.0


def _param(block: str, key: str) -> str:
    m = re.search(rf'\b{key}\s+"([^"]*)"', block)
    return m.group(1) if m else ""


def extract_transformers(mdl_path: str | Path) -> list[MdlTransformer]:
    text = Path(mdl_path).read_text(encoding="latin-1")
    out: list[MdlTransformer] = []
    for m in _BLOCK_RE.finditer(text):
        b = m.group(1)
        sb = _param(b, "SourceBlock")
        if "transformer_lib/" not in sb:
            continue
        block_type = sb.split("/")[-1]
        conn = None
        for frag, pair in CONNECTION_MAP.items():
            if re.search(rf"\b{frag}\b|XFR {frag}", block_type):
                conn = pair
                break
        if conn is None:
            raise ValueError(
                f"{mdl_path}: transformer block type {block_type!r} has no "
                f"known connection mapping; refusing to guess"
            )
        out.append(
            MdlTransformer(
                source_file=str(Path(mdl_path).name),
                name=_param(b, "Name").replace("\\n", " "),
                block_type=block_type,
                kva=float(_param(b, "Rating")),
                vp_v=float(_param(b, "Vp")),
                vs_v=float(_param(b, "Vs")),
                z_pct=float(_param(b, "Z")),
                xr=float(_param(b, "XR")),
                hv_connection=conn[0],
                lv_connection=conn[1],
            )
        )
    return out
