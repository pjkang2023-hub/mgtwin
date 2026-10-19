"""
Translate a (simple) MATPOWER .m case file to Python dicts.

Scope: handles the subset of MATLAB used by the Banshee case files -- ordered
scalar assignments with arithmetic, complex literals (1i), conj/real/imag/sqrt,
and matrix blocks whose entries are numbers or previously assigned variables.
This is not a MATLAB interpreter and does not try to be one; anything outside
that subset raises rather than guessing.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

_MATRIX_RE = re.compile(r"mpc\.(\w+)\s*=\s*\[(.*?)\];", re.S)
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?);\s*$")

_SAFE = {
    "conj": np.conj,
    "real": np.real,
    "imag": np.imag,
    "sqrt": math.sqrt,
    "pi": math.pi,
    "abs": abs,
}


def _matlab_expr_to_python(expr: str) -> str:
    expr = expr.replace("^", "**")
    # 1i / 2.5i complex literals -> 1j (avoid touching identifiers like Vi)
    expr = re.sub(r"(?<![\w.])((?:\d+\.?\d*(?:[eE][+-]?\d+)?))i\b", r"\1j", expr)
    return expr


def parse_matpower_m(path: str | Path) -> dict:
    """
    Returns {"baseMVA": float, "bus": ndarray, "gen": ndarray, "branch": ndarray,
             "namespace": dict}  (namespace kept for provenance/debugging).
    """
    text = Path(path).read_text()

    # Strip comments. MATLAB has no block comments in these files; everything
    # from % to end of line goes, except we must not clip inside strings (none
    # of these files contain strings with %).
    stripped_lines = []
    for line in text.splitlines():
        cut = line.find("%")
        stripped_lines.append(line if cut < 0 else line[:cut])
    stripped = "\n".join(stripped_lines)

    # Pull out matrix blocks first so their contents don't confuse the
    # assignment pass, and remember their order of appearance.
    matrices_raw: dict[str, str] = {}
    def _grab(m: re.Match) -> str:
        matrices_raw[m.group(1)] = m.group(2)
        return ""
    remainder = _MATRIX_RE.sub(_grab, stripped)

    # Evaluate scalar assignments in order.
    ns: dict[str, object] = {}
    for raw in remainder.splitlines():
        m = _ASSIGN_RE.match(raw)
        if not m:
            continue
        name, expr = m.group(1), m.group(2)
        if name in ("mpc.version",):
            continue
        py = _matlab_expr_to_python(expr)
        try:
            ns[name] = eval(py, {"__builtins__": {}}, {**_SAFE, **ns})
        except Exception as exc:  # noqa: BLE001 -- refuse to guess
            raise ValueError(f"could not evaluate {name!r} = {expr!r}: {exc}") from exc

    # mpc.baseMVA is assigned as a scalar expression, caught above? No: the
    # assignment regex requires a bare identifier, and "mpc.baseMVA" has a dot.
    m = re.search(r"mpc\.baseMVA\s*=\s*(.+?);", stripped)
    if not m:
        raise ValueError("no mpc.baseMVA found")
    base_mva = float(eval(_matlab_expr_to_python(m.group(1)), {"__builtins__": {}}, {**_SAFE, **ns}))

    def eval_matrix(body: str) -> np.ndarray:
        rows = []
        for row in body.split(";"):
            row = row.strip()
            if not row:
                continue
            vals = []
            for tok in row.split():
                py = _matlab_expr_to_python(tok)
                v = eval(py, {"__builtins__": {}}, {**_SAFE, **ns})
                vals.append(float(np.real(v)))
            rows.append(vals)
        width = max(len(r) for r in rows)
        if any(len(r) != width for r in rows):
            raise ValueError("ragged matrix rows")
        return np.array(rows)

    out: dict = {"baseMVA": base_mva, "namespace": ns}
    for key, body in matrices_raw.items():
        out[key] = eval_matrix(body)
    return out


if __name__ == "__main__":
    import sys

    mpc = parse_matpower_m(sys.argv[1])
    print(f"baseMVA = {mpc['baseMVA']}")
    for k in ("bus", "gen", "branch"):
        if k in mpc:
            print(f"{k}: {mpc[k].shape}")
