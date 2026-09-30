"""W4 restructure tool (frozen one-off, kept for provenance/resume): move modules with `git mv`, leave an alias shim
at the old path, and rewrite `rrp` import statements in src/ and tests/ to the new paths.

    python research/scripts/2026-09-26/w4_move.py old.mod=new.mod [old2=new2 ...]      # from the repo root

Only import statements are rewritten (never string literals: e.g. "rrp.morphology.generators" is spec-lineage DATA).
Files listed in EXCLUDED (being edited by the W1 contact track) are neither moved nor rewritten.
The shim makes the old module path the SAME module object (sys.modules alias): private names, monkeypatching and
pickles that reference the old path keep working. Modules with a __main__ block forward `python -m old` via runpy.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
EXCLUDED: set[str] = set()      # P6 done (W1 merged, D-101); was the legged files being edited on track/contact
SHIM_MARK = "_sys.modules[__name__] = _importlib.import_module("

SHIM = '''"""Deprecated import path (W4 restructure, docs/strategy.md): moved to `{new}`.

This old path stays importable and is the SAME module object (sys.modules alias), so private names, monkeypatching and
pickles that reference the old path keep working. New code must import `{new}` (tests/unit/test_layering.py).
"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings
{body}'''
BODY_PLAIN = '''
_warnings.warn("{old} is deprecated; import {new}", DeprecationWarning, stacklevel=2)
{mark}"{new}")
'''
BODY_MAIN = '''
if __name__ == "__main__":        # `python -m {old}` keeps working: run the new module as __main__
    import runpy as _runpy
    _warnings.warn("{old} is deprecated; use python -m {new}", DeprecationWarning, stacklevel=1)
    _runpy.run_module("{new}", run_name="__main__", alter_sys=True)
else:
    _warnings.warn("{old} is deprecated; import {new}", DeprecationWarning, stacklevel=2)
    {mark}"{new}")
'''


def mod_path(mod: str) -> Path:
    p = SRC.joinpath(*mod.split(".")).with_suffix(".py")
    return p if p.exists() else SRC.joinpath(*mod.split("."), "__init__.py")


def mod_name(p: Path) -> str:
    parts = list(p.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def is_shim(p: Path) -> bool:
    return SHIM_MARK in p.read_text()


def resolve_rel(cur_mod: str, is_pkg: bool, dots: int, rest: str) -> str:
    base = cur_mod.split(".") if is_pkg else cur_mod.split(".")[:-1]
    base = base[:len(base) - (dots - 1)]
    return ".".join(base + ([rest] if rest else []))


FROM_RE = re.compile(r"^(?P<ind>\s*)from\s+(?P<mod>\.+[\w.]*|rrp[\w.]*)\s+import\s+(?P<rest>.*)$")
IMPORT_RE = re.compile(r"^(?P<ind>\s*)import\s+(?P<mod>rrp[\w.]*)(?P<rest>\s+as\s+\w+)?\s*(?P<cmt>#.*)?$")


def rewrite_file(p: Path, mapping: dict[str, str], own_old: str | None = None) -> bool:
    """Rewrite imports in p. own_old: the module's OLD name if p itself was moved (relative imports resolve there)."""
    cur = own_old or (mod_name(p) if p.is_relative_to(SRC) else "tests.x")
    is_pkg = p.name == "__init__.py"
    lines = p.read_text().split("\n")
    changed = False
    for i, ln in enumerate(lines):
        m = FROM_RE.match(ln)
        if m:
            mod = m["mod"]
            if mod.startswith("."):
                dots = len(mod) - len(mod.lstrip("."))
                absmod = resolve_rel(cur, is_pkg, dots, mod.lstrip("."))
            else:
                absmod = mod
            new = None
            if absmod in mapping:
                new = mapping[absmod]
            elif own_old and mod.startswith(".") and not any(f"{absmod}.{n.strip().split(' ')[0]}" in mapping
                                                             for n in m["rest"].strip("()").split(",")):
                new = absmod                        # moved file: relative import -> absolute (old target)
                new = mapping.get(new, new)
            if new is None:
                # `from pkg import sub [as x], ...` where sub moved
                names = [n.strip() for n in m["rest"].split("#")[0].strip().strip("()").split(",") if n.strip()]
                subs = [(n.split(" as ")[0].strip(), (n.split(" as ")[1].strip() if " as " in n else None)) for n in names]
                moved = [(s, a) for s, a in subs if f"{absmod}.{s}" in mapping]
                if moved:
                    if len(moved) != len(subs) or m["rest"].rstrip().endswith("("):
                        raise SystemExit(f"{p}:{i+1}: mixed/multiline package import, fix by hand: {ln}")
                    out = []
                    for s, a in moved:
                        tgt = mapping[f"{absmod}.{s}"]
                        pkg, base = tgt.rsplit(".", 1)
                        alias = a or s
                        out.append(f"{m['ind']}from {pkg} import {base}" + (f" as {alias}" if alias != base else ""))
                    lines[i] = "\n".join(out)
                    changed = True
                elif mod.startswith(".") and own_old:
                    lines[i] = f"{m['ind']}from {absmod} import {m['rest']}"
                    changed = True
                continue
            lines[i] = f"{m['ind']}from {new} import {m['rest']}"
            changed = True
            continue
        m = IMPORT_RE.match(ln)
        if m and m["mod"] in mapping:
            if not m["rest"]:
                raise SystemExit(f"{p}:{i+1}: bare `import {m['mod']}` (attribute access), fix by hand")
            lines[i] = f"{m['ind']}import {mapping[m['mod']]}{m['rest']}" + (f"  {m['cmt']}" if m["cmt"] else "")
            changed = True
    if changed:
        p.write_text("\n".join(lines))
    return changed


def has_main(p: Path) -> bool:
    return re.search(r'^if __name__ == ["\']__main__["\']', p.read_text(), re.M) is not None


def write_shim(old: str, new: str, main: bool, p: Path):
    body = (BODY_MAIN if main else BODY_PLAIN).format(old=old, new=new, mark=SHIM_MARK)
    p.write_text(SHIM.format(new=new, body=body))


def ensure_pkg(new: str):
    parts = new.split(".")[:-1]
    for k in range(2, len(parts) + 1):
        init = SRC.joinpath(*parts[:k], "__init__.py")
        if not init.exists():
            raise SystemExit(f"missing package {init} (create the skeleton first)")


def move(mapping: dict[str, str]):
    for old, new in mapping.items():
        if old in EXCLUDED:
            raise SystemExit(f"{old} is excluded until W1 merges")
        ensure_pkg(new)
        src, dst = mod_path(old), SRC.joinpath(*new.split(".")).with_suffix(".py")
        if dst.exists():
            raise SystemExit(f"{dst} exists")
        main = has_main(src)
        subprocess.run(["git", "mv", str(src), str(dst)], check=True, cwd=ROOT)
        rewrite_file(dst, mapping, own_old=old)
        write_shim(old, new, main, src)
        subprocess.run(["git", "add", str(src)], check=True, cwd=ROOT)
    moved_new = {SRC.joinpath(*n.split(".")).with_suffix(".py") for n in mapping.values()}
    for p in sorted(list(SRC.rglob("*.py")) + list((ROOT / "tests").rglob("*.py"))):
        if p in moved_new or is_shim(p) or (p.is_relative_to(SRC) and mod_name(p) in EXCLUDED):
            continue
        rewrite_file(p, mapping)


if __name__ == "__main__":
    move(dict(a.split("=", 1) for a in sys.argv[1:]))
