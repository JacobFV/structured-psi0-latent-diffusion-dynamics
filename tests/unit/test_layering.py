"""Import layering of the rrp package (docs/architecture.md section 1).

Checked over the AST of every module, including function-level (lazy) imports:
1. every module belongs to a layer package;
2. no import goes to a higher layer (ALLOWED lists the justified exceptions);
3. the sub-package graph is acyclic;
4. nothing but the CLI imports rrp.viz;
5. no shim/alias modules (D-140: moved names are imported from their new path everywhere).
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

LAYER = {
    "core": 0, "ops": 0.5, "bodies": 1, "tasks": 2, "envs": 3, "policies": 4, "harness": 5, "viz": 6, "cli": 7,
}

# Permanent, justified exceptions (importer, imported) -> reason.
ALLOWED: dict[tuple[str, str], str] = {
    ("rrp.harness.eval.latency", "rrp.ops.runtime"):
        "records the broker's active leases next to a latency measurement (lazy, guarded; external-load annotation)",
    ("rrp.policies.teachers.functional_composition", "rrp.harness.eval.dual_teacher_quality"):
        "a suite experiment script (`rrp suite composition`, registered by path in cli/tools.py) whose teacher-intervention "
        "episodes run on harness.rollout through dual_teacher_quality.run_teacher_episode (lazy, inside run())",
    ("rrp.harness.dag", "rrp.harness.pipelines.base"):
        "plan_dag loads the family/stage registry (`_load_families`, lazy, inside plan_dag) because the RunConfigs it builds "
        "validate family and stage names against it; the library entry must not depend on what the caller imported",
}


def _modules():
    out = {}
    for p in SRC.joinpath("rrp").rglob("*.py"):
        parts = list(p.relative_to(SRC).with_suffix("").parts)
        is_pkg = parts[-1] == "__init__"
        if is_pkg:
            parts = parts[:-1]
        out[".".join(parts)] = (p, is_pkg)
    return out


MODS = _modules()


def _in_type_checking(node, parents):
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            return True
    return False


def _imports(name):
    """(imported module, lineno) for every rrp import in module `name` (lazy ones included)."""
    p, is_pkg = MODS[name]
    tree = ast.parse(p.read_text())
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    out = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)) and _in_type_checking(n, parents):
            continue
        if isinstance(n, ast.Import):
            out += [(a.name, n.lineno) for a in n.names if a.name.startswith("rrp")]
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            if n.level:
                base = name.split(".") if is_pkg else name.split(".")[:-1]
                base = base[:len(base) - (n.level - 1)]
                mod = ".".join(base + ([mod] if mod else []))
            if not mod.startswith("rrp"):
                continue
            for a in n.names:
                sub = f"{mod}.{a.name}"
                out.append((sub if sub in MODS else mod, n.lineno))
    return out


def _package(mod: str) -> str | None:
    parts = mod.split(".")
    return parts[1] if len(parts) >= 2 and parts[1] in LAYER else None


def _node(mod: str) -> str:
    """Cycle-check node: a sub-package (e.g. rrp.policies.nets) as a whole; the interface module `<layer>.base` on its
    own (it sits below everything in its layer); every other flat module of a layer package grouped with its layer."""
    parts = mod.split(".")
    if len(parts) >= 3 and MODS.get(".".join(parts[:3]), (None, False))[1]:
        return ".".join(parts[:3])
    if len(parts) == 3 and parts[2] == "base":
        return mod
    return ".".join(parts[:2])


def _edges():
    return [(n, m, ln) for n in MODS if n != "rrp" for m, ln in _imports(n) if m != "rrp"]


def test_every_module_has_a_layer():
    missing = [n for n in MODS if n != "rrp" and _package(n) is None]
    assert not missing, f"modules outside the layer packages: {missing}"


def test_import_layering():
    bad = {}
    for a, b, ln in _edges():
        pa, pb = _package(a), _package(b)
        if pa and pb and LAYER[pb] > LAYER[pa] and (a, b) not in ALLOWED:
            bad[(a, b)] = f"{a}:{ln} ({pa}, layer {LAYER[pa]}) imports {b} ({pb}, layer {LAYER[pb]})"
    assert not bad, "upward imports:\n" + "\n".join(sorted(bad.values()))
    stale = [k for k in ALLOWED if k not in {(a, b) for a, b, _ in _edges()}]
    assert not stale, f"ALLOWED exceptions that no longer occur (delete them): {stale}"


def test_package_graph_is_acyclic():
    """Over sub-packages (rrp.policies.nets, rrp.harness.eval, rrp.envs.mujoco ...) and flat layer modules."""
    g: dict[str, set[str]] = {}
    for a, b, _ in _edges():
        na, nb = _node(a), _node(b)
        if na != nb and (a, b) not in ALLOWED:
            g.setdefault(na, set()).add(nb)
    seen, stack, cycles = set(), [], []

    def visit(v):
        if v in stack:
            cycles.append(stack[stack.index(v):] + [v])
            return
        if v in seen:
            return
        seen.add(v)
        stack.append(v)
        for w in sorted(g.get(v, ())):
            visit(w)
        stack.pop()

    for v in sorted(g):
        visit(v)
    assert not cycles, f"package import cycles: {cycles}"


def test_nothing_imports_viz():
    bad = [f"{a}:{ln}" for a, b, ln in _edges()
           if b.startswith("rrp.viz") and not a.startswith(("rrp.viz", "rrp.cli"))]
    assert not bad, bad


def test_no_shim_modules():
    bad = [n for n, (p, _) in MODS.items()
           if "sys.modules[__name__]" in (t := p.read_text()) or ("DeprecationWarning" in t and "deprecated" in t.lower()
                                                                  and "import_module" in t)]
    assert not bad, f"shim/alias modules (D-140: import the new path instead): {bad}"
