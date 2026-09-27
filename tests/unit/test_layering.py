"""Import layering of the rrp package (W4, docs/repo_structure_audit.md "Target structure").

contracts -> physics -> bodies -> tasks -> envs -> features -> {teachers, models} -> controllers -> data
-> evaluation -> training -> pipelines -> {orchestration, service, cli, core (the public re-export module, W11)};
research may import anything, nothing imports it.
Within the audit's "teachers/controllers/models" and "training/evaluation" layers the order is fixed as shown
(controllers load models; training runs evaluation rollouts, evaluation never imports training).

Checked over the AST of every real (non-shim) module, including function-level (lazy) imports:
1. no import goes to a higher layer (KNOWN lists the remaining exceptions, each with the phase that removes it);
2. the package graph of the new layout is acyclic (so evaluation does not import training: training -> evaluation only);
3. nothing imports rrp.research;
4. new code imports new paths, never a deprecated shim path (only the modules still PENDING a move may);
5. every shim points where PLANNED says, and the only real modules left in legacy packages are PENDING ones.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

LAYER = {
    "contracts": 0, "physics": 1, "bodies": 2, "tasks": 3, "envs": 4, "features": 5,
    "teachers": 6, "models": 6, "controllers": 6.5, "data": 7, "evaluation": 8, "training": 8.5, "pipelines": 9,
    "orchestration": 10, "cli": 10, "service": 10, "core": 10, "research": 11,
}

# old module -> new module. Moved entries are shims at the old path; the rest are still PENDING (real code at the old path).
PLANNED = {
    # P1 contracts / physics / features
    "rrp.control.psi_contracts": "rrp.contracts.psi",
    "rrp.sim.snapshot_contract": "rrp.physics.snapshot",
    "rrp.data.features": "rrp.features.featurizer",
    "rrp.data.features_multi": "rrp.features.multi",
    "rrp.control.legged_latent": "rrp.features.legged",
    # P2 bodies / envs / teachers / controllers / models
    "rrp.morphology.aloha": "rrp.bodies.aloha",
    "rrp.morphology.catalog": "rrp.bodies.catalog",
    "rrp.morphology.compiler": "rrp.bodies.compiler",
    "rrp.morphology.fixtures": "rrp.bodies.fixtures",
    "rrp.morphology.generators": "rrp.bodies.generators",
    "rrp.morphology.importers": "rrp.bodies.importers",
    "rrp.morphology.surgery": "rrp.bodies.surgery",
    "rrp.morphology.variants": "rrp.bodies.variants",
    "rrp.morphology.legged_catalog": "rrp.evaluation.legged_catalog",
    "rrp.control.ik": "rrp.bodies.ik",
    "rrp.sim.native": "rrp.envs.native",
    "rrp.sim.dual": "rrp.envs.dual",
    "rrp.sim.scenario": "rrp.envs.scenario",
    "rrp.sim.dual_scenarios": "rrp.envs.dual_scenarios",
    "rrp.sim.sensors": "rrp.envs.sensors",
    "rrp.sim.fixtures": "rrp.envs.fixtures",
    "rrp.control.joint_targets": "rrp.envs.joint_targets",
    "rrp.control.teachers": "rrp.teachers.arm",
    "rrp.control.dual_teachers": "rrp.teachers.dual",
    "rrp.control.legged_teachers": "rrp.teachers.legged",
    "rrp.control.dual_validate": "rrp.teachers.dual_validate",
    "rrp.control.functional_composition": "rrp.teachers.functional_composition",
    "rrp.control.latent_realizer": "rrp.controllers.latent_realizer",
    "rrp.policy.runner": "rrp.controllers.policy_runner",
    "rrp.policy.latent_runner": "rrp.controllers.latent_runner",
    "rrp.policy.registry": "rrp.service.policy_registry",
    "rrp.model.attention": "rrp.models.attention",
    "rrp.model.backbone": "rrp.models.backbone",
    "rrp.model.batch": "rrp.models.batch",
    "rrp.model.binding_aug": "rrp.models.binding_aug",
    "rrp.model.codec": "rrp.models.codec",
    "rrp.model.flow": "rrp.models.flow",
    "rrp.model.latent_batch": "rrp.models.latent_batch",
    "rrp.model.latent_probes": "rrp.models.latent_probes",
    "rrp.model.legged_latent": "rrp.models.legged_latent",
    "rrp.model.qa": "rrp.models.qa",
    "rrp.model.semantic_latent": "rrp.models.semantic_latent",
    "rrp.learning.checkpoint": "rrp.models.checkpoint",
    "rrp.learning.critics": "rrp.models.critics",
    # P3 data / training / evaluation
    "rrp.learning.data": "rrp.data.chunks",
    "rrp.learning.packed": "rrp.data.packed",
    "rrp.learning.dual_latent": "rrp.data.dual_latent",
    "rrp.learning.adapt": "rrp.training.adapt",
    "rrp.learning.behavior": "rrp.training.behavior",
    "rrp.learning.branching": "rrp.training.branching",
    "rrp.learning.expo": "rrp.training.expo",
    "rrp.learning.flow_sde": "rrp.training.flow_sde",
    "rrp.learning.grpo": "rrp.training.grpo",
    "rrp.learning.latent_grpo": "rrp.training.latent_grpo",
    "rrp.learning.latent_train": "rrp.training.latent_train",
    "rrp.learning.legged_bc": "rrp.training.legged_bc",
    "rrp.learning.legged_dagger": "rrp.training.legged_dagger",
    "rrp.learning.legged_latent_train": "rrp.training.legged_latent_train",
    "rrp.learning.replay_buffer": "rrp.training.replay_buffer",
    "rrp.learning.rollout": "rrp.training.rollout",
    "rrp.learning.sft": "rrp.training.sft",
    "rrp.learning.swap_alignment": "rrp.training.swap_alignment",
    "rrp.learning.synthetic": "rrp.training.synthetic",
    "rrp.learning.vlm_train": "rrp.training.vlm_train",
    "rrp.evaluation.campaign": "rrp.training.campaign",
    "rrp.evaluation.baseline_campaign": "rrp.training.baseline_campaign",
    "rrp.evaluation.latent_campaign": "rrp.training.latent_campaign",
    # P4 orchestration / cli
    "rrp.ops.broker": "rrp.orchestration.broker",
    "rrp.ops.budget": "rrp.orchestration.budget",
    "rrp.ops.cgroup": "rrp.orchestration.cgroup",
    "rrp.ops.child": "rrp.orchestration.child",
    "rrp.ops.discovery": "rrp.orchestration.discovery",
    "rrp.ops.gpu": "rrp.contracts.workload",
    "rrp.ops.jobs": "rrp.orchestration.jobs",
    "rrp.ops.runtime": "rrp.orchestration.runtime",
    "rrp.ops.telemetry": "rrp.orchestration.telemetry",
    "rrp.ops.watchdog": "rrp.orchestration.watchdog",
    "rrp.cli_ext": "rrp.cli.ext",
    "rrp.cli_ml": "rrp.cli.data",
    "rrp.cli_train": "rrp.cli.train",
    "rrp.cli_latent": "rrp.cli.latent",
    "rrp.cli_dual_latent": "rrp.cli.dual_latent",
    "rrp.cli_adapt": "rrp.cli.adapt",
    # P5 research (one-off diagnostics and modules referenced only from research docs)
    "rrp.learning.legged_t1_diag": "rrp.research.legged_t1_diag",
    "rrp.learning.qa_train": "rrp.research.qa_train",
    "rrp.evaluation.system2_eval": "rrp.research.system2_eval",
    "rrp.evaluation.bc_semantic_edits": "rrp.research.bc_semantic_edits",
    "rrp.evaluation.latent_slice1_report": "rrp.research.latent_slice1_report",
    "rrp.model.system2": "rrp.research.system2",
    # P6 (after the W1 contact track merges; these files are being edited on track/contact)
    "rrp.morphology.contact": "rrp.physics.contact",
    "rrp.morphology.legged": "rrp.bodies.legged",
    "rrp.sim.legged": "rrp.envs.legged",
    "rrp.control.legged_core": "rrp.envs.legged_core",
    "rrp.control.legged_vec": "rrp.envs.legged_vec",
    "rrp.control.legged_tracker": "rrp.envs.legged_tracker",
    "rrp.control.tracker_nets": "rrp.envs.tracker_nets",
    "rrp.control.reward_schedule": "rrp.training.reward_schedule",
    "rrp.control.tracker_training": "rrp.training.tracker_training",
    "rrp.control.tracker_validation": "rrp.evaluation.tracker_validation",
}

# Remaining layer violations (importer, imported) -> the phase that removes them. The test fails when a violation is not
# listed here AND when a listed one no longer occurs (so this list only shrinks).
KNOWN: dict[tuple[str, str], str] = {
}

# Permanent, justified exceptions.
ALLOWED: dict[tuple[str, str], str] = {
    ("rrp.evaluation.latency", "rrp.ops.runtime"):
        "records the broker's active leases next to a latency measurement (lazy, guarded; external-load annotation)",
    ("rrp.evaluation.latency", "rrp.orchestration.runtime"): "same, after the P4 move",
}

SHIM_RE = re.compile(r'_sys\.modules\[__name__\] = _importlib\.import_module\("([\w.]+)"\)')


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


def _shim_target(name):
    p, _ = MODS[name]
    m = SHIM_RE.search(p.read_text())
    return m.group(1) if m else None


SHIMS = {n: t for n in MODS if (t := _shim_target(n))}


def _real(name: str) -> str:
    while name in SHIMS:
        name = SHIMS[name]
    return name


def _in_type_checking(node, parents):
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            return True
    return False


def _imports(name):
    """(imported module as written, lineno) for every rrp import in module `name` (lazy ones included)."""
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
    """Layout package of a module: its new package, or (for a PENDING legacy module) the package it is planned for."""
    mod = _real(mod)
    if mod in PLANNED:
        return PLANNED[mod].split(".")[1]
    parts = mod.split(".")
    if len(parts) >= 2 and parts[1] in LAYER:
        return parts[1]
    return None


def _real_modules():
    return [n for n in MODS if n not in SHIMS and n != "rrp" and not MODS[n][0].name == "__init__.py"]


def _edges():
    edges = []
    for n in _real_modules():
        for m, ln in _imports(n):
            if m in ("rrp",):
                continue
            edges.append((n, m, ln))
    return edges


def test_every_module_has_a_layer():
    missing = [n for n in _real_modules() if _package(n) is None]
    assert not missing, f"modules outside the layout (add them to a layout package or PLANNED): {missing}"


def test_shims_point_where_planned_and_legacy_code_is_pending():
    for old, new in SHIMS.items():
        if old.startswith("rrp.") and old.count(".") >= 1 and old in PLANNED:
            assert PLANNED[old] == new, f"shim {old} -> {new}, PLANNED says {PLANNED[old]}"
        assert new in MODS, f"shim {old} points to a missing module {new}"
    legacy_real = [n for n in _real_modules() if n.split(".")[1] not in LAYER]
    unplanned = legacy_real            # W4 P6 done: legacy packages hold only shims (every move in PLANNED is made)
    assert not unplanned, (f"new code in a legacy package (control/sim/morphology/model/learning/policy/ops): {unplanned}. "
                           "Put new modules in the layout packages (research/tracks/restructure.md), or add the planned "
                           "home to PLANNED if the file must stay put until a move.")


def test_import_layering():
    bad = {}
    for a, b, ln in _edges():
        pa, pb = _package(a), _package(b)
        if pa is None or pb is None or pa == "research":
            continue
        if LAYER[pb] > LAYER[pa]:
            bad[(a, _real(b))] = f"{a}:{ln} ({pa}, layer {LAYER[pa]}) imports {_real(b)} ({pb}, layer {LAYER[pb]})"
    unexpected = {k: v for k, v in bad.items() if k not in KNOWN and k not in ALLOWED}
    stale = [k for k in KNOWN if k not in bad]
    assert not unexpected, "upward imports:\n" + "\n".join(sorted(unexpected.values()))
    assert not stale, f"KNOWN violations that no longer occur (delete them): {stale}"


def test_package_graph_is_acyclic():
    g: dict[str, set[str]] = {}
    for a, b, _ in _edges():
        pa, pb = _package(a), _package(b)
        if pa and pb and pa != pb and pa != "research" and (a, _real(b)) not in KNOWN and (a, _real(b)) not in ALLOWED:
            g.setdefault(pa, set()).add(pb)
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


def test_nothing_imports_research():
    bad = [f"{a}:{ln}" for a, b, ln in _edges() if _real(b).startswith("rrp.research") and not a.startswith("rrp.research")]
    assert not bad, bad


def test_new_code_uses_new_paths():
    pending = {n for n in _real_modules() if n in PLANNED}
    bad = [f"{a}:{ln} imports deprecated {b} (use {_real(b)})" for a, b, ln in _edges()
           if b in SHIMS and a not in pending]
    assert not bad, "\n".join(bad)
