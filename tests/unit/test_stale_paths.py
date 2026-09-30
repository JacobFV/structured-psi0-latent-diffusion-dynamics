"""No live file names a module path that D-140 / S2 moved (docs/architecture.md section 10 table; D-146 F0).

The removed top-level packages are `rrp.{contracts,controllers,data,evaluation,features,models,orchestration,physics,
pipelines,teachers,training}`; their new homes are listed in `.old/docs/architecture_s7-10.md` section 10. Recorded
version-string constants (`rrp.evaluation.gates/v1`, ...) keep their names on purpose (D-146 item 7) and the pickle
remap of pre-D-140 datasets names `rrp.data.features` on purpose; both are allowed. Files owned by another wave-0
readiness unit are listed in EXEMPT until that unit and X1 have merged (X1 deletes the list)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIVE = ("src", "tests", "recipes", "ops/bin")
REMOVED = re.compile(r"\brrp\.(?:contracts|controllers|data|evaluation|features|models|orchestration|physics|pipelines|"
                     r"teachers|training)\b[.\w]*")
RECORDED_VERSION = re.compile(r"\brrp\.[\w.]+/v\d")            # e.g. "rrp.evaluation.gates/v1 (D-112)"
LEGACY_PICKLE = ("rrp.data.features",)                          # harness/data/collect.py LEGACY_PICKLE_MODULES, test_golden
SKIP_PARTS = {"__pycache__"}
GOLDENS = "tests/data/"                                         # recorded goldens keep their strings
EXEMPT = {                                                      # owned by another wave-0 unit (readiness.md); X1 removes
    "src/rrp/" + p for p in """
    policies/relations/base.py policies/relations/ops.py policies/relations/catalog.py policies/nets/batch.py
    policies/nets/probes.py policies/nets/checkpoint.py policies/nets/flow.py harness/data/relgen/__init__.py cli/tools.py
    tasks/spec.py harness/eval/evaluate.py harness/rollout.py cli/harness.py policies/teachers/__init__.py envs/base.py
    core/runconfig.py core/provenance.py harness/dag.py harness/pipelines/base.py cli/dag.py policies/features/kinfeat.py
    viz/export/relations.py policies/psi0/nets.py policies/pointer.py harness/train/pointer.py
    bodies/legged.py bodies/humanoid_gen.py bodies/importers.py harness/pipelines/legged.py harness/train/legged_bc.py
    harness/train/legged_dagger.py harness/pipelines/dual.py policies/teachers/dual_coord.py policies/teachers/dual_smooth.py
    harness/data/collect_dual.py harness/data/dual_pairs.py harness/data/dual_quality.py harness/eval/ladder.py
    harness/eval/ladder_cli.py harness/eval/hooks.py viz/record.py""".split()
}


def stale_lines(text: str) -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if RECORDED_VERSION.search(line):
            line = RECORDED_VERSION.sub("", line)
        for m in REMOVED.finditer(line):
            if not any(m.group(0).startswith(a) for a in LEGACY_PICKLE):
                out.append((i, m.group(0)))
    return out


def live_files():
    for top in LIVE:
        for p in sorted((ROOT / top).rglob("*")):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and not (SKIP_PARTS & set(p.parts)) and not rel.startswith(GOLDENS):
                yield p


def test_scanner_flags_removed_prefixes_and_allows_recorded_constants():
    assert stale_lines("see rrp.training.latent_train and rrp.evaluation.ladder.run_ladder") == [
        (1, "rrp.training.latent_train"), (1, "rrp.evaluation.ladder.run_ladder")]
    assert stale_lines('GATES_VERSION = "rrp.evaluation.gates/v1 (D-112)"') == []
    assert stale_lines("LEGACY = {'rrp.data.features': 1}") == []
    assert stale_lines("rrp.harness.train.latent_train, rrp.policies.teachers.dual, rrp.core.provenance") == []


def test_no_removed_module_paths_in_live_trees():
    bad = []
    for p in live_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel in EXEMPT or rel == "tests/unit/test_stale_paths.py" or p.suffix in (".pyc", ".pt", ".npz", ".png"):
            continue
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        bad += [f"{rel}:{n}: {s}" for n, s in stale_lines(text)]
    assert not bad, "removed module paths (see docs section 10 table):\n" + "\n".join(bad[:40])
