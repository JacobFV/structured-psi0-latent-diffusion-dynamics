"""D-146 round-2 X1: one hooks module, one file digest, one Wilson, the renamed online-episode driver, and rollout's
check that a failure reason is inside the vocabulary its task declares."""
import ast
import importlib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "rrp"


def test_one_hooks_module_beside_rollout():
    from rrp.harness import hooks
    assert (SRC / "harness" / "hooks.py").is_file() and not (SRC / "harness" / "eval" / "hooks.py").exists()
    assert {"feasibility", "settle", "end_when", "recorder"} <= set(hooks.HOOKS) and {"session", "arm", "dual"} == set(hooks.TASK_HOOKS)
    for rel in ("harness/eval/evaluate.py", "harness/data/collect.py"):
        tree = ast.parse((SRC / rel).read_text())
        assigned = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        assert not ({"HOOKS", "TASK_HOOKS"} & assigned) and not ({"_Hold", "_TeacherEnd"} & classes), rel


def test_online_episode_driver_is_not_named_rollout():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rrp.harness.train.rollout")
    assert importlib.import_module("rrp.harness.train.online_episodes").drive


def test_one_file_digest(tmp_path):
    import hashlib
    from rrp.core.provenance import file_digest
    p = tmp_path / "f.bin"
    p.write_bytes(b"x" * 3_000_000)
    full = hashlib.sha256(b"x" * 3_000_000).hexdigest()
    assert file_digest(p, length=None) == full and file_digest(p) == full[:16] and file_digest(p, length=8) == full[:8]
    # no other chunked file hash in the tree (in-memory array hashes are not file digests)
    offenders = [f"{f.relative_to(SRC)}:{n.lineno}" for f in SRC.rglob("*.py") if f != SRC / "core" / "provenance.py"
                 for n in ast.walk(ast.parse(f.read_text()))
                 if isinstance(n, ast.Call) and ast.unparse(n.func) == "iter" and n.args
                 and isinstance(n.args[0], ast.Lambda) and "read" in ast.unparse(n.args[0])]
    assert not offenders, f"chunked file hashes outside core.provenance.file_digest: {offenders}"


def test_failure_reason_outside_the_declared_vocabulary_is_refused():
    from rrp.harness.hooks import EndWhen
    from rrp.harness.rollout import UndeclaredFailureReason, check_failure_reason
    from rrp.tasks.spec import Judgement, TaskSpec
    task = TaskSpec("t", {"x": {}}, 1.0, lambda env, t, T: Judgement(False), failure_reasons=("fell", "wrong_value"))
    for ok in (None, "fell", "timeout", "physics_divergence", "wrong_value:cell_3"):      # rollout's own reasons are declared
        check_failure_reason(ok, task)
    with pytest.raises(UndeclaredFailureReason, match="'tripped'"):
        check_failure_reason("tripped", task)
    with pytest.raises(UndeclaredFailureReason):
        check_failure_reason("hook_end", task)
    check_failure_reason("hook_end", task, [EndWhen(lambda i, e: True)])                  # a hook declares its own reason
    check_failure_reason("tripped", task, [EndWhen(lambda i, e: True, reason="tripped")])
