"""RL: the last private step loops are gone. Every tick of collect / ladder / latent_eval / hooks / grasp_rig runs
through `harness.rollout` (nested rollouts for settle and hold stretches), the ladder's budget is `TaskSpec.max_steps`,
the prev-action input is a hook (no session monkeypatch), and the ladder CLI reads sealed targets from one place."""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "rrp" / "harness"
OWNED = ["data/collect.py", "eval/ladder.py", "eval/ladder_cli.py", "eval/latent_eval.py", "eval/hooks.py",
         "eval/grasp_rig.py"]
# The one tick an Env owns: `_RigEnv.step` (the bare-model bench env is the Env, so `rollout` drives it).
ALLOWED = {("eval/grasp_rig.py", "mj_step", "_RigEnv.step")}


def _stepping_calls(rel):
    tree = ast.parse((SRC / rel).read_text())
    out = []

    def visit(node, scope):
        for ch in ast.iter_child_nodes(node):
            sc = scope + [ch.name] if isinstance(ch, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) else scope
            if isinstance(ch, ast.Call):
                f = ch.func
                name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
                if name in ("step", "mj_step"):
                    out.append((rel, name, ".".join(sc[-2:]) if len(sc) > 1 else ".".join(sc)))
            visit(ch, sc)
    visit(tree, [])
    return out


@pytest.mark.parametrize("rel", OWNED)
def test_no_direct_step_calls(rel):
    bad = [c for c in _stepping_calls(rel) if c not in ALLOWED]
    assert not bad, f"private step call(s) outside harness.rollout: {bad}"


def test_grasp_rig_steps_only_in_its_env():
    assert [c for c in _stepping_calls("eval/grasp_rig.py")] == [("eval/grasp_rig.py", "mj_step", "_RigEnv.step")]


def test_install_prev_action_is_gone():
    import rrp.harness.eval.ladder as L
    assert not hasattr(L, "install_prev_action")
    import inspect
    from rrp.harness.eval.latent_eval import disturbance_test
    assert "session_hook" not in inspect.signature(disturbance_test).parameters
    assert "prev_action" in inspect.signature(disturbance_test).parameters


def test_ladder_task_carries_max_steps():
    from rrp.harness.eval.ladder import LadderConfig, _ladder_task
    t = _ladder_task(LadderConfig(route="teacher", robot="ur5e_pg2", seeds=(3,), max_steps=37))
    assert t.max_steps == 37 and "dropped_off_table" in t.failure_reasons


@pytest.mark.parametrize("robot", ["xarm7_pg2", "xarm7_tf3", "panda_tf3", "gen3_pg2"])
def test_ladder_cli_refuses_sealed_targets(robot):
    from rrp.harness.eval.ladder_cli import main
    with pytest.raises(SystemExit, match="target bodies"):
        main(["--route", "teacher", "--robot", robot, "--n", "1", "--seed-start", "3000000", "--tag", "x",
              "--out", "/tmp/never"])


def test_ladder_cli_uses_the_one_sealed_predicate():
    src = (SRC / "eval" / "ladder_cli.py").read_text()
    assert "is_sealed_target" in src and '"xarm7_pg2"' not in src and "is_armdiv_sealed" not in src
