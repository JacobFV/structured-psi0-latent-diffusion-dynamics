"""One stepping site (D-146 round-2 X1): the episode loop is `harness/rollout.py`; nothing else ticks an env or the
simulator. Scans `src/rrp` for `.step(` and `mj_step(` calls outside `harness/rollout.py` and the env modules (`envs/`),
ignoring (a) the body of an `Env.step` implementation (that method IS an env module, wherever it is defined) and
(b) optimizer / scheduler steps (receiver named opt*, sch*, sched*, optim*). What remains must be in ALLOW, keyed
`path::qualified.function` with the reason. The one oracle look-ahead is `policies/oracle.py::ShadowTeacher.lookahead`
(a snapshot-restored look-ahead in a layer that may not import the harness). An entry that no longer matches a call is an
error, so the list can only shrink. The three loops that were episode loops in disguise now run on rollout
(`harness/data/collect_dual.py`, `harness/train/pointer/diagnostics.py`; the dual teacher's per-arm planner method is no
longer called `step`); what stays is (a) oracles that advance their OWN twin / snapshot, (b) physics settle tests of a
freshly built model and (c) vectorised pools with no per-seed episodes."""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "rrp"
ENV_DIRS = ("envs/",)
ROLLOUT = "harness/rollout.py"
STEP_NAMES = {"step", "mj_step", "mj_step1", "mj_step2"}
OPTIMIZER = re.compile(r"^(opt|optim|optimizer|sch|sched|scheduler)\w*$", re.I)

ALLOW = {
    "policies/oracle.py::ShadowTeacher.lookahead":
        "THE oracle look-ahead: the teacher's next H commands on a snapshot that is restored",
    "policies/pointer/runtime.py::TeacherOracleSource._advance":
        "privileged pointer teacher oracle: advances its own twin env (never the evaluated one), not an episode loop",
    "policies/legged.py::OracleShadow.demo":
        "privileged oracle shadow that unrolls a tracker on its own copied MjData to label chunk actions, not the evaluated env",
    "bodies/surgery.py::validate_physics":
        "settle test of a freshly built model (finite state, drift), not an episode",
    "harness/eval/legged_catalog.py::physics_check":
        "settle test of a catalogued body (stands, finite), not an episode",
    "harness/eval/humanoid_eval.py::gap_smoke_main":
        "vectorised WarpGapEnv smoke eval (256 lanes), not a per-seed episode",
    "harness/train/tracker_training.py::train":
        "vectorised PPO rollout pool of the tracker trainer, not a per-seed episode",
    "harness/train/warp_tracker_ppo.py::train":
        "vectorised PPO rollout on the warp tracker env, not a per-seed episode",
}


def step_calls(text: str) -> list[tuple[int, str, str]]:
    """(line, qualified enclosing function, call text) of every stepping call outside an `Env.step` body and outside an
    optimizer / scheduler receiver."""
    tree = ast.parse(text)
    out: list[tuple[int, str, str]] = []

    def visit(node, scope: list[str]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, scope + [child.name])
                continue
            if isinstance(child, ast.Call):
                f = child.func
                name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
                if name in STEP_NAMES and not (scope and scope[-1] == "step"):
                    recv = f.value if isinstance(f, ast.Attribute) else None
                    tail = (recv.attr if isinstance(recv, ast.Attribute) else recv.id if isinstance(recv, ast.Name)
                            else "")
                    if not (name == "step" and OPTIMIZER.match(tail or "")):
                        out.append((child.lineno, ".".join(scope), ast.unparse(f)))
            visit(child, scope)

    visit(tree, [])
    return out


def scan() -> tuple[list[str], set[str]]:
    bad, seen = [], set()
    for p in sorted(SRC.rglob("*.py")):
        rel = p.relative_to(SRC).as_posix()
        if rel == ROLLOUT or rel.startswith(ENV_DIRS):
            continue
        for line, fn, call in step_calls(p.read_text()):
            key = f"{rel}::{fn}"
            seen.add(key)
            if key not in ALLOW:
                bad.append(f"src/rrp/{rel}:{line}: {call}(...) in {fn or '<module>'}")
    return bad, seen


def test_scanner_flags_env_and_simulator_steps_and_ignores_optimizers():
    src = ("import mujoco\n"
           "def loop(env, d, opt, sch):\n    env.step(1)\n    mujoco.mj_step(1, d)\n    opt.step()\n    sch.step()\n"
           "class Net:\n    def fit(self):\n        self.opt_q.step()\n        self.env.step(None)\n"
           "class E:\n    def step(self, c):\n        mujoco.mj_step(self.m, self.d)\n")
    assert step_calls(src) == [(3, "loop", "env.step"), (4, "loop", "mujoco.mj_step"), (10, "Net.fit", "self.env.step")]


def test_no_stepping_outside_rollout_and_env_modules_but_the_allowlist():
    bad, _ = scan()
    assert not bad, "stepping outside harness/rollout.py and envs/ (route it through rollout, or justify in ALLOW):\n" \
        + "\n".join(bad)


def test_allowlist_has_no_stale_entries_and_one_oracle_lookahead():
    _, seen = scan()
    assert not (set(ALLOW) - seen), f"stale ALLOW entries: {sorted(set(ALLOW) - seen)}"
    assert "policies/oracle.py::ShadowTeacher.lookahead" in ALLOW
    assert [k for k in ALLOW if k.startswith("policies/oracle.py")] == ["policies/oracle.py::ShadowTeacher.lookahead"]


def test_unported_loops_are_not_allowed_and_the_list_stays_small():
    for gone in ("harness/data/collect_dual.py::collect_dual_episode", "harness/train/pointer/diagnostics.py::cmd_edit",
                 "policies/teachers/dual.py::DualTeacherBase.act"):
        assert gone not in ALLOW, gone
    assert len(ALLOW) <= 8, "the stepping ALLOW list may only shrink"
    assert all(len(why) > 30 and "not yet" not in why for why in ALLOW.values()), "each entry states why it is not an episode loop"
