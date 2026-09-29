"""D-126 move parity: scripts/ladder.py -> rrp.evaluation.ladder_cli and the legged summary scripts ->
rrp.evaluation.legged_summaries. The pre-move scripts are frozen verbatim (git 733b02a) under tests/data/legacy_scripts;
old and new run as subprocesses on the same inputs and must produce identical stdout and output files.
Only exception: the ladder rows' `wall_s` (per-episode wall-clock time) is dropped before comparing the .jsonl rows."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
LEGACY = REPO / "tests" / "data" / "legacy_scripts"
TIMING_FIELDS = ("wall_s",)


def _run(script: Path, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               PYTHONWARNINGS="ignore::DeprecationWarning")
    return subprocess.run([sys.executable, str(script), *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=120)


def _files(d: Path) -> dict[str, bytes]:
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


# ------------------------------------------------------------------ ladder
LADDER_ARGS = ["--route", "teacher", "--robot", "panda_pg2", "--n", "1", "--max-steps", "30", "--batch", "1", "--no-compare"]


def _ladder_pair(tmp_path, extra):
    res = {}
    for tag, script in (("old", LEGACY / "ladder.py"), ("new", REPO / "scripts" / "ladder.py")):
        out = tmp_path / tag
        r = _run(script, [*LADDER_ARGS, *extra, "--out", str(out)], REPO)
        assert r.returncode == 0, r.stderr[-2000:]
        res[tag] = (r, out)
    return res


@pytest.mark.menagerie
@pytest.mark.parametrize("extra", [[], ["--tag", "shift", "--object-shift", "5,0.01,-0.01", "--prev-action", "own"]])
def test_ladder_cli_matches_legacy_script(tmp_path, extra):
    res = _ladder_pair(tmp_path, extra)
    (ro, do), (rn, dn) = res["old"], res["new"]
    assert ro.stdout == rn.stdout
    fo, fn = _files(do), _files(dn)
    assert sorted(fo) == sorted(fn) and len(fo) == 2
    for name in fo:
        if name.endswith(".summary.json"):
            assert fo[name] == fn[name], name
        else:
            ro_ = [json.loads(x) for x in fo[name].decode().splitlines()]
            rn_ = [json.loads(x) for x in fn[name].decode().splitlines()]
            assert len(ro_) == len(rn_) == 1
            for a, b in zip(ro_, rn_):
                assert a.keys() == b.keys() and list(a) == list(b)
                for k in TIMING_FIELDS:
                    a.pop(k), b.pop(k)
                assert a == b


def test_ladder_cli_parser_and_guards_match_legacy(tmp_path):
    for args in (["--help"], ["--route", "teacher", "--robot", "xarm7_pg2", "--out", str(tmp_path / "x")],
                 ["--route", "teacher", "--robot", "panda_pg2", "--seed-start", "5", "--out", str(tmp_path / "x")],
                 ["--route", "bogus"]):
        ro, rn = _run(LEGACY / "ladder.py", args, REPO), _run(REPO / "scripts" / "ladder.py", args, REPO)
        assert (ro.returncode, ro.stdout, ro.stderr) == (rn.returncode, rn.stdout, rn.stderr), args


# ------------------------------------------------------------------ legged summaries (synthetic rows)
def _legged_row(rng, seed, edit, t_edit=1.0):
    trace, x, y, a = [], 0.0, 0.0, float(rng.uniform(-1, 1))
    for i in range(12):
        t = 0.25 * i
        x += 0.05 + 0.02 * rng.standard_normal(); y += 0.02 * rng.standard_normal(); a += 0.05 * rng.standard_normal()
        trace.append(dict(t=t, pose=[x, y, a], contact=[int(v) for v in rng.integers(0, 2, 4)]))
    packets = [dict(t=0.5 * i, ev=int(i > 3), dz_norm=float(abs(rng.standard_normal()))) for i in range(6)]
    ok = bool(rng.integers(0, 2))
    return dict(seed=seed, source="learned:flow_x", edit=edit, success=ok, failure_stage=None if ok else "walk_to_b",
                events={"walk_to_a": "succeeded", "walk_to_b": "succeeded" if ok else "active"}, t_edit=t_edit,
                trace=trace, packets=packets, fell=bool(rng.integers(0, 2)),
                waypoints=dict(a=[1.0, float(rng.uniform(-1, 1))], b=[2.0, float(rng.uniform(-1, 1))]))


def _suite(d: Path, edits=("none", "halt", "mirror_goal")):
    d.mkdir(parents=True)
    rng = np.random.default_rng(7)
    for e in edits:
        rows = [_legged_row(rng, s, e, t_edit=0.0 if (e == "none" and s == 3) else 1.0) for s in range(6)]
        if e == "mirror_goal":
            rows[0]["trace"] = rows[0]["trace"][-1:]        # too-short window -> skipped pair
        (d / f"{e}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


@pytest.mark.parametrize("tool,args", [
    ("legged_ladder_summary", lambda d: [str(d / "none.jsonl"), str(d / "halt.jsonl"), str(d / "x.summary.json")]),
    ("legged_edit_effects", lambda d: [str(d)]),
    ("legged_mirror_effect", lambda d: [str(d), "mirror_goal", "halt"]),
])
def test_legged_summaries_match_legacy_scripts(tmp_path, tool, args):
    outs = {}
    for tag, script in (("old", LEGACY / f"{tool}.py"), ("new", REPO / "scripts" / f"{tool}.py")):
        d = _suite(tmp_path / tag / "suite")
        r = _run(script, args(d), tmp_path / tag)
        assert r.returncode == 0, r.stderr[-2000:]
        outs[tag] = (r.stdout.replace(str(tmp_path / tag), "<T>"), {k: v.replace(str(tmp_path / tag).encode(), b"<T>")
                                                                     for k, v in _files(tmp_path / tag).items()})
    assert outs["old"][0] == outs["new"][0] and outs["old"][0]
    assert outs["old"][1] == outs["new"][1]
    assert len(outs["new"][1]) > 3                           # the tool wrote its output file(s)


def test_legged_summaries_library_calls_are_repeatable(tmp_path):
    """Unlike the scripts' module-level default rng, each library call starts its own rng(0) stream."""
    from rrp.harness.eval.legged_summaries import edit_effects, wilson
    d = _suite(tmp_path / "s")
    assert edit_effects(d) == edit_effects(d)
    lo, hi = wilson(3, 10)
    assert lo < 0.3 < hi and math.isclose(wilson(0, 0)[1], 0.0)
