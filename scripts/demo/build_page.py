"""Build docs/demo/index.html (demo sprint) from saved raw outputs.

Every table cell that is a measured number is read from a raw JSON/JSONL file here and printed with its path.
Narrative numbers that live only in a decision record cite the decision id (research/decisions.md) plus the raw path
named there. Raw files are small copies pulled by scripts/demo/pull_raw.sh into docs/demo/raw/artifacts/runs/<path>;
git-tracked results under artifacts/runs/ are read in place.

Usage (from the worktree root): python3 scripts/demo/build_page.py
"""
from __future__ import annotations

import datetime as dt
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "docs/demo/raw/artifacts/runs"
OUT = ROOT / "docs/demo/index.html"
VID = ROOT / "docs/demo/video"
HOST_RUNS = Path.home() / "work/relational-robot-policy/artifacts/runs"   # host-only raw outputs (BC job)

USED: set[str] = set()


def rawpath(p: str) -> Path:
    for base in (RAW, ROOT):
        f = base / p
        if f.exists():
            USED.add(p)
            return f
    raise FileNotFoundError(p)


def J(p: str):
    return json.loads(rawpath(p).read_text())


def JL(p: str):
    return [json.loads(x) for x in rawpath(p).read_text().splitlines() if x.strip()]


def have(p: str) -> bool:
    return any((b / p).exists() for b in (RAW, ROOT))


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def esc(s) -> str:
    return html.escape(str(s))


def src(*paths: str) -> str:
    """Inline source tag: decision ids (D-0xx) or raw paths."""
    parts = []
    for p in paths:
        if p.startswith("D-"):
            parts.append(f'<span class="did">{esc(p)}</span>')
        else:
            parts.append(f'<code>{esc(p)}</code>')
    return f'<span class="src">source: {", ".join(parts)}</span>'


def badge(kind: str, text: str | None = None) -> str:
    label = dict(teacher="scripted_teacher (privileged)", oracle="ORACLE diagnostic (not deployable)",
                 learned="learned", bc="learned (BC baseline)", none="not shown", run="running",
                 ok="verified", fail="fails")[kind]
    return f'<span class="badge b-{kind}">{esc(text or label)}</span>'


def frac(k, n, ci=True):
    if n is None:
        return "—"
    lo, hi = wilson(k, n)
    s = f"<b>{k}/{n}</b>"
    if ci:
        s += f' <span class="ci">[{lo:.2f}–{hi:.2f}]</span>'
    return s


def table(head: list[str], rows: list[list[str]], cls: str = "") -> str:
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


# ---------------------------------------------------------------- videos
VIDEOS = [
    # (file, kind, title, caption, source)
    ("2026-09-25_scripted_teacher_panda_pg2_pick_place_s3000002_success.mp4", "teacher",
     "pick_place · panda_pg2 · seed 3000002 · success",
     "Scripted teacher on a dev scene. Privileged poses, native joint targets, same tracker as every learned route.",
     "artifacts/runs/demo_video/INDEX.md"),
    ("2026-09-25_scripted_teacher_parm6_tf3_pick_place_s3000003_success.mp4", "teacher",
     "pick_place · parm6_tf3 (3-finger hand) · seed 3000003 · success",
     "Same teacher and task on a different body (6-DoF arm, three-finger gripper).",
     "artifacts/runs/demo_video/INDEX.md"),
    ("2026-09-25_scripted_teacher_panda_pg2_pick_place_paired_patient0_s3000001_success.mp4", "teacher",
     "binding pair · task bound to cube #0 · success",
     "Paired binding scene: the physical scene is identical in all three clips; only the task's patient binding differs.",
     "artifacts/runs/demo_video/INDEX.md"),
    ("2026-09-25_scripted_teacher_panda_pg2_pick_place_paired_patient1_s3000001_success.mp4", "teacher",
     "binding pair · task bound to cube #1 · success", "Same scene, patient rebound to cube #1.",
     "artifacts/runs/demo_video/INDEX.md"),
    ("2026-09-25_scripted_teacher_panda_pg2_pick_place_paired_patient2_s3000001_success.mp4", "teacher",
     "binding pair · task bound to cube #2 · success", "Same scene, patient rebound to cube #2.",
     "artifacts/runs/demo_video/INDEX.md"),
    ("2026-09-25_dual_scripted_teacher_assign_left_panda_pg2__ur5e_pg2_s3000001_success.mp4", "teacher",
     "dual-arm assign_left · panda (L) + UR5e (R) · success",
     "Manipulator-assignment pair: the task graph binds the LEFT arm. Identical initial scene to the next clip.",
     "artifacts/video/INDEX.md"),
    ("2026-09-25_dual_scripted_teacher_assign_right_panda_pg2__ur5e_pg2_s3000001_success.mp4", "teacher",
     "dual-arm assign_right · same scene · success", "The task graph binds the RIGHT arm; the teacher reads it from the public graph.",
     "artifacts/video/INDEX.md"),
    ("2026-09-25_dual_scripted_teacher_handover_panda_pg2__ur5e_pg2_s3000001_success.mp4", "teacher",
     "dual-arm handover · panda gives, UR5e receives · success", "Two-assembly task with take / offer / receive / release events.",
     "artifacts/video/INDEX.md"),
]
BC_VIDEOS = [
    ("2026-09-25_learned_bc_direct1701_final_parm5s_tf3_pick_place_s2000029_success.mp4", "bc",
     "plain BC final · held-out source body parm5s_tf3 · seed 2000029",
     "Direct-action BC (final, 26.3k updates) on a body it was never trained on (held-out source body).",
     "artifacts/runs/latent_slice1_b1fix/baseline_direct_action/seed1701/eval/source.summary.json"),
    ("2026-09-25_learned_bc_direct1701_final_zeroshot_newbody_xarm7_pg2_pick_place_s2000000_failure.mp4", "bc",
     "plain BC final · zero-shot NEW body xarm7_pg2 · FAILURE",
     "Budget 0 on a sealed target arm: 0/100. The unseen kinematic chain is not solved without target data.",
     "artifacts/runs/latent_slice1_b1fix/baseline_direct_action/seed1701/eval/xarm7_pg2_b0.summary.json"),
    ("2026-09-25_ladder_learned_panda_pg2_s3000000_direct1701_u12000_success.mp4", "bc",
     "plain BC · direct1701_u12000 · panda_pg2 · seed 3000000 · success",
     "Direct-action flow policy (B-1 fixed), chunk every 8 ticks, same tracker and scenes as the ladder.",
     "artifacts/runs/baselines_bc_ladder/panda_pg2/learned_direct1701_u12000.summary.json"),
    ("2026-09-25_ladder_learned_panda_pg2_s3000013_direct1701_u12000_success.mp4", "bc",
     "plain BC · direct1701_u12000 · panda_pg2 · seed 3000013 · success in this render",
     "The evaluation row of this seed FAILED (grasp); this GPU re-render succeeds. Flow sampling noise and GPU/CPU "
     "numerics make single episodes vary; the table's rates are the evidence, not the clips.",
     "artifacts/runs/baselines_bc_ladder/panda_pg2/learned_direct1701_u12000.summary.json"),
    ("2026-09-25_ladder_learned_parm6_tf3_s3000003_direct1701_u12000_failure-lift.mp4", "bc",
     "plain BC · direct1701_u12000 · parm6_tf3 · seed 3000003 · FAILURE (lift) in this render",
     "Same checkpoint on the 3-finger body. The evaluation row of this seed succeeded; this re-render fails at lift "
     "(run-to-run variance, see previous clip).",
     "artifacts/runs/baselines_bc_ladder/parm6_tf3/learned_direct1701_u12000.summary.json"),
    ("2026-09-25_ladder_learned_panda_pg2_s3000001_codec1701_u13152_success.mp4", "bc",
     "plain BC · action-only codec codec1701_u13152 · panda_pg2 · seed 3000001 · success",
     "Action-only codec baseline (flow on a 4-d action codec latent), same route.",
     "artifacts/runs/baselines_bc_ladder/panda_pg2/learned_codec1701_u13152.summary.json"),
]
TRIPTYCH = [
    ("2026-09-25_triptych_panda_pg2_s3000008_teacher_bc-direct1701_u12000_oracle-jointfix.mp4", "teacher",
     "same scene, three controllers · panda_pg2 · seed 3000008",
     "Left: R0 scripted teacher (privileged), success. Middle: plain BC learned:direct1701_u12000, success. Right: R1 ORACLE "
     "diagnostic, jointly retrained latent bundle (jointfix), failure at lift. Each panel carries its own source caption.",
     "artifacts/runs/demo_video/tri_3000008/INDEX.md"),
    ("2026-09-25_triptych_panda_pg2_s3000018_teacher_bc-direct1701_u12000_oracle-jfdag1.mp4", "teacher",
     "same scene, three controllers · panda_pg2 · seed 3000018",
     "Left: teacher, success. Middle: plain BC, success. Right: R1 ORACLE jfdag1 — this is the seed of its only evaluation "
     "success (D-048), but in this re-render it fails at approach: the 1/30 does not even reproduce reliably.",
     "artifacts/runs/demo_video/tri_3000018/INDEX.md"),
]
LADDER_VIDEOS = [
    ("2026-09-25_ladder_oracle_panda_pg2_s3000018_jfdag1_success.mp4", "oracle",
     "R1 oracle · jointfix + DAgger-1 system 0 · seed 3000018 · SUCCESS (the only one of 30)",
     "Teacher-encoded packet (uses the teacher's FUTURE actions) → learned system 0 → tracker. Re-anchored every 8 ticks "
     "(valid, D-049). This is the single success in 30 matched seeds (D-048); it is a diagnostic, not a deployable policy.",
     "artifacts/runs/ladder_v1/panda_pg2/oracle_zero_jfdag1_reanchor.summary.json"),
    ("2026-09-25_ladder_oracle_panda_pg2_s3000003_jfdag1_failure-approach.mp4", "oracle",
     "R1 oracle · jointfix + DAgger-1 · seed 3000003 · failure",
     "Same route, typical failure. This GPU re-render fails at approach; the CPU evaluation row of the same seed failed "
     "later, at lift (GPU/CPU numerical nondeterminism in closed loop).",
     "artifacts/runs/ladder_v1/panda_pg2/oracle_zero_jfdag1_reanchor.summary.json"),
    ("2026-09-25_ladder_oracle_panda_pg2_s3000008_jointfix_failure-lift.mp4", "oracle",
     "R1 oracle · jointly retrained Stage A (B-1 fix) · seed 3000008 · failure at lift",
     "The arm now reaches the cube (6 mm) but heads for the object rather than the oracle packet's pregrasp waypoint. "
     "Observation only: the oracle packet itself may be stale off the teacher trajectory (see the confound in §4).",
     "artifacts/runs/ladder_v1/panda_pg2/oracle_zero_jointfix_reanchor.summary.json"),
    ("2026-09-25_causal_edit_rebind_obj_oracle_latent_sem_v1_panda_pg2_s3000001.mp4", "oracle",
     "semantic edit rebind_obj · oracle packets (latent_sem_v1, pre-fix) · side by side",
     "Left: control packets; right: packets generated for the task REBOUND to distractor0. System 0 frozen. The packet "
     "changes behaviour but not in the edited direction; neither side grasps (D-041). Pre-B-1-fix model.",
     "artifacts/runs/acceptance_semantic_sem_v1rep/semantic_summary_oracle.json"),
    ("2026-09-25_learned_latent_v2s22k_base_panda_pg2_pick_place_s3000000_failure.mp4", "learned",
     "R2 generated · learned:flow_latent_sem_v2@22k · failure (B-1 contaminated)",
     "System i flow → packet → system 0 from reset. Never approaches the cube. Trained before the B-1 fix, so this clip "
     "shows the bug, not the architecture (D-044).",
     "D-042"),
]


def video_card(v) -> str:
    f, kind, title, cap, s = v
    lab = {"teacher": "teacher | BC | oracle" if f.startswith("2026-09-25_triptych") else "teacher | BC | stateless oracle" if "orcbctriptych" in f else "scripted_teacher", "oracle": "oracle diagnostic", "learned": "learned", "bc": "learned"}[kind]
    if kind == "learned":
        lab = ("teacher | BC | learned:flow_jointfix@" + (f.split("flowjf_s")[1].split(".")[0] if "flowjf_s" in f else "20k → sys-0 " + f.split("flowjf20k_")[1].split(".")[0])) if "r2triptych" in f \
            else ("learned:flow_jointfix@20k → sys-0 " + f.split("flowjf20k_")[1].split("_cpu")[0]) if "flowjf20k_" in f else "learned:flow_latent_sem_v2@22k"
    if kind == "bc":
        lab = "learned:" + ("direct1701 final (BC)" if "bc_direct1701_final" in f else "direct1701_u12000 (BC)" if "_s30000" not in f else
                            f.split("_s30000")[1].split("_", 1)[1].rsplit("_", 1)[0])
    exists = (VID / f).exists()
    size = f"{(VID / f).stat().st_size / 1e6:.2f} MB" if exists else "missing"
    body = (f'<video controls muted loop playsinline preload="metadata" src="video/{esc(f)}"></video>' if exists
            else '<div class="novid">video not rendered yet</div>')
    return (f'<figure class="vid">{body}<figcaption><span class="badge b-{kind}">{esc(lab)}</span> '
            f'<b>{esc(title)}</b><br>{esc(cap)}<br>{src(s)} <span class="muted">· {esc(f)} · {size}</span>'
            f'</figcaption></figure>')


# ---------------------------------------------------------------- sections
def ladder_rows():
    L = "ladder_v1/{r}/{t}.summary.json"

    def cell(r, t):
        p = L.format(r=r, t=t)
        if not have(p):
            return "—", None
        d = J(p)
        return frac(d["success"], d["n"]), d

    specs = [
        ("R0", "teacher", "scripted teacher (shadow check only)", "teacher_shadow_zero", None, "teacher"),
        ("R1", "oracle", "sem_v1 Stage-A system 0 (trained with the B-1 crutch)", "oracle_zero", None, "oracle"),
        ("R1", "oracle", "system 0 refit on frozen v1 encoder, B-1 fixed (ft)", "oracle_zero_rzft", "oracle_zero_rzft_reanchor", "oracle"),
        ("R1", "oracle", "… + DAgger round 1", "oracle_zero_dag1", "oracle_zero_dag1_reanchor", "oracle"),
        ("R1", "oracle", "… + DAgger round 2", "oracle_zero_dagger2", "oracle_zero_dagger2_reanchor", "oracle"),
        ("R1", "oracle", "… anchored system-0 input", "oracle_zero_anchor", "oracle_zero_anchor_reanchor", "oracle"),
        ("R1", "oracle", "Stage A retrained jointly with the fix + anchor (jointfix)", "oracle_zero_jointfix", "oracle_zero_jointfix_reanchor", "oracle"),
        ("R1", "oracle", "jointfix + system-0 DAgger round 1 (jfdag1)", "oracle_zero_jfdag1", "oracle_zero_jfdag1_reanchor", "oracle"),
        ("R1", "oracle", "binding v4 SEM bundle (paired data, B-1 fixed, anchored)", "oracle_zero_bindv4sem", "oracle_zero_bindv4sem_reanchor", "oracle"),
        ("R1", "oracle", "binding v4 NOSEM bundle (capacity-matched control)", "oracle_zero_bindv4nosem", "oracle_zero_bindv4nosem_reanchor", "oracle"),
        ("R2", "learned", "learned:flow_latent_sem_v2@24543 → sem_v1 system 0 (both pre-fix)", "generated_v2s24543_zero", None, "learned"),
    ]
    rows = []
    for rung, _, what, plain, re_, kind in specs:
        pp, dp = cell("panda_pg2", plain)
        qp, dq = cell("parm6_tf3", plain)
        pr, dr = cell("panda_pg2", re_) if re_ else ("n/a", None)
        qr, _ = cell("parm6_tf3", re_) if re_ else ("n/a", None)
        d = dr or dp
        stages = ", ".join(f"{k} {v}" for k, v in (d or {}).get("failed_stage", {}).items()) if d else "—"
        mtc = f'{d["min_tcp_cube_m"] * 100:.1f} cm' if d else "—"
        rows.append([f'<span class="badge b-{kind}">{rung}</span>',
                     f'{esc(what)}<br><span class="muted" style="font-size:.8em">tag {esc(plain)}</span>',
                     pp, pr, qp, qr, esc(stages), mtc])
    return rows


SEM_VIDEOS = [
    ("2026-09-25_semantic_edit_rebind_obj_scripted_teacher_panda_pg2_k31000080.mp4", "teacher",
     "rebind (paired scene) · teacher · left control, right edited",
     "Only the task's binding changes (the entity descriptor now names the other cube). The teacher goes to the new cube.",
     "artifacts/runs/acceptance_sprint_sem_teacher_paired/semantic_summary_teacher.json"),
    ("2026-09-25_semantic_edit_rebind_obj_oracle_ladder_latent_sem_b1fix_anchor_panda_pg2_k31000080.mp4", "oracle",
     "rebind (same scene) · ORACLE packet, jointfix bundle · approaches and touches the new cube",
     "The oracle packet encodes the teacher's demonstration to the new cube, so following it shows system 0 reads the "
     "packet's content, not that anything reads the supplied binding (weak evidence). No task success.",
     "artifacts/runs/acceptance_sprint_sem_b1fix_oracle/semantic_summary_oracle.json"),
    ("2026-09-25_semantic_edit_irrelevant_distractor_oracle_ladder_latent_sem_b1fix_anchor_panda_pg2_k31000080.mp4", "oracle",
     "irrelevant edit (control) · ORACLE packet, same scene · unchanged",
     "The matched irrelevant edit (an unbound object's belief moved 10 cm) leaves the approach unchanged.",
     "artifacts/runs/acceptance_sprint_sem_b1fix_oracle/semantic_summary_oracle.json"),
    ("2026-09-25_semantic_edit_swap_arm_scripted_teacher_panda_pg2__ur5e_pg2_k3000000.mp4", "teacher",
     "arm assignment swap · teacher · dual-arm",
     "The actor of take/place is rebound to the other arm; the other arm does the task.",
     "artifacts/runs/acceptance_sprint_arm_teacher/arm_summary_teacher.json"),
    ("2026-09-25_semantic_edit_rebind_desc_learned_bc_direct1701_u12000_panda_pg2_k3000008.mp4", "bc",
     "descriptor rebind · plain BC (not latent) · IGNORED",
     "The competent BC controller does not follow a pure binding change: it keeps going to the original cube.",
     "artifacts/runs/acceptance_sprint_sem_bc_pp/semantic_summary_bc.json"),
    ("2026-09-25_semantic_edit_goal_shift_learned_bc_direct1701_u12000_panda_pg2_k3000008.mp4", "bc",
     "goal edit · plain BC (not latent) · followed",
     "Goal belief moved 12 cm: BC places the cube at the new goal.",
     "artifacts/runs/acceptance_sprint_sem_bc_pp/semantic_summary_bc.json"),
]


R2_VIDEOS = [
    ("2026-09-25_ladder_generated_parm6_tf3_s3000012_flowjf20k_gendag1noqd_cpu_success.mp4", "learned",
     "R2 generated · learned:ladder_flow_jointfix final → system 0 gendag1_noqd · parm6_tf3 · seed 3000012 · SUCCESS",
     "The deployable latent route (system i's own packets, no teacher, no BC in the loop) completes pick-and-place. One of 9/30 "
     "evaluation successes on parm6_tf3, re-rendered with the model on CPU like the evaluation. A CPU re-render of another "
     "success seed (3000003) failed at transport, and GPU re-renders of 3 success seeds failed late (place/transport).",
     "ladder_v1/parm6_tf3/generated_zero_ladder_flow_jointfix_snap_final_s20000_rzgendag1noqd.summary.json"),
    ("2026-09-25_r2triptych_parm6_tf3_s3000012_teacher_bc-direct1701_u12000_generated-flowjf20k_gendag1noqd.mp4", "learned",
     "same scene · teacher | plain BC | R2 (flow final → system 0 gendag1_noqd) · parm6_tf3 · seed 3000012 · GPU render",
     "This GPU render: teacher success; BC fails at lift (BC also varies run to run); R2 carries the cube and fails at place.",
     "artifacts/runs/demo_video/r2_parm6_tf3_3000012/INDEX.md"),
    ("2026-09-25_r2triptych_parm6_tf3_s3000038_teacher_bc-direct1701_u12000_generated-flowjf20k_rzbcdag2.mp4", "learned",
     "same scene · teacher | plain BC | R2 generated (flow_jointfix final → system 0 jfbcdag2) · parm6_tf3 · seed 3000038",
     "Seed 3000038 is the ONLY R2 success in the evaluation (1/30). In this re-render the R2 panel fails at grasp: the "
     "first deployable-route success does not reproduce on demand (flow sampling noise), so treat it as a single event.",
     "ladder_v1/parm6_tf3/generated_zero_flowjf_s20000_rzbcdag2.summary.json"),
    ("2026-09-25_r2triptych_panda_pg2_s3000029_teacher_bc-direct1701_u12000_generated-flowjf_s4000.mp4", "learned",
     "same scene · teacher | plain BC | R2 generated (flow_jointfix@4000) · panda_pg2 · seed 3000029",
     "Right panel: the deployable latent route, system i's own packet → system 0. This render: teacher success, BC success, "
     "R2 failure at grasp. (The evaluation row of this seed got furthest of all 30, failing only at place.)",
     "artifacts/runs/demo_video/r2_panda_pg2_3000029/INDEX.md"),
    ("2026-09-25_r2triptych_panda_pg2_s3000008_teacher_bc-direct1701_u12000_generated-flowjf_s4000.mp4", "learned",
     "same scene · teacher | plain BC | R2 generated (flow_jointfix@4000) · panda_pg2 · seed 3000008",
     "This render: teacher success, BC success, R2 failure at approach (the evaluation row reached the cube, 5 mm, and "
     "failed at grasp).",
     "artifacts/runs/demo_video/r2_panda_pg2_3000008/INDEX.md"),
]


ORCBC_VIDEOS = [
    ("2026-09-25_ladder_oracle_parm6_tf3_s3000008_jfbcdag1long_orcbc_success.mp4", "oracle",
     "stateless R1 · system 0 jfbcdag1long · parm6_tf3 · seed 3000008 · success",
     "Packet = E(the chunk the competent BC would execute now); no teacher state. Success in both the evaluation and this render.",
     "ladder_v1/parm6_tf3/oracle_zero_jfbcdag1long_orcbc.summary.json"),
    ("2026-09-25_ladder_oracle_parm6_tf3_s3000006_jfbcdag1long_orcbc_failure-lift.mp4", "oracle",
     "stateless R1 · jfbcdag1long · parm6_tf3 · seed 3000006 · failure (lift) in this render",
     "The evaluation row of this seed succeeded; this re-render fails at lift (run-to-run variance).",
     "ladder_v1/parm6_tf3/oracle_zero_jfbcdag1long_orcbc.summary.json"),
    ("2026-09-25_ladder_oracle_panda_pg2_s3000017_jfbcdag1long_orcbc_success.mp4", "oracle",
     "stateless R1 · jfbcdag1long · panda_pg2 · seed 3000017 · success",
     "One of the 3 panda successes; success in both the evaluation and this render.",
     "ladder_v1/panda_pg2/oracle_zero_jfbcdag1long_orcbc.summary.json"),
    ("2026-09-25_ladder_oracle_panda_pg2_s3000000_jfbcdag1long_orcbc_failure-approach.mp4", "oracle",
     "stateless R1 · jfbcdag1long · panda_pg2 · seed 3000000 · failure (approach)",
     "Typical panda failure: the hand does not settle over the cube.",
     "ladder_v1/panda_pg2/oracle_zero_jfbcdag1long_orcbc.summary.json"),
]


ORCBC_TRI = [
    ("2026-09-25_orcbctriptych_parm6_tf3_s3000008_teacher_bc_orcbc-jfbcdag2.mp4", "teacher",
     "same scene · teacher | plain BC | stateless R1 → system 0 jfbcdag2 · parm6_tf3 · seed 3000008",
     "All three succeed, in the evaluation and in this render. Right panel: ORACLE DIAGNOSTIC packet = E(BC's chunk), realized by the refit system 0.",
     "ladder_v1/parm6_tf3/oracle_zero_jfbcdag2_orcbc.summary.json"),
    ("2026-09-25_orcbctriptych_panda_pg2_s3000000_teacher_bc_orcbc-jfbcdag2.mp4", "teacher",
     "same scene · teacher | plain BC | stateless R1 jfbcdag2 · panda_pg2 · seed 3000000",
     "All three succeed in this render. The evaluation row of the stateless R1 for this seed FAILED (place): single episodes vary.",
     "ladder_v1/panda_pg2/oracle_zero_jfbcdag2_orcbc.summary.json"),
    ("2026-09-25_orcbctriptych_panda_pg2_s3000001_teacher_bc_orcbc-jfbcdag2.mp4", "teacher",
     "same scene · teacher | plain BC | stateless R1 jfbcdag2 · panda_pg2 · seed 3000001",
     "Teacher and BC succeed; the stateless R1 fails at lift in this render (its evaluation row succeeded).",
     "ladder_v1/panda_pg2/oracle_zero_jfbcdag2_orcbc.summary.json"),
]


def _re_step(path: str) -> int:
    import re
    m = re.search(r"s(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def sec_sprint():
    """Top 'sprint update' block, generated from the sprint agents' raw outputs when present."""
    parts = []
    T = "artifacts/runs/acceptance_sprint_sem_teacher_paired/semantic_summary_teacher.json"
    O = "artifacts/runs/acceptance_sprint_sem_b1fix_oracle/semantic_summary_oracle.json"
    B = "artifacts/runs/acceptance_sprint_sem_bc_pp/semantic_summary_bc.json"
    A = "artifacts/runs/acceptance_sprint_arm_teacher/arm_summary_teacher.json"
    if have(T) and have(O):
        def ci(d):
            return f'{d["mean"]:+.3f} <span class="ci">[{d["lo"]:+.3f}, {d["hi"]:+.3f}]</span>' if d else "—"

        def fc(s, c):
            v = s.get(c, {}).get("first_contact_new_frac")
            return f'{v["k"]}/{v["n"]}' if v else "—"

        def goal(s):
            g = s.get("goal_shift")
            return f'{g["cube_at_shifted_goal"]}/{g["n"]}' if g else "—"
        rows = []
        t = J(T)["summary"]
        arm = ""
        if have(A):
            a = J(A)["summary"]
            e, c = a["swap_arm"]["first_contact_is_edited_to"], a["control"]["first_contact_is_edited_to"]
            arm = f'{e["k"]}/{e["n"]} (control {c["k"]}/{c["n"]})'
        rows.append([f'{badge("teacher")}<br>paired scenes', "binding (descriptor)", f'{fc(t, "rebind_obj")} / {fc(t, "control")}',
                     ci(t["_contrasts"].get("rebind_obj-irrelevant_distractor:pref_min")), goal(t), arm or "—"])
        o = J(O)["summary"]
        rows.append([f'{badge("oracle", "ORACLE: E(jointfix) + teacher demo")}<br>paired scenes', "binding (descriptor)",
                     f'{fc(o, "rebind_obj")} / {fc(o, "control")}',
                     ci(o["_contrasts"].get("rebind_obj-irrelevant_distractor:pref_min")), goal(o), "n/a (single-arm bundle)"])
        if have(B):
            b = J(B)["summary"]
            rows.append([f'{badge("bc", "learned:direct1701_u12000 (BC, NOT latent)")}<br>canonical scenes', "binding (descriptor only)",
                         f'{fc(b, "rebind_desc")} / {fc(b, "control")}',
                         ci(b["_contrasts"].get("rebind_desc-irrelevant_distractor:pref_min")), goal(b), "n/a"])
            rows.append([f'{badge("bc", "learned:direct1701_u12000 (BC, NOT latent)")}<br>canonical scenes', "object beliefs swapped",
                         f'{fc(b, "rebind_obj")} / {fc(b, "control")}',
                         ci(b["_contrasts"].get("rebind_obj-irrelevant_distractor:pref_min")), "(same run)", "n/a"])
        for v in ("sem", "nosem"):
            V = f"artifacts/runs/acceptance_sprint_sem_v4{v}_oracle/semantic_summary_oracle.json"
            if have(V):
                o4 = J(V)["summary"]
                orth = o4["_contrasts"].get("rebind_obj-orthogonal_matched:pref_min")
                rows.append([f'{badge("oracle", f"ORACLE: E(binding v4 {v.upper()}) + teacher demo")}<br>paired scenes', "binding (descriptor)",
                             f'{fc(o4, "rebind_obj")} / {fc(o4, "control")}',
                             ci(o4["_contrasts"].get("rebind_obj-irrelevant_distractor:pref_min"))
                             + (f'<br><span class="ci">vs orthogonal edit: {orth["mean"]:+.3f} [{orth["lo"]:+.3f}, {orth["hi"]:+.3f}]</span>' if orth else ""),
                             goal(o4), "n/a"])
        BR = "artifacts/runs/acceptance_sprint_sem_best_orcbc/semantic_summary_oracle.json"
        bb_ctrl = bb_goal = bb_irr = bb_orth = bb_orth_s = bb_rb = "—"
        if have(BR):
            bb = J(BR)["summary"]
            f2 = lambda c, k: f'{bb[c][k]}/{bb[c]["n"]}'
            bb_ctrl, bb_goal, bb_irr, bb_orth = f2("control", "privileged_success"), f2("goal_shift", "cube_at_shifted_goal"), f2("irrelevant_distractor", "cube_at_shifted_goal"), f2("orthogonal_matched", "cube_at_shifted_goal")
            bb_orth_s, bb_rb = f2("orthogonal_matched", "privileged_success"), fc(bb, "rebind_desc")
            g_ = bb["goal_shift"]
            rows.append([f'{badge("oracle", "BEST ROUTE, ORACLE: E(BC chunk for the edited context) → system 0 jfbcdag2")}<br>canonical scenes',
                         "binding (descriptor only)",
                         f'{fc(bb, "rebind_desc")} / {fc(bb, "control")}',
                         ci(bb["_contrasts"].get("rebind_desc-irrelevant_distractor:pref_min")) + '<br><span class="ci">by abandoning the old cube, not reaching the new one</span>',
                         f'<b>{g_["cube_at_shifted_goal"]}/{g_["n"]}</b><br><span class="ci">irrelevant {bb["irrelevant_distractor"]["cube_at_shifted_goal"]}/{bb["irrelevant_distractor"]["n"]}, orthogonal {bb["orthogonal_matched"]["cube_at_shifted_goal"]}/{bb["orthogonal_matched"]["n"]}; '
                         f'{ci(bb["_contrasts"].get("goal_shift-irrelevant_distractor:goal_pref"))} m beyond irrelevant</span>',
                         "n/a"])
        rows.append([f'{badge("learned", "learned: v4 sem / nosem flows")}', "generated route", badge("run"), badge("run"), badge("run"), badge("run")])
        parts.append("<h3>Semantic interventions at the level each route reaches (sprint_semantic)</h3>"
                     + table(["route", "edit type", "rebind: first touch on the NEW object (edit / control)",
                              "rebind effect beyond the matched irrelevant edit, min-distance preference (m) [95% CI]",
                              "goal edit: cube at the new goal", "arm swap: edited-to arm touches first"], rows)
                     + f"""<p>Reading: the edits and metrics are valid (the teacher follows them). The oracle packet makes system 0
approach the rebound object, but that packet encodes the teacher's demonstration toward it, so this is weak evidence
(system 0 reads packet content, not the binding). The competent BC controller follows a goal edit and a swap of object
<i>beliefs</i>, but <b>ignores a pure binding change</b>: that is the capability the semantic packet is meant to add, and the
generated-route test on the binding-v4 flows is pending. The final BC reference (D-065) follows goal edits (23/24, 16/17) and object-belief swaps
(23/24, 15/17), leaves irrelevant edits unchanged, and ignores descriptor-only rebinds: the latent packet must beat this reference, especially on binding,
to support the semantic claim. <b>Best latent route: system 0 causally executes goal content carried in the packet (oracle diagnostic, D-062).</b>
Packet = E(the BC chunk for the edited context) → system 0 jfbcdag2, panda_pg2, 48 seeds, control success {bb_ctrl}. A valid goal edit puts
the cube at the NEW goal in {bb_goal} vs {bb_irr} and {bb_orth} under the matched controls. A probe-orthogonal edit of matched norm drops success to
{bb_orth_s}. A binding edit is NOT followed ({bb_rb} first touch on the new object). <b>This does not show that semantic supervision adds control:</b>
the BC expert already follows goal edits, the packet encodes BC's chunk, and bindings are not carried because BC ignores them.
{src(BR, 'D-062')} <b>Binding v4 sem vs nosem (oracle route, D-059):</b> no semantic
advantage at the behaviour level. Neither v4 system 0 is competent before refitting, and the sem effect is not separable from a
matched-norm probe-orthogonal edit. The claim that semantic supervision adds causal control remains <b>not shown</b>. {src('D-059')} {src(T, O, B, A, 'research/tracks/acceptance.md (SPRINT SEMANTIC RESULTS)')}</p>
<div class="grid">{''.join(video_card(v) for v in SEM_VIDEOS)}</div>""")

    # R2 rows keyed by the checkpoints recorded in each summary (flow snapshot x system 0), matched dev seeds only
    r2rows = {}
    for f in (RAW / "ladder_v1").glob("*/generated_zero_*.summary.json"):
        d = J(str(f.relative_to(RAW)))
        c = d.get("checkpoints") or {}
        fl = (c.get("flow") or {}).get("path", "")
        rep = (c.get("representation") or {}).get("path", "")
        if "ladder_flow_jointfix" not in fl:
            continue
        fresh = " · FRESH seeds " + f.name.split("fresh")[1].split(".")[0] + "+ (not the matched set)" if "fresh" in f.name else ""
        step = (Path(fl).parent.name.replace("ladder_flow_", ""), _re_step(fl))
        s0 = Path(rep).parent.name.replace("ladder_rz_jointfix_", "").replace("ladder_latent_sem_b1fix_anchor", "jointfix")
        r2rows.setdefault((step, s0, fresh), {})[f.parent.name] = d
    if r2rows:
        rows = []
        for (step, s0, fresh), per in sorted(r2rows.items()):
            desc0 = {"gendag1": " (as gendag1_noqd but WITH the joint-velocity input)", "gendag1_qdd": " (gendag1 variant with joint-velocity dropout)", "gendag1_noqd": " (refit from bcdag2: no joint-velocity input, BC-expert DAgger incl. states visited with GENERATED packets, z-noise 0.3; config configs/ladder/rz_jointfix_gendag1_noqd.json on track/ladder)"}.get(s0, "")
            cells = [f'<span class="badge b-learned">R2 learned:ladder_flow_{step[0]}{"@" + str(step[1]) if step[1] >= 0 else " (final)"}</span> → system 0 {esc(s0)}<b>{esc(fresh)}</b><span class="ci">{esc(desc0)}</span>']
            for r in ("panda_pg2", "parm6_tf3"):
                d = per.get(r)
                if d:
                    cells += [frac(d["success"], d["n"]),
                              esc(", ".join(f"{kk} {v}" for kk, v in d["failed_stage"].items() if kk != "success"))
                              + f' <span class="ci">(min TCP–cube {d["min_tcp_cube_m"] * 100:.1f} cm)</span>']
                else:
                    cells += ["—", "—"]
            rows.append(cells)
        known = {"jointfix": "jointfix", "jfdag1": "jfdag1 (shadow DAgger r1)", "jfdag2df08": "jfdag2df08 (shadow DAgger r1+r2)",
                 "jfbcdag1": "jfbcdag1 (BC-expert DAgger)", "jfbcdag1long": "jfbcdag1long (BC-expert DAgger, 16k steps, lr 3e-4)", "jfnoqd": "jfnoqd (no joint-velocity input)", "jfbcdag2": "jfbcdag2 (BC-expert DAgger round 2)", "jfbcdag3": "jfbcdag3 (BC-expert DAgger round 3)", "jfbig16k": "jfbig16k (larger system 0, 16k steps, BC-expert DAgger)", "jfnoqd": "jfnoqd (no joint-velocity input)", "bindv4sem": "binding v4 SEM bundle", "bindv4sembcdag1": "binding v4 SEM + BC-expert DAgger r1", "bindv4nosembcdag1": "binding v4 NOSEM + BC-expert DAgger r1", "bindv4nosem": "binding v4 NOSEM bundle (capacity-matched control)"}
        found = sorted({f.name[len("oracle_zero_"):-len("_orcbc.summary.json")] for f in (RAW / "ladder_v1").glob("*/oracle_zero_*_orcbc.summary.json")},
                       key=lambda t: (list(known).index(t) if t in known else 99, t))
        for t_, lab in ((t, known.get(t, t)) for t in found):
            ps = [f"ladder_v1/{r}/oracle_zero_{t_}_orcbc.summary.json" for r in ("panda_pg2", "parm6_tf3")]
            if not any(have(p) for p in ps):
                continue
            cells = [f'<span class="badge b-oracle">R1 ORACLE, stateless: E(BC chunk) → system 0 {esc(lab)}</span>']
            for p in ps:
                if have(p):
                    d = J(p)
                    cells += [frac(d["success"], d["n"]),
                              esc(", ".join(f"{k} {v}" for k, v in d["failed_stage"].items() if k != "success"))]
                else:
                    cells += ["—", "—"]
            rows.append(cells)
        for tag, lab in (("direct1701_u12000", "learned:direct1701_u12000 (plain BC, same seeds)"),):
            cells = [f'<span class="badge b-bc">{lab}</span>']
            for r in ("panda_pg2", "parm6_tf3"):
                d = J(f"artifacts/runs/baselines_bc_ladder/{r}/learned_{tag}.summary.json")
                cells += [frac(d["success"], d["n"]), esc(", ".join(f"{k} {v}" for k, v in d["failed_stage"].items() if k != "success"))]
            rows.append(cells)
        parts.insert(0, "<h3>Best latent route so far vs BC (sprint_latent, SPRINT BEST ROUTE): R2 generated and stateless R1 on the B-1-fixed bundle</h3>"
                     + table(["controller", "panda_pg2", "failures (stage)", "parm6_tf3", "failures (stage)"], rows)
                     + f"""<p>R2 = system i's own packets (flow trained with <code>zero_prev_action</code> on the jointly trained
encoder, snapshots as training proceeds) → the jointly trained system 0 → tracker; no teacher in the loop. Same 30 matched
dev seeds as BC. The flow is still training (20k steps planned); rows are added as snapshots are evaluated.
The stateless R1 rows replace the confounded shadow-teacher oracle: the packet is E(the chunk the competent BC would
execute at the current state), with no teacher state. <b>Diagnosis (D-052 and its 20:15 refinement): both stages fall short. System 0's underfit is the first gate, and the
generator is also short at this snapshot</b> (generator-gap row below). On its own training pack it explains only ~30% of the teacher's 1-step motion (underfit). With the original jointfix system 0, packets that encode the competent BC's own chunks
still give 0/30. <b>Improving system 0's fit is the first lever that converts:</b> refitting system 0 with DAgger relabelled by the stateless BC expert
(jfbcdag1long, then round 2 jfbcdag2) lifts the same stateless oracle route to the first latent-path successes (D-053; the deployable R2 route follows, D-063) and
then further (D-055; see rows), still far below BC and not deployable
(the packet encodes BC's own chunk). R2 through this refit (rows “→ system 0 rzlong”) is the next test. At BC's own states, system 0 explains only part of BC's 1-step motion, and none of it on the first tick of each
packet. R2 fails at the same stages as the stateless oracle. Next: fix the system-0 fit offline, gated on arm error at BC states ≤ 20% of hold-still before any closed-loop run. {src('D-052')}
{src('ladder_v1/<robot>/generated_zero_flowjf_s<step>.summary.json', 'artifacts/runs/baselines_bc_ladder/', 'research/tracks/ladder.md (SPRINT BEST ROUTE)')}</p>
<div class="grid wide">{''.join(video_card(v) for v in ORCBC_TRI if (VID / v[0]).exists())}</div>
<div class="grid">{''.join(video_card(v) for v in ORCBC_VIDEOS if (VID / v[0]).exists())}</div>
<div class="grid wide">{''.join(video_card(v) for v in R2_VIDEOS if (VID / v[0]).exists())}</div>""")
    L = "ladder_localize/{r}/bc_direct1701_u12000__{t}.json"
    if have(L.format(r="panda_pg2", t="jointfix")):
        rows = []
        tags = sorted({f.name[len("bc_direct1701_u12000__"):-5] for f in (RAW / "ladder_localize").glob("*/bc_direct1701_u12000__*.json")},
                      key=lambda t: (("__" in t), t != "jointfix", t))
        for t_ in tags:
            gen = "__" in t_
            lab = (f"system 0 {t_.split('__')[0]}, GENERATED packet from learned:ladder_flow_{t_.split('__')[1].replace('flowjf', 'jointfix@').replace('_s', '')}"
                   if gen else f"system 0 {t_}, oracle packet E(BC chunk)")
            cells = [esc(lab)]
            for r in ("panda_pg2", "parm6_tf3"):
                p = L.format(r=r, t=t_)
                if have(p):
                    d = J(p)["summary"]
                    e = d["sys0_gen_err_arm"] if gen else d["sys0_bcoracle_err_arm"]
                    x = (f' <span class="ci">|z_gen − z_bc| / |z_bc| = {d["z_gen_vs_bcoracle_rel"]:.2f}</span>'
                         if gen and d.get("z_gen_vs_bcoracle_rel") is not None else "")
                    cells.append(f'{e:.4f} / {d["hold_still_ref_arm"]:.4f} '
                                 f'<span class="ci">({e / d["hold_still_ref_arm"]:.0%} of hold-still)</span>{x}')
                else:
                    cells.append("—")
            rows.append(cells)
        parts.append("<h3>Stateless localization on BC-visited states (sprint_latent)</h3><details><summary>offline system-0 error table (click to expand)</summary>"
                     + table(["system 0", "panda_pg2 arm error / hold-still ref", "parm6_tf3 arm error / hold-still ref"], rows)
                     + "</details>" + f"""<p>No teacher state: BC drives the matched seeds; every 8 ticks the packet is E(the chunk BC actually executed
next) {badge('oracle', 'ORACLE DIAGNOSTIC')}, and system 0 is scored against BC's executed command (1-step, normalized).
The jointly trained system 0 realizes these packets below the hold-still error (a partial, not a precise, realization), and shadow-teacher DAgger
<i>raised</i> its error (to 83–133% of hold-still) (consistent with stale labels). Its first tick after each new packet is as bad as holding still.
<b>Mechanism found (sprint_latent):</b> system 0 largely copies the current joint <i>velocity</i> rather than following the packet. At
BC-visited states, zeroing only the joint-velocity input collapses its commanded step gain from 0.88 to 0.12 relative to BC. From rest this is
a fixed point, and it is the same class of proprioceptive shortcut as B-1. BC-expert DAgger penalizes it, which explains why the refits help. Removing
the velocity input alone (jfnoqd) also moves the stateless route from 0/30 to its row's value. A refit without the velocity input is under test.
{src('ladder_localize/bias/', 'research/tracks/ladder.md (SPRINT BEST ROUTE, 21:58)', 'D-056')}
The generated-packet row measures the generator gap at the same states: through system 0 the generated packet is
no better than holding still. So at this flow snapshot both stages fall short. {src(L.format(r='<robot>', t='<tag>'), 'research/tracks/ladder.md (sprint)')}</p>""")
    if not parts:
        return ""
    now = dt.datetime.now().strftime("%H:%M")
    return (f'<section id="sprint" class="update"><h2 style="border:0;margin-top:.3rem">Sprint update ({now} PDT)</h2>'
            + "".join(parts) + "</section>")



def sec_architecture():
    return """
<section id="arch"><h2>1 · Architecture</h2>
<p>One architecture is under test. System i (a flow model) generates a structured continuous <b>packet z</b>
[4 knots × assemblies × 64]. Semantic objectives supervise <i>that packet</i>. System 0 consumes the same packet together
with morphology, current proprioception/touch and phase, and emits native joint targets every 50 ms to a joint tracker.
Meaning must be shown by causal edits of the received packet, not by probes of hidden states.</p>
<div class="arch" role="img" aria-label="system II to system i to packet z to system 0 to tracker">
 <div class="blk dim"><div class="bt">System II</div><div class="bd">VLM (psi0) task grounding<br><span class="badge b-none">not shown</span></div></div>
 <div class="arr">→<small>task graph + bindings</small></div>
 <div class="blk"><div class="bt">System i</div><div class="bd">flow model, public observations → packet every 0.4 s (NFE 8)</div></div>
 <div class="arr">→</div>
 <div class="blk key"><div class="bt">packet z</div><div class="bd">knots × assemblies × 64<br>versioned: latent-space + bundle fingerprint, role-ordered slots, null slots</div></div>
 <div class="arr">→</div>
 <div class="blk"><div class="bt">System 0</div><div class="bd">realizer: z + morphology + proprio/touch + phase → native joint targets, 50 ms</div></div>
 <div class="arr">→</div>
 <div class="blk"><div class="bt">Tracker</div><div class="bd">joint-target servo in MuJoCo (the same for teacher and learned routes)</div></div>
</div>
<div class="arch2">
 <div><b>Stage A</b> trains encoder E (teacher's next 16 commands + context → z), system 0, and packet probes
 (held_by, acting_on, relative position, subtask, goal/observed effect). <b>Stage B</b> trains system i to generate E's
 packets from public observations.</div>
 <div><b>Routes compared on matched scenes (the ladder):</b> R0 teacher → tracker; R1 <span class="badge b-oracle">ORACLE</span>
 E(teacher's future actions) → system 0; R2 system i → system 0. R1 isolates system 0 from the generator.</div>
</div>
</section>"""


def sec_works():
    t = []
    for r in ("panda_pg2", "parm6_tf3"):
        d = J(f"ladder_v1/{r}/teacher_shadow_zero.summary.json")
        k1 = J(f"lead_teacher_reanchor/{r}_K8.json")
        t.append([esc(r), frac(d["success"], d["n"]), f'{d["track_q_rad"]:.3f} rad / {d["track_tcp_m"] * 100:.1f} cm',
                  frac(k1["success"], k1["n"]),
                  f'<code>ladder_v1/{r}/teacher_shadow_zero.summary.json</code>, <code>lead_teacher_reanchor/{r}_K8.json</code>'])
    single = table(["body", "teacher success (dev seeds)", "tracker lag q / TCP", "teacher re-anchored every 8 ticks", "raw"], t)

    sem = J("acceptance_semantic_teacher/semantic_summary_teacher.json")["summary"]
    srows = []
    for c, what, key in (("control", "cube placed in zone", "cube_in_zone"),
                         ("rebind_obj", "distractor0 lifted + placed, cube untouched", "distractor0_in_zone"),
                         ("goal_shift", "cube at the shifted goal", "cube_at_shifted_goal"),
                         ("irrelevant_distractor", "same as control", "cube_in_zone")):
        s = sem[c]
        srows.append([f"<code>{c}</code>", esc(what), frac(s[key], s["n"])])
    semt = table(["edit", "expected", "teacher"], srows)

    dual = {}
    for task in ("support_insert", "handover"):
        for r in JL(f"dualarm_teacher_ref/{task}.jsonl"):
            k = (task, r["pair"])
            a = dual.setdefault(k, [0, 0])
            if r.get("status") != "infeasible":
                a[1] += 1
                a[0] += bool(r.get("privileged_success"))
    pairs = sorted({p for _, p in dual})
    drows = [[esc(p) + (" <i>(held-out source pair)</i>" if p == "parm5l_pg2__parm5s_tf3" else ""),
              frac(*dual[("support_insert", p)], ci=False), frac(*dual[("handover", p)], ci=False)] for p in pairs]
    dualt = table(["arm pair (left__right)", "support_insert", "handover"], drows)

    lg = J("legged_vlm_teacher_ref/eval_dev.summary.json")["per_body"]
    lrows = [[esc(b), frac(v["success"], v["n"], ci=False), str(v["fell"])] for b, v in lg.items()]
    legt = table(["body", "success", "falls"], lrows)

    LAT = "artifacts/runs/lead_latency_final/flowjf20k_vs_direct18k.json"
    lat = J(LAT)
    pi = lat["paired_interleaved_nfe8"]
    return f"""
<section id="works"><h2>2 · What works today</h2>
<p class="lede">Everything in this section is the <b>scripted teacher</b> (privileged: it reads true object poses) or
infrastructure. It shows the tasks, bodies, scenes and edits are physically achievable and that the evaluation stack is
sound. <b>It is not evidence that a learned policy works.</b></p>
<h3>Single-arm pick_place {badge('teacher')}</h3>
{single}
<p>Matched dev scenes (first 30 feasible seeds from 3,000,000; 300 ticks). The tracker follows commands closely on every
route, so no failure below is a tracker failure. {src('D-044', 'D-049')}</p>
<h3>Semantic edits are achievable {badge('teacher')}</h3>
{semt}
{src('acceptance_semantic_teacher/semantic_summary_teacher.json', 'D-041')}
<h3>Dual-arm (M = 2 role-ordered assemblies) {badge('teacher')}</h3>
{dualt}
<p>Success / feasible, dev seeds 3,000,000–019. Plus the manipulator-assignment family: 2,480 identical-scene pairs,
2,203 valid (both assignments succeed and only the assigned arm holds the bar). {src('dualarm_teacher_ref/*.jsonl', 'D-043', 'research/pairs/assign_pick_place_v1.json')}</p>
<p>Legged and humanoid bodies: see §2b.</p>
<h3>Infrastructure {badge('ok')}</h3>
<ul>
<li>Stage A / Stage B training (standardized, resumable), system-0 runtime at 50 ms, closed-loop eval, disturbance test, labelled renders. {src('D-031')}</li>
<li>Packet contract: latent-space version = config hash + encoder-weights hash; a retrained bundle with the same config is rejected (unit test). {src('D-038')}</li>
<li>Ladder evaluator with matched scenes, shadow teacher labels and failure-stage localization. {src('D-044')}</li>
<li>Acceptance suite: valid semantic edits (rebind_obj, goal_shift) plus irrelevant-edit controls; teacher / oracle / generated routes. {src('D-041')}</li>
<li>Packet-policy GRPO with exact per-step likelihoods (unit-tested); on hold, no signal from a non-competent base. {src('D-042')}</li>
<li>Sealed-protocol baseline runner (direct actions, action-only codec) with exact resume. {src('D-035')}</li>
</ul>
<h3>Latency {badge('ok', 'within budget')}</h3>
<p>Final checkpoints, quiet GPU, {pi['latent_obs_to_first_command']['n']} interleaved pairs at NFE 8: the latent route (learned:ladder_flow_jointfix 20k → system 0)
observation → first native command p95 <b>{pi['latent_obs_to_first_command']['p95']:.1f} ms</b> vs the real B-1-fixed BC checkpoint
learned:direct1701_u18000 observation → chunk p95 <b>{pi['direct_obs_to_chunk']['p95']:.1f} ms</b> → overhead ratio <b>{pi['p95_overhead_ratio']:.3f}×</b>
(limit {pi['threshold_p95_ratio']}×). System-0 tick p95 {lat['system0_tick']['p95']:.1f} ms, {lat['system0_deadline_misses']} misses of the 50 ms deadline
({lat['system0_tick']['n']} ticks). {src(LAT, 'D-058')}<br>
<span class="muted">Supersedes the earlier 1.12× estimate (D-041), which used random direct-path weights on a loaded peer.</span></p>
<h3>Binding pair: identical scene, three different task bindings {badge('teacher')}</h3>
<div class="grid">{''.join(video_card(v) for v in VIDEOS if "paired" in v[0])}</div>
<p>Single-arm, dual-arm, legged and humanoid clips: §2b.</p>
</section>"""


LEGGED = ["go2", "anymal_c", "pquad4", "hexapod6_long", "hexapod6", "sprawl4", "sprawl8"]
HUMANOID = ["t1", "g1", "h1"]
ARMS = ["panda_pg2", "panda_tf3", "parm6_tf3", "parm6_pg2", "parm5s_tf3", "parm5l_pg2", "parm7_pg2", "parm7", "ur5e_pg2", "ur5e", "sawyer_pg2", "sawyer_tf3",
        "sawyer", "xarm7_pg2", "xarm7_tf3", "xarm7", "proc", "procedural"]


BODY_NOTES = {
    "parm7_pg2": "Weak teacher on this procedural arm: it grasps but never completes the place (0/2 seeds; only 2 seeds tried).",
    "xarm7_pg2": "Held-out TARGET body of the sealed protocol: teacher only; no learned policy was run on it.",
    "g1": "g1's frozen tracker supports only forward walking and arc turns: 18/20 with the arc_only teacher, 17/20 with 2 falls with the default teacher.",
    "h1": "Weakest humanoid: 14/20; it reaches both waypoints but drifts during the halt or times out.",
    "parm5s_tf3": "Procedural arm (held-out source body).",
}
MONTAGE = "2026-09-25_scripted_teacher_legged_montage_10bodies.mp4"


def body_clips():
    """Scripted-teacher clips from artifacts/video/INDEX.md, grouped by morphology (auto-picks up new sprint_bodies clips)."""
    import re
    import shutil
    idx = ROOT / "artifacts/video/INDEX.md"
    groups = {"arms": {}, "dual-arm": {}, "legged": {}, "humanoid": {}}
    if not idx.exists():
        return groups
    for line in idx.read_text().splitlines():
        m = re.match(r"- `([^`]+\.mp4)` — (.*)", line)
        if not m:
            continue
        f, desc = m.group(1), m.group(2)
        low = (f + " " + desc).lower()
        if "scripted_teacher" not in low or any(t in f for t in ("causal_edit", "semantic_edit", "triptych", "ladder_", "bc_semantic")):
            continue
        src_ = ROOT / "artifacts/video" / f
        if not src_.exists() or src_.stat().st_size > 2_500_000:
            continue
        name = f.lower()
        if "__" in name or "dual" in name:
            mm = re.search(r"teacher_(.+?)_([a-z0-9]+_[a-z0-9]+__[a-z0-9]+_[a-z0-9]+)_s\d", name)
            grp, body = "dual-arm", (f"{mm.group(1)} · {mm.group(2)}" if mm else name[:40])
        else:
            grp = body = None
            for g, names in (("humanoid", HUMANOID), ("legged", LEGGED), ("arms", ARMS)):
                for b_ in names:
                    if re.search(rf"(^|_){re.escape(b_)}(_|$)", name.replace(".mp4", "")):
                        grp, body = g, b_
                        break
                if grp:
                    break
            if not grp:
                continue
        groups[grp].setdefault(body, []).append((f, desc))
    return groups


def md_table_to_html(md: str) -> str:
    """Render the pipe tables and paragraphs of a track's result section (raw paths inside stay visible)."""
    import re
    out, rows = [], []

    def inline(t):
        t = esc(t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)

    def flush():
        if rows:
            out.append(table([inline(c) for c in rows[0]], [[inline(c) for c in r] for r in rows[1:]]))
            rows.clear()
    for line in md.splitlines():
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            rows.append(cells)
        else:
            flush()
            if line.strip() and not line.startswith("## "):
                out.append(f"<p>{inline(line.lstrip('#').strip())}</p>" if line.startswith("#") else inline(line) + " ")
    flush()
    return "".join(out)


def legged_research():
    p = ROOT / "research/tracks/legged_vlm.md"
    txt = p.read_text() if p.exists() else ""
    if "## LEGGED RESEARCH RESULT" not in txt:
        import subprocess
        try:
            txt = subprocess.run(["git", "-C", str(ROOT), "show", "origin/track/legged_vlm:research/tracks/legged_vlm.md"],
                                 capture_output=True, text=True, timeout=20).stdout
        except Exception:
            txt = ""
    i = txt.find("## LEGGED RESEARCH RESULT")
    head = f"<h3>Legged / humanoid research (agent legged; D-060)</h3>"
    if i < 0:
        return head + f"<p>{badge('run')} Legged research restarted (D-060): BC positive control, the ladder (teacher / BC / stateless oracle / R2) and packet edits on go2 → hexapod6 → t1/g1. Results pending.</p>"
    j = txt.find("\n## ", i + 5)
    sec = txt[i:j if j > 0 else None]
    USED.add("research/tracks/legged_vlm.md")
    return head + '<div class="mdsec">' + md_table_to_html(sec) + "</div>" + src("research/tracks/legged_vlm.md (LEGGED RESEARCH RESULT)", "D-060")


def sec_bodies():
    groups = body_clips()
    lg = J("legged_vlm_teacher_ref/eval_dev.summary.json")["per_body"]
    lrows = [[esc(b), frac(v["success"], v["n"], ci=False), str(v["fell"])] for b, v in lg.items()]
    legt = table(["legged body", "teacher success", "falls"], lrows)
    extra = ""
    BR = "artifacts/runs/sprint_bodies_teacher_ref"
    ref_files = sorted((ROOT / BR).glob("*.summary.json")) + sorted((RAW / "sprint_bodies_teacher_ref").glob("*.summary.json")) \
        if ((ROOT / BR).exists() or (RAW / "sprint_bodies_teacher_ref").exists()) else []
    for f in ref_files:
        if "legged_prior" in f.name:
            continue
        try:
            d = json.loads(f.read_text())
            rows = [[esc(b), frac(v.get("success", 0), v.get("n", 0), ci=False), str(v.get("fell", "")), esc(v.get("note", ""))]
                    for b, v in (d.get("per_body") or {}).items()]
            if rows:
                extra += f"<h3>Teacher reference, waypoint_contact, dev seeds 10000–10019 ({esc(f.name.replace('.summary.json', ''))}) {badge('teacher')}</h3>"
                extra += table(["body", "teacher success", "falls", "note"], rows) + src(str(f.relative_to(ROOT)) if str(f).startswith(str(ROOT / "artifacts")) else f.name)
        except Exception:
            pass
    html_ = []
    titles = {"arms": "Single arms (different kinematics and grippers)", "dual-arm": "Dual-arm (two role-ordered assemblies)",
              "legged": "Legged (one assembly per leg + body)", "humanoid": "Humanoids"}
    for g in ("arms", "dual-arm", "legged", "humanoid"):
        cards = []
        for b_, lst in sorted(groups[g].items()):
            plain = [c for c in lst if "paired" not in c[0]] or lst
            succ = [c for c in plain if "success" in c[0]]
            fail = [c for c in plain if "success" not in c[0]]
            pick = (succ[-1:] + fail[-2:]) if g in ("legged", "humanoid") else (succ[-1:] or fail[-1:]) + (fail[-1:] if succ else [])
            for f, desc in pick:
                if not (VID / f).exists():
                    import shutil
                    shutil.copy2(ROOT / "artifacts/video" / f, VID / f)
                outcome = "success" if "success" in f else ("FELL" if "fell" in f else "failure" if "fail" in f else "")
                note = BODY_NOTES.get(b_, "")
                cards.append((f, "teacher", f"{b_} · {outcome}".strip(" ·"), (note + " " if note else "") + desc[:260] + ("…" if len(desc) > 260 else ""),
                              "artifacts/video/INDEX.md"))
        body_html = ''.join(video_card(c) for c in cards) if cards else '<div class="novid">clips pending (sprint_bodies)</div>'
        html_.append(f"<h3>{titles[g]} {badge('teacher')}</h3><div class=\"grid\">{body_html}</div>")
        if g == "legged":
            html_.append(legt + f"<p>Teacher driving frozen body trackers, dev seeds 10000–10019. {src('legged_vlm_teacher_ref/eval_dev.summary.json', 'D-040')}</p>")
    research = legged_research()
    montage = ""
    if (ROOT / "artifacts/video" / MONTAGE).exists():
        if not (VID / MONTAGE).exists():
            import shutil
            shutil.copy2(ROOT / "artifacts/video" / MONTAGE, VID / MONTAGE)
        montage = '<div class="grid wide">' + video_card((MONTAGE, "teacher", "10 legged and humanoid bodies · montage",
            "Scripted teacher driving each body's frozen tracker on the waypoint_contact task (walk to A, walk to B, halt).",
            "artifacts/video/INDEX.md")) + "</div>"
    return f"""
<section id="bodies"><h2>2b · Bodies: morphology breadth</h2>
<p class="lede">The packet and system 0 are defined over a morphology graph, so the same interfaces cover arms with different
kinematics and grippers, two-arm pairs, legged robots and humanoids. <b>Honest scope:</b> every non-arm clip below is the
<b>scripted teacher driving a frozen tracker</b>. There is <b>no learned legged or humanoid model</b>, and dual-arm training was
deferred (D-040, D-043). Learned results on this page are single-arm pick_place only.</p>
{montage}
{extra}
{''.join(html_)}
{research}
</section>"""


def sec_matrix():
    rows = [
        ["single-arm pick_place (latent_sem)", badge("ok"),
         "R1 best: <b>1/30</b> (panda, jointfix + DAgger-1); 0/30 on parm6_tf3 " + src("D-048") + "; <b>confounded</b>: oracle packets are stale off the teacher trajectory (§4)",
         "0/30 (flow v2, pre-fix) " + src("D-044") + "; B-1-fixed flows " + badge("run"),
         "teacher does every edit; oracle: packet dependence, <b>no semantic control</b> " + src("D-041"),
         badge("none")],
        ["latent_nosem (control)", badge("ok"), "not evaluated after the fix", "binding v4 nosem " + badge("run"),
         "—", badge("none")],
        ["plain BC with fix (direct / codec)", badge("ok"), "n/a", "<b>25/30, 27/30</b> (direct); <b>28/30, 25/30</b> (codec) on panda / parm6, mid-training " + src("artifacts/runs/baselines_bc_ladder/"), "n/a", badge("none")],
        ["binding (object pairs)", badge("ok"), "v1 z does not carry the binding: focus_follows 0.0 " + src("binding_v1_reeval/sem_cf_probe_bindcf.json"),
         "binding v4 chains " + badge("run"), "not shown", badge("none")],
        ["dual-arm / assignment", badge("ok"), "teacher only", badge("none"), "pairs ready, teacher does both " + src("D-043"), badge("none")],
        ["legged / humanoid", "data + code", "teacher only", badge("none"), badge("none"), badge("none")],
        ["VLM system II", "smoke only", "n/a", badge("none"), badge("none"), badge("none")],
        ["latency", badge("ok"), "—", "p95 1.014× vs real BC checkpoint (≤ 1.25×), quiet GPU " + src("D-058"), "—", "—"],
    ]
    return f"""
<section id="matrix"><h2>3 · Evidence matrix</h2>
<p class="lede"><b>Headline: the latent-packet route is not competent in closed loop yet, and causal packet semantics are not shown.</b>
Plain BC with the same data and fix is competent on the same scenes (§6). The best latent oracle-route result is 1 success
in 30 seeds; the generated (deployable) latent route has no success after the B-1 fix yet.</p>
{table(["component", "implementation", "oracle-packet behaviour (diagnostic)", "generated-packet behaviour (deployable)", "semantic interventions", "held-out bodies"], rows, "matrix")}
<p>Full statement with every raw path: <code>research/reports/evidence_matrix.md</code>. Held-out bodies (xarm7_pg2, xarm7_tf3, panda_tf3) stay sealed until a source controller is competent.</p>
</section>"""


def sec_debug():
    t0 = J("ladder_smoke/t0_check_sem_panda.json")
    trows = []
    for k, lab in (("stored/t1", "stored (teacher's prev command in col 28), t = 1"),
                   ("stored/t5+", "stored, t ≥ 5"),
                   ("zero_prev/t1", "col 28 zeroed (= deployment), t = 1"),
                   ("zero_prev/t5+", "col 28 zeroed (= deployment), t ≥ 5")):
        if k in t0:
            d = t0[k]
            trows.append([esc(lab), f'{d["arm"]:.3f}', f'{d["arm_label_sq"]:.3f}', f'{d["grip"]:.3f}',
                          f'{d["grip_pred"]:.2f} / {d["grip_label"]:.2f}'])
    t0t = table(["system-0 input (training pack, panda_pg2)", "arm error", "arm hold-still ref", "gripper error", "gripper pred / label"], trows)
    lt = table(["rung", "system 0", "panda_pg2", "panda re-anchored", "parm6_tf3", "parm6 re-anchored",
                "failed stage (panda, re-anch. if run)", "min TCP–cube"], ladder_rows(), "ladder")
    inv = []
    for r in ("panda_pg2", "parm6_tf3"):
        for t in ("jointfix", "jfdag1"):
            p = f"ladder_v1/{r}/oracle_zero_{t}_reanchor_replan1.summary.json"
            if have(p):
                d = J(p)
                inv.append(f"{r} {t}: {d['success']}/{d['n']}")
    import collections
    sh = []
    for r in ("panda_pg2", "parm6_tf3"):
        p = f"artifacts/runs/baselines_bc_ladder/{r}/learned_direct1701_u12000.jsonl"
        if have(p):
            rows_ = [x for x in JL(p) if x["privileged_success"]]
            c = collections.Counter(x.get("final_teacher_phase") for x in rows_)
            sh.append(f"{r}: of {len(rows_)} BC successes the shadow teacher's final phase is "
                      + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    shadow = esc("; ".join(sh)) or "(raw rows not available)"
    srcbc = src("artifacts/runs/baselines_bc_ladder/<robot>/learned_direct1701_u12000.jsonl (final_teacher_phase)",
                "research/tracks/baselines.md")
    ka = [J(f"lead_teacher_reanchor/{r}_K1.json") for r in ("panda_pg2", "parm6_tf3")]
    kb = [J(f"lead_teacher_reanchor/{r}_K8.json") for r in ("panda_pg2", "parm6_tf3")]
    return f"""
<section id="debug"><h2>4 · The debugging story</h2>
<h3>Bug B-1: system 0 learned to copy the teacher's previous command</h3>
<p>Node features carry a previous-action slot (column 28). The datasets stored the <i>teacher's</i> previous command there;
an earlier fix zeroed column 2 by mistake. At deployment the column is always 0. So Stage-A system 0 learned a crutch that
does not exist at test time: on the training pack itself, zeroing the column makes it no better than holding still on the
arm, and wrong on the gripper. Every learned closed-loop number before the fix (acceptance, GRPO, flow dev evals) is
dominated by B-1 and says nothing about the architecture. {src('D-044', 'D-045')}</p>
{t0t}
{src('ladder_smoke/t0_check_sem_panda.json')}
<h3>The failure-localization ladder</h3>
<p>Matched dev scenes, 30 seeds each, deployment input (prev-action = 0), replan every 8 ticks.
<span class="badge b-oracle">R1</span> uses a packet encoded from the teacher's future actions: it was meant to ask whether
<i>system 0</i> can realize a good packet at all (but see the confound below). Wilson 95% intervals in brackets.</p>
{lt}
<p>{src('ladder_v1/<robot>/<tag>.summary.json', 'D-044', 'D-046', 'D-047', 'D-048')}. Rows marked n/a were not run with
re-anchoring. Refit rows use the frozen v1 encoder; "jointfix" retrains encoder and system 0 together with the fix.</p>
<h3>D-049 and after: is the oracle route a valid test?</h3>
<p>Re-anchoring the oracle expert changes the teacher whose actions are encoded. Check with the teacher itself driving:
no re-anchoring and re-anchoring every 8 ticks both give {kb[0]['success']}/{kb[0]['n']} (panda) and {kb[1]['success']}/{kb[1]['n']} (parm6);
re-anchoring <i>every tick</i> gives {ka[0]['success']}/{ka[0]['n']} and {ka[1]['success']}/{ka[1]['n']}, stuck in pregrasp.
So the every-8-ticks R1 route (the table above) is valid, and the replan-every-tick R1 runs
(<span class="muted">{esc('; '.join(inv))}</span>) are <b>withdrawn</b>. {src('lead_teacher_reanchor/*_K1.json', 'lead_teacher_reanchor/*_K8.json', 'D-049')}</p>
<p><b>Why R1 is confounded anyway (sprint_bc).</b> The shadow teacher is stateful and is not a valid expert on
learner-visited states. In the competent BC episodes of §6 it still ends far behind the task even when BC succeeds:
{shadow}. So the R1 oracle packet (which encodes that teacher's look-ahead) and the shadow-DAgger labels can say “hover at
pregrasp” while the task is actually progressing. R1's 0–1/30 therefore cannot separate “system 0 cannot realize packets”
from “the oracle packets are stale off the teacher's trajectory”. {srcbc}</p>
<p><b>Observation, not a cause:</b> in the valid re-anchored R1 runs the tool's median closest approach is 4–6 cm from the
cube, not the pregrasp waypoint 13 cm above it; system 0 heads for the object while the (possibly stale) oracle packet
encodes “go to pregrasp”. {src('D-047', 'D-049')}</p>
<div class="grid">{''.join(video_card(v) for v in LADDER_VIDEOS[:3])}</div>
</section>"""


def sec_semantic():
    te = J("acceptance_semantic_teacher/semantic_summary_teacher.json")["summary"]
    orc = J("acceptance_semantic_sem_v1rep/semantic_summary_oracle.json")["summary"]
    rows = []
    for c in ("control", "rebind_obj", "goal_shift", "irrelevant_distractor", "orthogonal_matched"):
        a, b = te.get(c), orc.get(c)
        f = lambda s: (f'<b>{s["followed"]}/{s["n"]}</b> as expected · cube lifted {s["cube_lifted"]} · '
                       f'distractor0 lifted {s["distractor0_lifted"]}') if s else "n/a"
        rows.append([f"<code>{c}</code>", f(a), f(b)])
    semt = table(["edit", f"teacher {badge('teacher', 'scripted_teacher')}", f"oracle route (latent_sem_v1, pre-fix) {badge('oracle', 'ORACLE')}"], rows)
    w = J("artifacts/runs/acceptance_causal_sem_v1i/window_summary.json")["window"]["conditions"]
    cm = lambda v, k: (f'{v[k]["mean"] * 100:.2f} [{v[k]["lo"] * 100:.2f}, {v[k]["hi"] * 100:.2f}]'
                       if k in v and v[k]["mean"] is not None else "—")
    wr = [[f"<code>{c}</code>", cm(w[c], "tcp_shift_norm_m"), cm(w[c], "tcp_shift_along_edit_m")]
          for c in ("control_replay", "zero", "shuffle", "rel+x", "rel-x", "rel+y", "cf+x", "cf-x", "focus_swap", "rand") if c in w]
    wt = table(["packet edit (8 ticks, window protocol)", "TCP shift vs control, cm [95% CI]", "shift along the edit, cm"], wr)
    return f"""
<section id="semantic"><h2>5 · Semantic interventions</h2>
<p class="lede">An intervention edits only the context the packet is generated for (which object is bound, where the goal is);
the physical scene is unchanged and system 0 is frozen. Irrelevant edits are the control. Zero/shuffle sensitivity and
probe accuracy do <b>not</b> count as semantic control.</p>
{semt}
<p>{src('acceptance_semantic_teacher/semantic_summary_teacher.json', 'acceptance_semantic_sem_v1rep/semantic_summary_oracle.json', 'D-041')}
The teacher performs every edit. The oracle route (pre-fix model) never grasps. Graded signal: rebinding moves the arm
8.4 [6.8, 9.7] cm from control vs 1.8 / 0.9 cm for the irrelevant edits, but toward the <i>original</i> cube
(−1.3 cm): packet dependence, not semantic control.</p>
<h3>Packet edits on a generator (learned:flow_latent_sem_v1 interrupted, pre-fix) {badge('learned', 'learned:flow_latent_sem_v1i')}</h3>
{wt}
<p>{src('artifacts/runs/acceptance_causal_sem_v1i/window_summary.json')} System 0 depends strongly on the packet (zero packet
5.85 cm vs control-replay exactly 0), but directed 5 cm geometry edits move the tool by fractions of a millimetre along
the edit. <b>No semantic control is shown.</b> Reruns on the B-1-fixed bundles:
see the <a href="#sprint">sprint update</a> at the top (B-1-fixed oracle, BC reference, teacher paired scenes).</p>
<div class="grid">{video_card(LADDER_VIDEOS[3])}</div>
</section>"""


def sec_bc():
    tl = HOST_RUNS / "latent_slice1_b1fix/baseline_direct_action/seed1701/source/train_log.jsonl"
    prog = ""
    if tl.exists():
        last = json.loads(tl.read_text().splitlines()[-1])
        prog = (f'Direct-action source training at build time: update {last["step"]:,} of ~26.3k. '
                f'{src("host: artifacts/runs/latent_slice1_b1fix/baseline_direct_action/seed1701/source/train_log.jsonl")}')
    rows = []
    import re as _re2
    bctags = sorted({f.name[len("learned_"):-len(".summary.json")] for f in (ROOT / "artifacts/runs/baselines_bc_ladder").glob("*/learned_*.summary.json")},
                    key=lambda t: (t.split("_u")[0], int(_re2.sub(r"\D", "", t.split("_u")[-1]) or 10**9)))
    for tag, what in ((t, ("direct-action BC" if t.startswith("direct") else "action-only codec BC") + (f", {int(_re2.sub(r'\D', '', t.split('_u')[-1])):,} updates" if _re2.sub(r'\D', '', t.split('_u')[-1]) else ", final checkpoint"))
                      for t in bctags):
        cells = []
        for r in ("panda_pg2", "parm6_tf3"):
            p = f"artifacts/runs/baselines_bc_ladder/{r}/learned_{tag}.summary.json"
            if have(p):
                d = J(p)
                st = ", ".join(f"{k} {v}" for k, v in d.get("failed_stage", {}).items() if k != "success")
                cells += [frac(d["success"], d["n"]), esc(st or "—")]
            else:
                cells += ["—", "—"]
        rows.append([f'<span class="badge b-bc">learned:{tag}</span><br><span class="muted" style="font-size:.8em">{what}</span>'] + cells)
    for r_, lab, kind in (("teacher_shadow_zero", "R0 scripted teacher", "teacher"),
                          ("oracle_zero_jfdag1_reanchor", "R1 oracle, best latent (jfdag1)", "oracle")):
        cells = []
        for r in ("panda_pg2", "parm6_tf3"):
            d = J(f"ladder_v1/{r}/{r_}.summary.json")
            cells += [frac(d["success"], d["n"]), esc(", ".join(f"{k} {v}" for k, v in d["failed_stage"].items() if k != "success") or "—")]
        rows.append([f'<span class="badge b-{kind}">{lab}</span>'] + cells)
    t = table(["controller", "panda_pg2", "failures (stage)", "parm6_tf3", "failures (stage)"], rows)
    ho = ""
    HO = [("artifacts/runs/baselines_bc_ladder/heldout/direct1701_u12000.summary.json", "learned:direct1701_u12000 (direct BC, 12k updates)"),
          ("artifacts/runs/latent_slice1_b1fix/baseline_action_only_codec/seed1701/eval/source.summary.json", "learned:codec1701 final (action-only codec BC, 26.3k updates)"),
          ("artifacts/runs/latent_slice1_b1fix/baseline_direct_action/seed1701/eval/source.summary.json", "learned:direct1701 final (direct BC)")]
    hr = []
    for hp, lab in HO:
        if not have(hp):
            continue
        h = J(hp)
        bodies = [b_ for b_ in h if not b_.startswith("_")]
        k = sum(h[b_]["successes"] for b_ in bodies)
        n = sum(h[b_]["attempted"] for b_ in bodies)
        hr.append([f'<span class="badge b-bc">{esc(lab)}</span>']
                  + [frac(h[b_]["successes"], h[b_]["attempted"]) for b_ in ("parm5s_tf3", "parm5l_pg2")]
                  + [frac(k, n), f"<code>{esc(hp)}</code>"])
    if hr:
        ho = ("<h3>Held-out source bodies (not in BC training)</h3>"
              + table(["controller", "parm5s_tf3", "parm5l_pg2", "pooled", "raw"], hr)
              + "<p>Protocol source-competence harness, seeds 2,000,000+, 50 episodes per body, infeasible scenes excluded. These are "
              "held-out <i>source</i> bodies, not the sealed target bodies (xarm7_pg2, xarm7_tf3, panda_tf3).</p>")
    zs = []
    for m, lab in (("baseline_direct_action", "learned:direct1701 final (direct BC)"), ("baseline_action_only_codec", "learned:codec1701 final (codec BC)")):
        cells = [f'<span class="badge b-bc">{esc(lab)}</span>']
        any_ = False
        for t in ("panda_tf3", "xarm7_pg2", "xarm7_tf3"):
            zp = f"artifacts/runs/latent_slice1_b1fix/{m}/seed1701/eval/{t}_b0.summary.json"
            if have(zp):
                d = J(zp)[t]
                cells.append(frac(d["successes"], d["attempted"]))
                any_ = True
            else:
                cells.append("—")
        if any_:
            zs.append(cells)
    if zs:
        ho += ("<h3>Zero-shot transfer to the sealed protocol's NEW bodies (budget 0: no target data)</h3>"
               + table(["controller", "panda_tf3 (known arm, new gripper)", "xarm7_pg2 (new arm)", "xarm7_tf3 (new arm + gripper)"], zs)
               + "<p>100 episodes per cell, seeds 2,000,000+. Plain BC transfers across a new gripper pairing but not to an unseen arm "
               "kinematic chain without target data: that is the gap the sealed latent-vs-baseline comparison is meant to test. "
               "No latent method has been run on these bodies (sealed until a source controller is competent). "
               + src("artifacts/runs/latent_slice1_b1fix/<method>/seed1701/eval/<target>_b0.summary.json",
                     "research/reports/latent_slice1_b1fix_baselines_tables.md", "D-064") + "</p>")
    vids = "".join(video_card(v) for v in BC_VIDEOS)
    return f"""
<section id="bc"><h2>6 · Positive control: plain behaviour cloning with the fix {badge('ok', 'competent')}</h2>
<div class="grid wide">{''.join(video_card(v) for v in TRIPTYCH)}</div>
<p class="lede"><b>Plain behaviour cloning with deployment-consistent input (B-1 fixed) is a competent controller on the
ladder's matched scenes, already at mid-training.</b> Same data, same seed (1701), same 30 dev seeds, same tracker and
privileged evaluator, prev-action input 0 as deployed. So the latent route's closed-loop failure comes from the latent
packet route (Stage A encoder / system 0 and/or the generator), <i>not</i> from the data, the demonstrations or the simulator.</p>
{t}
<p>Wilson 95% in brackets. BC failures are late (grasp / lift / transport / place timeouts), none at approach.
{src('artifacts/runs/baselines_bc_ladder/<robot>/learned_<tag>.summary.json', 'research/tracks/baselines.md (SPRINT BC RESULT)')}
panda_pg2 and parm6_tf3 are source-<i>training</i> bodies (as for the ladder). {prog}</p>
{ho}
<div class="grid">{vids}</div>
</section>"""


def sec_next():
    return """
<section id="next"><h2>7 · What is next</h2>
<ol>
<li><b>The clean test: R2 vs BC on the same seeds.</b> System i's own packet from a B-1-fixed flow → system 0,
compared with the competent plain-BC control (§6) on the ladder's matched scenes. No teacher is in the loop, so the
oracle confound does not apply. If R2 fails where BC succeeds, the gap is in the packet route itself.</li>
<li><b>Generated route on the fixed bundle</b>: flows trained with <code>zero_prev_action</code> on the jointly trained
and binding-v4 encoders, then R2 on matched seeds.</li>
<li><b>Semantic interventions on a competent route</b>: rebind_obj, goal_shift, manipulator assignment (dual-arm pairs),
each with irrelevant-edit controls; sem vs capacity-matched nosem.</li>
<li><b>Only then</b> the sealed four-way comparison on held-out bodies (xarm7_pg2, xarm7_tf3, panda_tf3) with the
≤ 1.25× latency constraint rerun on final checkpoints.</li>
</ol>
</section>"""


CSS = """
:root{--bg:#fbfbf9;--fg:#1d1f23;--mut:#5d6470;--card:#ffffff;--line:#e2e4e8;--acc:#2f5bd3;--code:#f1f2f4;
--teach:#1f7a4d;--teach-bg:#e3f4ea;--orc:#8a5a00;--orc-bg:#fdf0d5;--lrn:#2f5bd3;--lrn-bg:#e3eafc;--bc:#6b3fb5;--bc-bg:#eee6fa;
--none:#6b7280;--none-bg:#eceef1;--run:#0f6e80;--run-bg:#dff3f6;--ok:#1f7a4d;--ok-bg:#e3f4ea;--fail:#b3261e;--fail-bg:#fbe4e2}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#15171a;--fg:#e6e7ea;--mut:#9aa1ad;--card:#1d2024;--line:#30343a;--acc:#8fb0ff;--code:#262a30;
--teach:#7fd6a6;--teach-bg:#173325;--orc:#f2c265;--orc-bg:#3a2d10;--lrn:#9db7ff;--lrn-bg:#1d2947;--bc:#c7a8f5;--bc-bg:#2e2342;
--none:#aab0ba;--none-bg:#2a2e34;--run:#7dd3e0;--run-bg:#133a41;--ok:#7fd6a6;--ok-bg:#173325;--fail:#ff9b92;--fail-bg:#43201d}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#15171a;--fg:#e6e7ea;--mut:#9aa1ad;--card:#1d2024;--line:#30343a;--acc:#8fb0ff;--code:#262a30;
--teach:#7fd6a6;--teach-bg:#173325;--orc:#f2c265;--orc-bg:#3a2d10;--lrn:#9db7ff;--lrn-bg:#1d2947;--bc:#c7a8f5;--bc-bg:#2e2342;
--none:#aab0ba;--none-bg:#2a2e34;--run:#7dd3e0;--run-bg:#133a41;--ok:#7fd6a6;--ok-bg:#173325;--fail:#ff9b92;--fail-bg:#43201d}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
header{border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:8px}
h1{font-size:1.6rem;margin:.2rem 0}h2{font-size:1.3rem;margin:2rem 0 .5rem;padding-top:.5rem;border-top:1px solid var(--line)}
h3{font-size:1.05rem;margin:1.3rem 0 .4rem}
p,li{max-width:75ch}.lede{font-size:1.02rem}.muted,.ci{color:var(--mut)}.ci{font-size:.85em;white-space:nowrap}
code{background:var(--code);padding:0 .25em;border-radius:3px;font-size:.86em;overflow-wrap:anywhere}
nav{display:flex;flex-wrap:wrap;gap:.3rem .8rem;font-size:.9rem}nav a{color:var(--acc);text-decoration:none}
.src{display:inline;color:var(--mut);font-size:.82em}.did{font-weight:600;color:var(--mut)}
.badge{display:inline-block;padding:.05em .5em;border-radius:999px;font-size:.78em;font-weight:600;white-space:nowrap;border:1px solid transparent}
.b-teacher{color:var(--teach);background:var(--teach-bg)}.b-oracle{color:var(--orc);background:var(--orc-bg)}
.b-learned{color:var(--lrn);background:var(--lrn-bg)}.b-bc{color:var(--bc);background:var(--bc-bg)}
.b-none{color:var(--none);background:var(--none-bg)}.b-run{color:var(--run);background:var(--run-bg)}
.b-ok{color:var(--ok);background:var(--ok-bg)}.b-fail{color:var(--fail);background:var(--fail-bg)}
.tw{overflow-x:auto;margin:.5rem 0;border:1px solid var(--line);border-radius:6px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{padding:.4rem .55rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;background:var(--code)}tr:last-child td{border-bottom:0}
table.ladder td:nth-child(n+3):nth-child(-n+6){white-space:nowrap}
.arch{display:flex;align-items:stretch;gap:.4rem;margin:1rem 0;flex-wrap:nowrap}
.blk{flex:1 1 0;min-width:0;border:1.5px solid var(--line);border-radius:8px;background:var(--card);padding:.55rem}
.blk.key{border-color:var(--acc)}.blk.dim{opacity:.75;border-style:dashed}
.bt{font-weight:700}.bd{font-size:.82rem;color:var(--mut)}
.arr{align-self:center;font-size:1.3rem;color:var(--mut);text-align:center;flex:0 0 auto}.arr small{display:block;font-size:.65rem;max-width:5em}
.arch2{display:grid;grid-template-columns:1fr 1fr;gap:1rem;font-size:.92rem}
.grid.wide{grid-template-columns:1fr}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;margin:1rem 0}
figure.vid{margin:0;border:1px solid var(--line);border-radius:8px;background:var(--card);overflow:hidden}
figure.vid video{width:100%;display:block;background:#000;aspect-ratio:4/3}.grid.wide figure.vid video{aspect-ratio:auto;max-height:420px}
figcaption{padding:.5rem .6rem;font-size:.84rem;overflow-wrap:anywhere}.novid{padding:2rem;text-align:center;color:var(--mut)}
.update{border:1.5px solid var(--acc);border-radius:8px;padding:.6rem .9rem;background:var(--card)}
#theme{float:right;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:.25rem .6rem;cursor:pointer}
.legend{display:flex;flex-wrap:wrap;gap:.4rem;margin:.5rem 0}
@media (max-width:720px){.arch{flex-direction:column}.arr{transform:rotate(90deg);padding:0}.arr small{display:none}
.arch2{grid-template-columns:1fr}.grid{grid-template-columns:1fr}h1{font-size:1.35rem}}
"""

JS = """
(function(){var b=document.getElementById('theme'),r=document.documentElement;
function cur(){return r.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light')}
b.onclick=function(){r.dataset.theme=cur()==='dark'?'light':'dark';try{localStorage.setItem('rrp-theme',r.dataset.theme)}catch(e){}};
try{var s=localStorage.getItem('rrp-theme');if(s)r.dataset.theme=s}catch(e){}})();
"""


def build(updates_html: str = ""):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    bcs = [J(str(f.relative_to(ROOT)))["success"] for f in (ROOT / "artifacts/runs/baselines_bc_ladder").glob("*/learned_*.summary.json")
           if "heldout" not in str(f)]
    bc_lo, bc_hi = min(bcs), max(bcs)
    r2 = []
    for f in (RAW / "ladder_v1").glob("*/generated_zero_*.summary.json"):
        if "fresh" in f.name or "v2s24543" in f.name:
            continue
        d = J(str(f.relative_to(RAW)))
        r2.append((d["success"], f.parent.name, f.name[len("generated_zero_"):-len(".summary.json")], d["n"]))

    def best_of(glob_pat, root=RAW / "ladder_v1"):
        best = {}
        for f in root.glob(glob_pat):
            d = json.loads(f.read_text())
            r = f.parent.name
            tag = f.name.replace(".summary.json", "")
            if "fresh" in tag or "v2s24543" in tag:   # other seed set / pre-fix flow
                continue
            if r not in best or d["success"] > best[r][0]:
                best[r] = (d["success"], d["n"], tag)
        return best
    sc_rows = []
    for lab, kind, bst, note in (
            ("R0 scripted teacher (privileged)", "teacher", best_of("*/teacher_shadow_zero.summary.json"), "reference"),
            ("plain BC, B-1 fixed (learned, best checkpoint)", "bc", best_of("*/learned_*.summary.json", ROOT / "artifacts/runs/baselines_bc_ladder"), "positive control"),
            ("latent: stateless oracle packet → system 0 (diagnostic)", "oracle", best_of("*/oracle_zero_*_orcbc.summary.json"), "not deployable"),
            ("latent: R2 generated packet → system 0 (deployable)", "learned", best_of("*/generated_zero_*.summary.json"), "the architecture test")):
        cells = [f'<span class="badge b-{kind}">{esc(lab)}</span>']
        for r in ("panda_pg2", "parm6_tf3"):
            if r in bst:
                k, n, tag = bst[r]
                cells.append(f'{frac(k, n)}<br><span class="ci">{esc(tag.replace("learned_", "").replace("oracle_zero_", "").replace("generated_zero_", "").replace("teacher_shadow_zero", "teacher"))}</span>')
            else:
                cells.append("—")
        cells.append(esc(note))
        sc_rows.append(cells)
    scoreboard = ("<h3 style=\"margin-top:.8rem\">Scoreboard: pick_place on the 30 matched dev scenes (best checkpoint per row)</h3>"
                  + table(["controller", "panda_pg2", "parm6_tf3", "role"], sc_rows)
                  + '<p class="muted" style="font-size:.85em">Best-of selections across checkpoints and variants are optimistic for every row alike; each '
                  'cell names the run it comes from, and per-run tables follow below. Semantic control of the packet: <b>not shown</b> (D-059, D-062).</p>')
    r2_best = "0/30 on every snapshot" if not r2 or max(r2)[0] == 0 else \
        "{}/{} ({}, {}; still far below BC)".format(max(r2)[0], max(r2)[3], max(r2)[1], max(r2)[2])
    body = (sec_architecture() + sec_works() + sec_bodies() + sec_matrix() + sec_debug() + sec_semantic() + sec_bc() + sec_next())
    used = "".join(f"<li><code>{esc(p)}</code></li>" for p in sorted(USED))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Structured Latent Packets</title>
<meta name="description" content="Honest demo of a controller-facing semantic latent packet for robot manipulation: what works, what fails, and why.">
<style>{CSS}</style></head><body><main>
<header><button id="theme" type="button" aria-label="toggle light/dark theme">◐ theme</button>
<h1>Structured latent packets for cross-body robot control — demo</h1>
<p class="muted">Built {now} PDT from saved raw outputs by <code>scripts/demo/build_page.py</code>. Research state, not a product.</p>
<div class="legend">Controller-source labels:
{badge('teacher')} {badge('oracle')} {badge('learned', 'learned:<ckpt>')} {badge('bc', 'learned BC baseline')} {badge('none')} {badge('run')}</div>
<nav><a href="#sprint">sprint update</a><a href="#arch">architecture</a><a href="#works">what works</a><a href="#bodies">bodies</a><a href="#matrix">evidence</a><a href="#debug">debugging</a>
<a href="#semantic">semantic edits</a><a href="#bc">BC control</a><a href="#next">next</a><a href="#sources">sources</a></nav>
</header>
<p class="lede"><b>Bottom line.</b> The scripted teacher solves every task and edit shown here on single-arm, dual-arm and legged
bodies, and the pipeline runs end to end within the latency budget (p95 overhead 1.014×, D-058). <b>Plain behaviour cloning on the same data is competent</b> ({bc_lo}–{bc_hi} of 30 on matched scenes across mid-training checkpoints), so data and
evaluation are sound. <b>The latent-packet route is not competent yet</b>: its best oracle-diagnostic variant succeeds 1 time
in 30, and causal packet semantics are not shown. We found and fixed a
train/deploy mismatch (bug B-1). The latent route's remaining failure is <b>not localized yet</b>: the oracle-packet
diagnostic turned out to be confounded (§4). In the clean test, the generated route against BC on the same seeds, the latent
route's best result so far is {r2_best} (sprint update). A stateless oracle and a generator-gap measurement show that both system 0 (underfit; the first gate) and the generator
(at the current flow snapshot) fall short (D-052).</p>
{scoreboard}
{updates_html}
{sec_sprint()}
{body}
<section id="sources"><h2>Sources</h2>
<p>Decision ids refer to <code>research/decisions.md</code>. Raw paths are relative to <code>artifacts/runs/</code> on the
shared peer store (small copies under <code>docs/demo/raw/artifacts/runs/</code>) unless they start with <code>artifacts/</code> (git-tracked).
Videos: <code>docs/demo/video/</code>; index lines in <code>artifacts/video/INDEX.md</code>.</p>
<details><summary>raw files read to build this page ({len(USED)})</summary><ul>{used}</ul></details>
</section>
</main><script>{JS}</script></body></html>"""
    OUT.write_text(page)
    # artifact contract: no doctype/html/head/body; <title> then <style>, then content and script
    head_end = page.index("</head><body>")
    title = page[page.index("<title>"):page.index("</title>") + 8]
    style = page[page.index("<style>"):page.index("</style>") + 8]
    content = page[head_end + len("</head><body>"):page.rindex("</body>")]
    art = title + "\n" + style + "\n" + content + "\n"
    for tag in ("<!doctype", "<html", "<head>", "<head ", "<body", "</body>", "</html>", "</head>"):
        assert tag not in art.lower(), tag
    (OUT.parent / "artifact.html").write_text(art)
    for v in VID.glob("*.mp4"):
        if f"video/{v.name}" not in page:
            v.unlink()
    print(OUT, len(page), "bytes;", len(USED), "raw files; artifact.html", len(art), "bytes")


if __name__ == "__main__":
    upd = ROOT / "docs/demo/updates.html"
    build(upd.read_text() if upd.exists() else "")
