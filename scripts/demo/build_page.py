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
        lab = "learned:flow_latent_sem_v2@22k" if "semantic_edit" not in f else "learned (deployable): flow_jointfix@20k → sys-0 gendag1_noqd"
        for key, flow in (("flowgdag1_", "flow_jointfix_gdag1"), ("flowjfft_", "flow_jointfix_ft"), ("flowjf20k_", "flow_jointfix@20k"), ("flowjf_s", "flow_jointfix@")):
            if key in f:
                import re as _rr
                rest = _rr.sub(r"_(success|fell|failure.*)$", "", f.split(key)[1].split(".")[0].replace("_cpu", ""))
                lab = (f"learned:{flow}{rest}" if key == "flowjf_s" else f"learned:{flow} → sys-0 {rest}")
                break
        if "r2triptych" in f:
            lab = "teacher | BC | " + lab
    if "_legged_" in f and kind in ("bc", "oracle", "learned"):
        lab = {"bc": "learned: legged BC", "oracle": "ORACLE diagnostic (legged)", "learned": "learned (legged)"}[kind]
    elif kind == "bc":
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


GEN_SEM_VIDEOS = [
    ("2026-09-26_semantic_edit_goal_shift_learned_ladder_flow_jointfix_parm6_tf3_k3000052.mp4", "learned",
     "DEPLOYABLE route · goal edit · parm6_tf3 · left unedited, right edited",
     "Only the BELIEF of the goal zone moves 12 cm (no marker is drawn there). The cube ends at the new goal, 12 cm from the green zone. Selected clear seed; pooled aggregate 23/80 vs ≤1/80 per control (D-075).",
     "artifacts/runs/acceptance_sprint_sem_gen_jf_parm6/semantic_summary_generated.json"),
    ("2026-09-26_semantic_edit_rebind_desc_learned_ladder_flow_jointfix_parm6_tf3_k3000052.mp4", "learned",
     "DEPLOYABLE route · binding edit (descriptor only) · parm6_tf3",
     "The task entity's descriptor now names the other cube; the arm approaches and touches the NEW cube (pooled first approach 60/80 vs 0/80). The rebound task is rarely completed (new cube lifted 7/80).",
     "artifacts/runs/acceptance_sprint_sem_gen_jf_parm6/semantic_summary_generated.json"),
    ("2026-09-26_semantic_edit_orthogonal_matched_learned_ladder_flow_jointfix_parm6_tf3_k3000052.mp4", "learned",
     "DEPLOYABLE route · CONTROL: probe-orthogonal packet edit of matched norm · parm6_tf3",
     "Same seed; behaviour is unchanged (approaches and touches the original cube).",
     "artifacts/runs/acceptance_sprint_sem_gen_jf_parm6/semantic_summary_generated.json"),
]

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
    ("2026-09-25_r2triptych_panda_pg2_s3000012_teacher_bc-direct1701_u12000_generated-flowgdag1_gendag3noqd_cpu.mp4", "learned",
     "same scene · teacher | plain BC u12000 | R2 best recipe (flow gdag1 → system 0 gendag3_noqd) · panda_pg2 · seed 3000012",
     "This render: teacher success; BC (12k-update checkpoint) fails at grasp; the deployable latent route SUCCEEDS. Best recipe: 10/30 panda, 22/30 parm6 "
     "(D-070). Re-rendered with the model on CPU; of 4 re-rendered evaluation-success seeds of this recipe, 2 succeeded again (panda 3000012, parm6 3000019).",
     "ladder_v1/panda_pg2/generated_zero_flowgdag1_rzgendag3_noqd.summary.json"),
    ("2026-09-25_r2triptych_parm6_tf3_s3000019_teacher_bc-direct1701_u12000_generated-flowgdag1_gendag3noqd_cpu.mp4", "learned",
     "same scene · teacher | plain BC u12000 | R2 best recipe · parm6_tf3 · seed 3000019",
     "This render: all three succeed (R2 on CPU; teacher and BC panels from an earlier GPU render of the same seed).",
     "ladder_v1/parm6_tf3/generated_zero_flowgdag1_rzgendag3_noqd.summary.json"),
    ("2026-09-25_ladder_generated_parm6_tf3_s3000011_flowjfft_gendag2noqd_cpu_success.mp4", "learned",
     "R2 generated · learned:ladder_flow_jointfix_ft → system 0 gendag2_noqd · parm6_tf3 · seed 3000011 · SUCCESS",
     "The deployable latent route (system i's own packets; no teacher or BC in the loop) on the current best parm6 recipe (16/30). "
     "Re-rendering 5 of its evaluation-success seeds on CPU reproduced 3 successes (3000011, 3000015, 3000016); 3000010 failed at lift, 3000017 at approach.",
     "ladder_v1/parm6_tf3/generated_zero_flowjfft10k_rzgendag2noqd.summary.json"),
    ("2026-09-25_ladder_generated_parm6_tf3_s3000015_flowjfft_gendag2noqd_cpu_success.mp4", "learned",
     "R2 generated · same recipe · parm6_tf3 · seed 3000015 · SUCCESS", "Second reproduced success.",
     "ladder_v1/parm6_tf3/generated_zero_flowjfft10k_rzgendag2noqd.summary.json"),
    ("2026-09-25_ladder_generated_parm6_tf3_s3000010_flowjfft_gendag2noqd_cpu_failure-lift.mp4", "learned",
     "R2 generated · same recipe · parm6_tf3 · seed 3000010 · failure (lift) in this render",
     "Its evaluation row succeeded; this re-render fails at lift. Single episodes vary.",
     "ladder_v1/parm6_tf3/generated_zero_flowjfft10k_rzgendag2noqd.summary.json"),
    ("2026-09-25_ladder_generated_panda_pg2_s3000013_flowjf20k_gendag2noqd_cpu_success.mp4", "learned",
     "R2 generated · learned:ladder_flow_jointfix final → system 0 gendag2_noqd · panda_pg2 · seed 3000013 · SUCCESS",
     "Best panda recipe (6/30). Re-rendering 4 of its evaluation-success seeds reproduced 1 success; the others failed at grasp, lift or transport.",
     "ladder_v1/panda_pg2/generated_zero_ladder_flow_jointfix_snap_final_s20000_rzgendag2noqd.summary.json"),
    ("2026-09-25_ladder_generated_panda_pg2_s3000009_flowjf20k_gendag2noqd_cpu_failure-lift.mp4", "learned",
     "R2 generated · same recipe · panda_pg2 · seed 3000009 · failure (lift) in this render",
     "Its evaluation row succeeded.", "ladder_v1/panda_pg2/generated_zero_ladder_flow_jointfix_snap_final_s20000_rzgendag2noqd.summary.json"),
    ("2026-09-25_r2triptych_panda_pg2_s3000000_teacher_bc-direct1701_u12000_generated-flowjf20k_gendag2noqd_cpu.mp4", "learned",
     "same scene · teacher | plain BC u12000 | R2 (flow final → gendag2_noqd) · panda_pg2 · seed 3000000 · CPU render",
     "This render: teacher success; BC (12k-update checkpoint) fails at grasp; R2 carries the cube and fails at place.",
     "artifacts/runs/demo_video/r2_panda_pg2_3000000/INDEX.md"),
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


def recipe_sets(r, recipe):
    """All evaluations of an R2 recipe on robot r: {'dev': d, '3000100': d, ...}."""
    import re
    out = {}
    for f in (RAW / "ladder_v1" / r).glob(f"generated_zero_{recipe}*.summary.json"):
        t = f.name[len(f"generated_zero_{recipe}"):-len(".summary.json")]
        m = re.fullmatch(r"(?:_(?:fresh|s)(3000\d00))?", t)
        if not m:
            continue
        key = "dev" if (m.group(1) in (None, "3000000")) else m.group(1)
        out[key] = J(str(f.relative_to(RAW)))
    return out


FROZEN = [("flowgdag2h_rzgendag3_noqd", "flow_jointfix_gdag2h → system 0 gendag3_noqd", "02:30 (current)"),
          ("flowgdag1_rzgendag3_noqd", "flow_jointfix_gdag1 → system 0 gendag3_noqd", "01:10 (superseded)")]


def sec_final_route():
    """SPRINT BEST ROUTE FINAL (frozen by sprint_latent), rendered from the raw summaries it names."""
    robots = ("panda_pg2", "parm6_tf3", "parm5s_tf3", "parm5l_pg2")
    def one(r, t):
        p = f"ladder_v1/{r}/{t}.summary.json"
        return J(p) if have(p) else None
    rows = []
    def add(lab, kind, vals):
        cells = [f'<span class="badge b-{kind}">{lab}</span>'] + [frac(d["success"], d["n"]) if d else "—" for d in vals]
        if any(c != "—" for c in cells[1:]):
            rows.append(cells)
    add("R0 scripted_teacher (privileged), dev", "teacher", [one(r, "teacher_shadow_zero") or one(r, "teacher_heldout_ref") for r in robots])
    bc = lambda r, suf="": one(r, f"learned_bc_direct1701_u12000{suf}") or (J(f"artifacts/runs/baselines_bc_ladder/{r}/learned_direct1701_u12000.summary.json") if suf == "" and have(f"artifacts/runs/baselines_bc_ladder/{r}/learned_direct1701_u12000.summary.json") else None)
    add("plain BC learned:direct1701_u12000, dev", "bc", [bc(r) for r in robots])
    add("R1 ORACLE (stateless): E(BC chunk) → system 0 gendag3_noqd, dev", "oracle", [one(r, "oracle_zero_gendag3noqd_orcbc") for r in robots])
    for recipe, lab, when in FROZEN:
        sets = {r: recipe_sets(r, recipe) for r in robots}
        keys = sorted({k for v in sets.values() for k in v}, key=lambda k: (k != "dev", k))
        for k in keys:
            if k != "dev":
                add(f"plain BC learned:direct1701_u12000, fresh seeds {k}+", "bc", [one(r, f"learned_bc_direct1701_u12000_fresh{k}") for r in robots])
            add(("<b>" if "current" in when else "") + f"R2 {lab}, {'dev' if k == 'dev' else 'fresh seeds ' + k + '+'} · frozen {when}" + ("</b>" if "current" in when else ""),
                "learned", [sets[r].get(k) for r in robots])
        pool = []
        for r in robots:
            ds = list(sets[r].values())
            pool.append((frac(sum(d["success"] for d in ds), sum(d["n"] for d in ds)) + f'<br><span class="ci">{len(ds)} seed set(s)</span>') if ds else "—")
        rows.append([f'<span class="badge b-learned">R2 {esc(lab)} pooled over ALL its seed sets · {when}</span>'] + pool)
    gd = sorted((RAW / "ladder_dagger_gdag2").glob("generated_*.summary.json"))
    tb = ""
    if gd:
        k = sum(json.loads(f.read_text())["success"] for f in gd)
        n = sum(json.loads(f.read_text())["n"] for f in gd)
        USED.update(str(f.relative_to(RAW)) for f in gd)
        tb = (f"<p>R2 flow_gdag1 on the {len(gd)} source-<i>training</i> bodies (24 seeds each, seeds 4,000,000+): <b>{k}/{n}</b> (these rollouts were also "
              f"collected as DAgger training data for the next round; the evaluated checkpoint had not trained on them). {src('ladder_dagger_gdag2/generated_<robot>.summary.json')}</p>")
    return f"""<h3>FINAL best latent route (frozen by sprint_latent): the deployable route works, below BC</h3>
{table(["controller", "panda_pg2", "parm6_tf3", "parm5s_tf3 (held-out source body)", "parm5l_pg2 (held-out source body)"], rows)}
{tb}
<p>Every evaluation of the frozen checkpoints on non-training seeds is listed, better or worse, for the current and the superseded freeze. BC rows on the same
seed sets are shown for a like-for-like comparison. No teacher, oracle or BC at run time on the R2 rows. Current freeze: system i
<code>ladder_flow_jointfix_gdag2h</code> (generator-DAgger round 2) and system 0 + encoder <code>ladder_rz_jointfix_gendag3_noqd/representation.pt</code>
(sha256 f60cde41…). The route is <b>distilled through a learned stateless expert</b>: system-0 DAgger labels and flow targets come from the learned BC policy (trained on
the same scripted-teacher demonstrations), not from the scripted teacher; the extra training seeds (3.2M–4.1M) are disjoint from every evaluation set (D-080). What made it work, all within the architecture: (1) removing a joint-velocity shortcut in system 0; (2) replacing the stale teacher
FSM with the stateless BC expert; (3) system-0 DAgger rounds, including states visited with system i's own packets, plus z-noise; (4) generator DAgger.
Not solved: panda grasp/lift and parm6 place. The ordering is R2 &lt; BC. Binding-v4 sem/nosem bundles are not competent with the same recipe yet, so no
deployable sem-vs-nosem comparison exists on the arm.
{src('research/tracks/ladder.md (SPRINT BEST ROUTE FINAL)', 'ladder_v1/<robot>/generated_zero_<recipe>[_s|_fresh<seed>].summary.json', 'D-070', 'D-072', 'D-080')}</p>
<div class="grid wide">{"".join(video_card(v) for v in R2_VIDEOS[:2] if (VID / v[0]).exists())}</div>"""


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
        gen_goal = gen_ctl = gen_orig = gen_orig_c = gen_appr = gen_pref = gen_newl = gen_n = pan_rb = "—"
        PAN = "artifacts/runs/acceptance_sprint_sem_gen_jf_panda/semantic_summary_generated.json"
        if have(PAN):
            pp_ = J(PAN)["summary"]
            pan_rb = f'{fc(pp_, "rebind_desc")} vs {fc(pp_, "control")} unedited'

        GEN = "artifacts/runs/acceptance_sprint_sem_gen_jf_parm6/semantic_summary_generated.json"
        GENP = "artifacts/runs/acceptance_sprint_sem_gen_jf_parm6_ext/semantic_summary_generated_pooled.json"
        GENX = "artifacts/runs/acceptance_sprint_sem_gen_jf_parm6_ext/semantic_summary_generated_ext.json"
        if have(GENP):
            GEN = GENP
        gen_ext = ""
        if have(GENX) and have("artifacts/runs/acceptance_sprint_sem_gen_jf_parm6/semantic_summary_generated.json"):
            g0 = J("artifacts/runs/acceptance_sprint_sem_gen_jf_parm6/semantic_summary_generated.json")["summary"]["goal_shift"]
            gx = J(GENX)["summary"]["goal_shift"]
            gen_ext = (f" The goal effect is smaller on the {gx['n']} replication seeds ({gx['cube_at_shifted_goal']}/{gx['n']}) than on the first "
                       f"{g0['n']} ({g0['cube_at_shifted_goal']}/{g0['n']}) (D-075).")
        if have(GEN):
            g2 = J(GEN)["summary"]
            n_ = g2["rebind_desc"]["n"]
            gen_goal = f'{g2["goal_shift"]["cube_at_shifted_goal"]}/{g2["goal_shift"]["n"]}'
            gen_ctl = "–".join(sorted({f'{g2[c]["cube_at_shifted_goal"]}' for c in ("irrelevant_distractor", "orthogonal_matched", "control_replay")})) + f'/{g2["control"]["n"]}'
            gen_orig, gen_orig_c = f'{g2["rebind_desc"]["cube_lifted"]}/{n_}', f'{g2["control"]["cube_lifted"]}/{g2["control"]["n"]}'
            gen_appr = f'{g2["rebind_desc"]["approached_first_new_frac"]["k"]}/{n_}'
            gen_pref = f'{g2["_contrasts"]["rebind_desc-irrelevant_distractor:pref_min"]["mean"]:+.2f}'
            gen_newl = f'{g2["rebind_desc"]["distractor0_lifted"]}/{n_}'
            gen_n = str(n_)
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
        for GEN, setlab in (("artifacts/runs/acceptance_sprint_sem_gen_jf_panda/semantic_summary_generated.json", "PANDA_PG2 (route not competent: approach-level only)"),
                            ("artifacts/runs/acceptance_sprint_sem_gen_jf_parm6_ext/semantic_summary_generated_ext.json", "replication, 39 new seeds"),
                            ("artifacts/runs/acceptance_sprint_sem_gen_jf_parm6/semantic_summary_generated.json", "first 41 seeds"),
                            ("artifacts/runs/acceptance_sprint_sem_gen_jf_parm6_ext/semantic_summary_generated_pooled.json", "pooled 80 seeds")):
          if have(GEN):
            gg = J(GEN)["summary"]
            rows.insert(0, [f'{badge("learned", "DEPLOYABLE: learned flow_jointfix@20k → system 0 gendag1_noqd")}<br>{"panda_pg2" if "PANDA" in setlab else "parm6_tf3"}, canonical scenes, {setlab.replace("PANDA_PG2 ", "")}',
                            "binding (descriptor only)",
                            f'<b>{fc(gg, "rebind_desc")}</b> / {fc(gg, "control")}<br><span class="ci">first approach new {gg["rebind_desc"]["approached_first_new_frac"]["k"]}/{gg["rebind_desc"]["n"]}; original cube lifted {gg["rebind_desc"]["cube_lifted"]}/{gg["rebind_desc"]["n"]} vs {gg["control"]["cube_lifted"]}/{gg["control"]["n"]} unedited; new cube lifted {gg["rebind_desc"]["distractor0_lifted"]}</span>',
                            ci(gg["_contrasts"].get("rebind_desc-irrelevant_distractor:pref_min")),
                            f'<b>{gg["goal_shift"]["cube_at_shifted_goal"]}/{gg["goal_shift"]["n"]}</b><br><span class="ci">irrelevant {gg["irrelevant_distractor"]["cube_at_shifted_goal"]}/{gg["irrelevant_distractor"]["n"]}, orthogonal {gg["orthogonal_matched"]["cube_at_shifted_goal"]}/{gg["orthogonal_matched"]["n"]}, noise replay {gg["control_replay"]["cube_at_shifted_goal"]}/{gg["control_replay"]["n"]}</span>',
                            "n/a"])
        rows.append([f'{badge("learned", "learned: binding v4 sem / nosem flows")}', "generated route", "not competent with the sprint recipe (R2 ≤ 1/30) " + src("D-080"), "—", "—", "—"])
        parts.append("<h3>Semantic interventions at the level each route reaches (sprint_semantic)</h3>"
                     + table(["route", "edit type", "rebind: first touch on the NEW object (edit / control)",
                              "rebind effect beyond the matched irrelevant edit, min-distance preference (m) [95% CI]",
                              "goal edit: cube at the new goal", "arm swap: edited-to arm touches first"], rows)
                     + f"""<p>Reading: the edits and metrics are valid (the teacher follows them). The oracle packet makes system 0
approach the rebound object, but that packet encodes the teacher's demonstration toward it, so this is weak evidence
(system 0 reads packet content, not the binding). The competent BC controller follows a goal edit and a swap of object
<i>beliefs</i>, but <b>ignores a pure binding change</b>: that is the capability the semantic packet is meant to add (see D-074 below: the deployable latent route does redirect its approach). The final BC reference (D-065) follows goal edits (23/24, 16/17) and object-belief swaps
(23/24, 15/17), leaves irrelevant edits unchanged, and ignores descriptor-only rebinds: the latent packet must beat this reference, especially on binding,
to support the semantic claim. <b>Arm central-claim result (D-074, replicated in D-075; deployable route, parm6_tf3, {gen_n} seeds pooled): task semantics in the context steer behaviour through the
generated packet, beyond matched controls.</b> A goal edit puts the cube at the new goal in {gen_goal}, against {gen_ctl} for three matched controls.{gen_ext}
A binding edit (only the task entity's descriptor changes) means the original cube is never lifted ({gen_orig} vs {gen_orig_c} unedited). The
arm first approaches the new cube in {gen_appr}, with closest approach {gen_pref} m toward it. <b>Plain BC ignores the same rebinding
(0/32)</b>, so this is the first place the latent route does something the direct-action baseline does not. <b>Caveats:</b> one body (panda is
not competent on this route), and the new cube is lifted in only {gen_newl}, so the rebound task is rarely completed. The flow may read the
binding through public predicate estimates that follow it. There is no nosem counterpart, so the role of semantic supervision is not isolated. On panda_pg2, where this route never transports, the binding edit also redirects the approach: first touch on the new cube {pan_rb} (D-077);
the goal edit cannot be tested there.
{src(GEN if have(GEN) else "D-074", "D-074", "D-075")}</p><div class="grid3">{"".join(video_card(v) for v in GEN_SEM_VIDEOS if (VID / v[0]).exists())}</div><p><b>Best latent route: system 0 causally executes goal content carried in the packet (oracle diagnostic, D-062).</b>
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
        if not fl or "ladder_ckpts" in fl or "grpo_base" in fl or "flow_latent_sem_v" in fl:   # pre-fix flows
            continue
        import re as _rq2
        _fm = _rq2.search(r"_(?:fresh|s)(3000[1-9]00)\.summary", f.name)
        fresh = " · FRESH seeds " + _fm.group(1) + "+ (not the matched set)" if _fm else ""
        import re as _re3
        _st = _re_step(fl)
        if _st < 0:
            _m = _re3.search(r"(\d+)k_rz", f.name)
            _st = int(_m.group(1)) * 1000 if _m else -1
        step = (Path(fl).parent.name.replace("ladder_flow_", "").replace("ladder_", ""), _st)
        s0 = Path(rep).parent.name.replace("ladder_rz_jointfix_", "").replace("ladder_latent_sem_b1fix_anchor", "jointfix").replace("ladder_rz_", "")
        r2rows.setdefault((step, s0, fresh), {})[f.parent.name] = d
    if r2rows:
        rows = []
        for (step, s0, fresh), per in sorted(r2rows.items(), key=lambda kv: (bool(kv[0][2]), -sum(d["success"] for d in kv[1].values()))):
            desc0 = {"gendag1": " (as gendag1_noqd but WITH the joint-velocity input)", "gendag2_noqd": " (round 2 of gendag1_noqd: DAgger on generated-packet states, no joint-velocity input)", "gendag3_noqd": " (round 3 of the same recipe)", "gendag1_qdd": " (gendag1 variant with joint-velocity dropout)", "gendag1_noqd": " (refit from bcdag2: no joint-velocity input, BC-expert DAgger incl. states visited with GENERATED packets, z-noise 0.3; config configs/ladder/rz_jointfix_gendag1_noqd.json on track/ladder)"}.get(s0, "")
            cells = [f'<span class="badge b-learned">R2 learned:flow_{step[0]}{"@" + str(step[1]) if step[1] >= 0 else " (final)"}</span> → system 0 {esc(s0)}<b>{esc(fresh)}</b><span class="ci">{esc(desc0)}</span>']
            for r in ("panda_pg2", "parm6_tf3"):
                d = per.get(r)
                if d:
                    cells += [frac(d["success"], d["n"]),
                              esc(", ".join(f"{kk} {v}" for kk, v in d["failed_stage"].items() if kk != "success"))
                              + f' <span class="ci">(min TCP–cube {d["min_tcp_cube_m"] * 100:.1f} cm)</span>']
                else:
                    cells += ["—", "—"]
            rows.append(cells)
        r2_rows_, rows = rows, []
        known = {"jointfix": "jointfix", "jfdag1": "jfdag1 (shadow DAgger r1)", "jfdag2df08": "jfdag2df08 (shadow DAgger r1+r2)",
                 "jfbcdag1": "jfbcdag1 (BC-expert DAgger)", "jfbcdag1long": "jfbcdag1long (BC-expert DAgger, 16k steps, lr 3e-4)", "jfnoqd": "jfnoqd (no joint-velocity input)", "jfbcdag2": "jfbcdag2 (BC-expert DAgger round 2)", "jfbcdag3": "jfbcdag3 (BC-expert DAgger round 3)", "jfbig16k": "jfbig16k (larger system 0, 16k steps, BC-expert DAgger)", "jfnoqd": "jfnoqd (no joint-velocity input)", "bindv4sem": "binding v4 SEM bundle", "bindv4sembcdag1": "binding v4 SEM + BC-expert DAgger r1", "bindv4nosembcdag1": "binding v4 NOSEM + BC-expert DAgger r1", "bindv4nosem": "binding v4 NOSEM bundle (capacity-matched control)"}
        found = sorted({f.name[len("oracle_zero_"):-len("_orcbc.summary.json")] for f in (RAW / "ladder_v1").glob("*/oracle_zero_*_orcbc.summary.json")},
                       key=lambda t: (list(known).index(t) if t in known else 99, t))
        for t_, lab in ((t, known.get(t, t)) for t in found):
            ps = [f"ladder_v1/{r}/oracle_zero_{t_}_orcbc.summary.json" for r in ("panda_pg2", "parm6_tf3")]
            if not any(have(p) for p in ps):
                continue
            cells = [f'<span class="badge b-oracle">R1 ORACLE, stateless: E(BC chunk) → system 0 {esc(lab)}</span>']
            tot = 0
            for p in ps:
                if have(p):
                    d = J(p)
                    tot += d["success"]
                    cells += [frac(d["success"], d["n"]),
                              esc(", ".join(f"{k} {v}" for k, v in d["failed_stage"].items() if k != "success"))]
                else:
                    cells += ["—", "—"]
            rows.append((tot, cells))
        orc_rows_ = [c for _, c in sorted(rows, key=lambda x: -x[0])]
        rows = []
        for tag, lab in (("direct1701_ufinal", "learned:direct1701 final (plain BC, same seeds)"), ("direct1701_u12000", "learned:direct1701_u12000 (plain BC mid-training, same seeds)")):
            if not have(f"artifacts/runs/baselines_bc_ladder/panda_pg2/learned_{tag}.summary.json"):
                continue
            cells = [f'<span class="badge b-bc">{lab}</span>']
            for r in ("panda_pg2", "parm6_tf3"):
                d = J(f"artifacts/runs/baselines_bc_ladder/{r}/learned_{tag}.summary.json")
                cells += [frac(d["success"], d["n"]), esc(", ".join(f"{k} {v}" for k, v in d["failed_stage"].items() if k != "success"))]
            rows.append(cells)
        hdr = ["controller", "panda_pg2", "failures (stage)", "parm6_tf3", "failures (stage)"]
        main_rows = rows + r2_rows_[:4] + orc_rows_[:2]
        rest = r2_rows_[4:] + orc_rows_[2:]
        rows = main_rows
        more = (f"<details><summary>all {len(rest)} other R2 / stateless-R1 rows, best first (click to expand)</summary>" + table(hdr, rest) + "</details>") if rest else ""
        parts.insert(0, "<h3>Best latent route so far vs BC (sprint_latent, SPRINT BEST ROUTE): R2 generated and stateless R1 on the B-1-fixed bundle</h3>"
                     + table(hdr, rows) + more
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
<div class="grid wide">{''.join(video_card(v) for v in R2_VIDEOS[2:] if (VID / v[0]).exists())}</div>""")
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
    parts.insert(0, sec_final_route())
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
    if "## LEGGED RESEARCH RESULT" not in txt and "## RESEARCH RESTART" not in txt:
        import subprocess
        try:
            txt = subprocess.run(["git", "-C", str(ROOT), "show", "origin/track/legged_vlm:research/tracks/legged_vlm.md"],
                                 capture_output=True, text=True, timeout=20).stdout
        except Exception:
            txt = ""
    i = txt.find("## LEGGED RESEARCH RESULT")
    if i < 0:
        i = txt.find("## RESEARCH RESTART")
    head = f"<h3>Legged / humanoid research (agent legged; D-060)</h3>"
    if i < 0:
        return head + f"<p>{badge('run')} Legged research restarted (D-060): BC positive control, the ladder (teacher / BC / stateless oracle / R2) and packet edits on go2 → hexapod6 → t1/g1. Results pending.</p>"
    j = txt.find("\n## ", i + 5)
    sec = txt[i:j if j > 0 else None]
    USED.add("research/tracks/legged_vlm.md")
    lab = "LEGGED RESEARCH RESULT" if sec.startswith("## LEGGED RESEARCH RESULT") else "RESEARCH RESTART, live state"
    import re as _r
    import shutil as _sh
    cards = []
    idx = ROOT / "artifacts/video/INDEX.md"
    for line in (idx.read_text().splitlines() if idx.exists() else []):
        m = _r.match(r"- `(\d{4}-\d\d-\d\d_legged_[^`]+\.mp4)` — (.*)", line)
        if not m:
            continue
        f, desc = m.group(1), m.group(2)
        if (ROOT / "artifacts/video" / f).exists() and (ROOT / "artifacts/video" / f).stat().st_size < 2_500_000:
            if "ctxedit" in f or f.endswith("s10010_unedited-5s.mp4"):
                continue
            kind = "bc" if "learned-bc" in f else "oracle" if "oracle" in f else "learned"
            if not (VID / f).exists():
                _sh.copy2(ROOT / "artifacts/video" / f, VID / f)
            cards.append((f, kind, f.replace("2026-09-25_legged_", "").replace(".mp4", "").replace("_", " "), desc[:300], "artifacts/video/INDEX.md"))
    vids_ = ('<div class="grid">' + "".join(video_card(c) for c in cards) + "</div>") if cards else ""
    import shutil as _sh
    CTX = [("2026-09-26_legged_learned-R2_sem_go2_s10010_unedited-5s.mp4", "learned", "unedited task context",
            "R2 deployable route (sem), go2 seed 10010, t = 0–5 s, no edit.", "artifacts/video/INDEX.md"),
           ("2026-09-26_legged_learned-R2_sem_go2_s10010_ctxedit-mirror_inactive-5s.mp4", "learned", "IRRELEVANT edit: inactive waypoint mirrored",
            "Control: from t = 2 s the packet is generated from a context in which only the inactive waypoint is mirrored. The path is essentially unchanged.", "artifacts/video/INDEX.md"),
           ("2026-09-26_legged_learned-R2_sem_go2_s10010_ctxedit-mirror_active-5s.mp4", "learned", "VALID edit: active waypoint mirrored",
            "From t = 2 s system i generates every packet from a context with the active waypoint mirrored, and the robot veers toward the mirrored goal. This seed is illustrative, not typical (see the suite means).", "artifacts/video/INDEX.md")]
    for c in CTX:
        if (ROOT / "artifacts/video" / c[0]).exists() and not (VID / c[0]).exists():
            _sh.copy2(ROOT / "artifacts/video" / c[0], VID / c[0])
    ctx_html = '<div class="grid3">' + "".join(video_card(c) for c in CTX if (VID / c[0]).exists()) + "</div>"
    ME = {v: J(f"artifacts/runs/legged_edits/go2/r2ctx_{v}_snap_s4000/mirror_effects.json") for v in ("sem", "nosem")
          if have(f"artifacts/runs/legged_edits/go2/r2ctx_{v}_snap_s4000/mirror_effects.json")}
    def me(v, k):
        if v not in ME or k not in ME[v]:
            return "—"
        m, lo, hi = ME[v][k]["toward_mirror_lateral_m"]
        return f'{m:+.2f} m <span class="ci">[{lo:+.2f}, {hi:+.2f}]</span>'
    me_m = lambda v: f'{ME[v]["mirror_active"]["toward_mirror_lateral_m"][0]:.2f}' if v in ME else "—"
    me_tab = table(["task-context edit (R2 deployable, go2, 19 seeds, t = 2–5 s)", "sem: lateral move toward the mirrored side", "nosem"],
                   [[esc(lab), me("sem", k), me("nosem", k)] for k, lab in (("mirror_active", "VALID: mirror the active waypoint"),
                                                                             ("mirror_goal", "VALID: mirror both waypoints"),
                                                                             ("mirror_inactive", "IRRELEVANT: mirror only the inactive waypoint"))]) if ME else ""
    d071 = f"""<p><b>Central claim, shown on one body for one semantic (D-071).</b> On go2's <b>deployable</b> route, editing only the task context
(mirroring the active waypoint) moves the robot <b>{me_m('sem')} m (sem) / {me_m('nosem')} m (nosem)</b> toward the mirrored side. The irrelevant control (mirroring the
inactive waypoint) moves it 0.04–0.05 m. So meaning in the supplied task context flows task → packet → behaviour for goal direction.
<b>Honest limits:</b> halt via the task context does not work (out of distribution: halts only occur at the final waypoint in the data).
Semantic supervision is not needed (nosem is at least as steerable), and the only sem-specific handle, a goal readout, is small (6–7 cm, 3–5× random).
This is one body and one semantic. {src('artifacts/runs/legged_edits/go2/r2ctx_{sem,nosem}_snap_s4000/mirror_effects.json', 'D-071')}</p>{me_tab}{ctx_html}"""
    reading = f"""<div class="update">{d071}<p><b>Legged headline (go2, from the legged agent's notes).</b> The <b>deployable latent route is competent on go2</b>:
system i's own packets → system 0 give nosem 30/30 and sem 29/30 on the 30 matched dev seeds, against plain BC 30/30 and the teacher 30/30
{badge('learned', 'learned:legged_flow_{sem,nosem}_go2_v2 snap_s4000')}. Unlike the arm, the legged system 0 is not the bottleneck: the stateless oracle route
gives nosem 30/30 and sem 25/30.
<b>hexapod6 is the second body through the deployable route: R2 sem 30/30, nosem 30/30</b> (BC 30/30, stateless oracle 30/30 for both; D-076).
<b>t1 humanoid is the third body through the deployable route, for nosem only: R2 nosem 26–27/30, at least BC's 24/30; sem 6–13/30</b> (D-079).
The oracle diagnostic was not predictive on t1: there the gap went the other way (stateless R1 sem 12/30 → 18/30 after one identical DAgger round, nosem 0/30),
and it reverses on the deployable route. <b>g1 humanoid: the positive control fails</b> (BC 2–7/30 across replan settings vs the arc-only teacher 25/30), so
no latent claim is made there.
<b>Semantic supervision, legged:</b> on hexapod6 the sem packet has more editable probe handles (halt −0.19 m forward vs nosem +0.03 m, no stop; turn ±0.6 gives
0.11 rad vs 0.03–0.07 rad; random edits ≈0), but task-context goal steering is equal (+0.165 vs +0.168 m) and success is equal. The sem advantage is therefore in
the packet's editable handles, not in context-to-behaviour control. {src('artifacts/runs/legged_edits/hexapod6/', 'D-079')}
{src('research/tracks/legged_vlm.md', 'D-070', 'artifacts/runs/legged_ladder/go2/r2_*_snap_s4000.jsonl')}</p>
<p><b>Packet edits (D-069, oracle route, go2, 20 seeds):</b> probe-guided halt changes forward progress by −1.24 m (sem) and −1.23 m (nosem), against −0.05 to
−0.19 m for random edits of matched norm. Yaw ±0.6 edits give sign-correct turns of 0.16–0.30 rad, against ≈0 for random edits. Goal-mirror is null and
per-leg contact edits are weak. This shows <b>packet → behaviour causality along probe directions on an oracle route</b>: the edit is applied to z, not to
the context. It does not show that system i puts the right semantics into the packet. <b>Semantic supervision shows no advantage</b>: nosem is as good or
better. {src('D-069')}</p></div>"""
    return head + f"<p>{badge('run') if 'RESTART' in lab else ''} The table below is rendered from the legged agent's track notes ({lab}); every row names its controller source.</p>" + reading + '<details><summary>legged agent\'s full live table (click to expand)</summary><div class="mdsec">' + md_table_to_html(sec) + "</div></details>" + src(f"research/tracks/legged_vlm.md ({lab})", "D-060") + vids_


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
kinematics and grippers, two-arm pairs, legged robots and humanoids. <b>Honest scope:</b> every clip in the body grids below is the
<b>scripted teacher driving a frozen tracker</b>. Learned legged results (go2 BC, oracle and deployable latent routes; BC on hexapod6
and t1) are in the research block at the end of this section and are labelled there. No dual-arm learned model exists yet
(dual-arm training was deferred, D-043), and g1 has no competent BC control.</p>
{montage}
{extra}
{''.join(html_)}
{research}
</section>"""


BEST = {}


def sec_matrix():
    def b(kind, r):
        v = BEST.get(kind, {}).get(r)
        return f"{v[0]}/{v[1]}" if v else "—"
    orc = f"stateless oracle best: <b>{b('oracle', 'panda_pg2')}</b> panda, <b>{b('oracle', 'parm6_tf3')}</b> parm6 (diagnostic) " + src("D-053", "D-055")
    gen = f"R2 best: <b>{b('learned', 'panda_pg2')}</b> panda, <b>{b('learned', 'parm6_tf3')}</b> parm6 vs BC {b('bc', 'panda_pg2')}, {b('bc', 'parm6_tf3')} " + src("D-063")
    rows = [
        ["single-arm pick_place (latent, B-1 fixed)", badge("ok"), orc, gen,
         "deployable route (parm6, 80 seeds): goal edit 23/80 vs ≤1/80; binding edit redirects the approach (BC ignores it) but the rebound task is rarely completed (D-074, D-075); no nosem counterpart, so <b>semantic supervision's role not isolated</b> " + src("D-059", "D-062", "D-074"),
         badge("none")],
        ["latent_nosem (control)", badge("ok"), "binding v4 nosem: see sprint rows", "v4 flows " + badge("run"), "no sem advantage over nosem " + src("D-059"), badge("none")],
        ["plain BC with fix (direct / codec)", badge("ok"), "n/a", f"<b>{b('bc', 'panda_pg2')}, {b('bc', 'parm6_tf3')}</b> (final) " + src("D-065"), "follows goal and belief edits, ignores binding " + src("D-065"), "zero-shot: panda_tf3 78–85/100, xarm7 0/100 " + src("D-064")],
        ["binding (object pairs)", badge("ok"), "v1 z does not carry the binding: focus_follows 0.0 " + src("binding_v1_reeval/sem_cf_probe_bindcf.json"),
         "binding v4 flows " + badge("run"), "not shown", badge("none")],
        ["dual-arm / assignment", badge("ok"), "teacher only", badge("none"), "pairs ready, teacher does both; v4 arm edits: no detectable effect " + src("D-043"), badge("none")],
        ["legged / humanoid", "verified on go2 (D-060)", "go2 stateless oracle: nosem 30/30, sem 25/30", "<b>go2 R2: nosem 30/30, sem 29/30</b> vs BC 30/30; hexapod6 R2 30/30 both; t1 R2 nosem 26–27/30 (≥ BC 24/30), sem 6–13/30 (D-079) " + src("research/tracks/legged_vlm.md"), "probe-direction halt/turn edits causal (go2, hexapod6); sem has stronger handles on hexapod6 but equal context control; mixed overall " + src("D-069", "D-079"), badge("none")],
        ["VLM system II", "smoke only", "n/a", badge("none"), badge("none"), badge("none")],
        ["latency", badge("ok"), "—", "p95 1.014× vs real BC checkpoint (≤ 1.25×) " + src("D-058"), "—", "—"],
    ]
    return f"""
<section id="matrix"><h2>3 · Evidence matrix</h2>
<p class="lede"><b>Headline: the deployable latent route now succeeds sometimes but is well below plain BC, and a semantic advantage of the
packet is not shown.</b> Best R2 {b('learned', 'panda_pg2')} (panda) and {b('learned', 'parm6_tf3')} (parm6) vs BC {b('bc', 'panda_pg2')} and {b('bc', 'parm6_tf3')} on the same scenes.</p>
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
<div class="grid wide">{''.join(video_card(v) for v in TRIPTYCH)}</div>
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
<li><b>Close the R2-vs-BC gap on the source bodies.</b> The gap from BC to the stateless oracle belongs to system 0, which still realizes
only part of the commanded motion. The gap from the oracle to R2 belongs to the generator. Continue what moved the numbers: DAgger on
learner-visited and generated-packet states, no proprioceptive shortcuts, with the offline gate before closed-loop runs.</li>
<li><b>Turn binding redirection into task completion, and isolate the semantic objective.</b> On parm6 the deployable route redirects its
approach after a pure binding change, which BC does not do, but it rarely lifts the new cube (D-074). Next: the same test on a capacity-matched
nosem bundle trained with the same recipe, and on panda once it is competent.</li>
<li><b>Semantic vs capacity-matched no-semantic packets on a competent route</b>: rebind, goal and manipulator-assignment edits with
irrelevant-edit controls. So far there is no advantage (D-059).</li>
<li><b>Only then</b> run the sealed four-way comparison on the held-out bodies. BC transfers to a new gripper but not to the unseen xarm7
arm (D-064), and that is where the latent route has to show its value.</li>
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
td .badge{white-space:normal}.tw{overflow-x:auto;margin:.5rem 0;border:1px solid var(--line);border-radius:6px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{padding:.4rem .55rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;background:var(--code)}tr:last-child td{border-bottom:0}
table.ladder td:nth-child(n+3):nth-child(-n+6){white-space:nowrap}
.arch{display:flex;align-items:stretch;gap:.4rem;margin:1rem 0;flex-wrap:nowrap}
.blk{flex:1 1 0;min-width:0;border:1.5px solid var(--line);border-radius:8px;background:var(--card);padding:.55rem}
.blk.key{border-color:var(--acc)}.blk.dim{opacity:.75;border-style:dashed}
.bt{font-weight:700}.bd{font-size:.82rem;color:var(--mut)}
.arr{align-self:center;font-size:1.3rem;color:var(--mut);text-align:center;flex:0 0 auto}.arr small{display:block;font-size:.65rem;max-width:5em}
.arch2{display:grid;grid-template-columns:1fr 1fr;gap:1rem;font-size:.92rem}
.grid.wide{grid-template-columns:1fr}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:.6rem 0}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;margin:1rem 0}
figure.vid{margin:0;border:1px solid var(--line);border-radius:8px;background:var(--card);overflow:hidden}
figure.vid video{width:100%;display:block;background:#000;aspect-ratio:4/3}.grid.wide figure.vid video{aspect-ratio:auto;max-height:420px}
figcaption{padding:.5rem .6rem;font-size:.84rem;overflow-wrap:anywhere}.novid{padding:2rem;text-align:center;color:var(--mut)}
.update{border:1.5px solid var(--acc);border-radius:8px;padding:.6rem .9rem;background:var(--card)}
#theme{float:right;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:.25rem .6rem;cursor:pointer}
.legend{display:flex;flex-wrap:wrap;gap:.4rem;margin:.5rem 0}
@media (max-width:720px){.grid3{grid-template-columns:1fr}.arch{flex-direction:column}.arr{transform:rotate(90deg);padding:0}.arr small{display:none}
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
            import re as _rq
            if "fresh" in tag or "v2s24543" in tag or _rq.search(r"_s3000[1-9]00$", tag):   # other seed set / pre-fix flow
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
        BEST[kind] = bst
    scoreboard = ("<h3 style=\"margin-top:.8rem\">Scoreboard: pick_place on the 30 matched dev scenes (best checkpoint per row)</h3>"
                  + table(["controller", "panda_pg2", "parm6_tf3", "role"], sc_rows)
                  + '<p class="muted" style="font-size:.85em">Best-of selections across checkpoints and variants are optimistic for every row alike; each '
                  'cell names the run it comes from, and per-run tables follow below. Semantic control of the packet: <b>not shown</b> (D-059, D-062).</p>')
    ob = BEST.get("oracle", {})
    orc_best = ", ".join(f"{v[0]}/{v[1]} {r}" for r, v in sorted(ob.items())) or "—"
    _h = lambda t: (lambda d: f"{d['success']}/{d['n']}")(J(f"ladder_v1/parm5s_tf3/{t}.summary.json")) if have(f"ladder_v1/parm5s_tf3/{t}.summary.json") else "—"
    ho_r2, ho_bc = _h("generated_zero_" + FROZEN[0][0]), _h("learned_bc_direct1701_u12000")
    _h2 = lambda t: (lambda d: f"{d['success']}/{d['n']}")(J(f"ladder_v1/parm5l_pg2/{t}.summary.json")) if have(f"ladder_v1/parm5l_pg2/{t}.summary.json") else "—"
    def _pool(r):
        ds = list(recipe_sets(r, FROZEN[0][0]).values())
        return f"{sum(d['success'] for d in ds)}/{sum(d['n'] for d in ds)} on {r}" if ds else "—"
    pool_txt = ", ".join(_pool(r) for r in ("panda_pg2", "parm6_tf3"))
    ho_r2b, ho_bcb = _h2("generated_zero_" + FROZEN[0][0]), _h2("learned_bc_direct1701_u12000")
    lb = BEST.get("learned", {})
    import re as _rb
    def _recipe_pool(r, tag):
        base = _rb.sub(r"_(?:fresh|s)3000\d00$", "", tag)
        ds = []
        for f in (RAW / "ladder_v1" / r).glob(f"{base}*.summary.json"):
            t = f.name[:-len(".summary.json")]
            if _rb.sub(r"_(?:fresh|s)3000\d00$", "", t) == base:
                ds.append(json.loads(f.read_text()))
        return sum(d["success"] for d in ds), sum(d["n"] for d in ds), len(ds)
    _rp = {r: _recipe_pool(r, v[2]) for r, v in lb.items() if r in ("panda_pg2", "parm6_tf3")}
    r2_best = (", ".join(f"{v[0]}/{v[1]} on {r} (that recipe over all its seed sets: {_rp[r][0]}/{_rp[r][1]})" for r, v in sorted(lb.items()) if r in ("panda_pg2", "parm6_tf3")) + " (best recipe per body on the matched seeds, a best-of selection; final BC " + ", ".join(f"{v[0]}/{v[1]}" for r, v in sorted(BEST.get("bc", {}).items()) if r in ("panda_pg2", "parm6_tf3")) + ")") if lb else "—"
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
<div class="update" id="supports"><h2 style="border:0;margin:.2rem 0 .4rem">What the evidence supports</h2>
<ul>
<li><b>Supported, on deployable routes (no teacher, oracle or BC at run time):</b> the central claim task → packet → behaviour. On the go2 quadruped,
editing the goal in the task context steers the robot (D-071). On the parm6 arm, goal edits (23/80 vs ≤1/80) and binding edits (the original cube is
never lifted; the approach goes to the new cube, also on panda) redirect behaviour beyond matched controls (D-074, D-075, D-077). Plain BC ignores the binding edit.</li>
<li><b>Competence:</b> the deployable latent route matches BC on go2 (nosem 30/30, sem 29/30) and hexapod6 (30/30 both), and on the t1 humanoid for nosem only (26–27/30 vs BC 24/30) (D-070, D-076, D-079). It is partial on the
arms: {pool_txt} pooled over the matched and fresh seed sets, against BC 24–30 of 30 on the same sets (D-072).</li>
<li><b>Semantic supervision: the evidence is mixed and small.</b> The sem packet has more editable probe handles on hexapod6 (halt −0.19 m vs +0.03 m;
turn 0.11 vs 0.03–0.07 rad). But context-to-behaviour control is equal on go2 and hexapod6. On the t1 humanoid the deployable route works for nosem
(26–27/30, at least BC's 24/30) and not for sem (6–13/30); the t1 oracle-route gap (sem 18/30 vs nosem 0/30) reverses on the deployable route. The arm result
has no nosem counterpart, and the binding-v4 sem/nosem bundles are not competent. No claim that semantic supervision improves control is supported (D-059, D-079).</li>
<li><b>Not tested:</b> the sealed held-out target bodies for the latent route. Plain BC transfers to a new gripper (78–85/100) but not to the unseen xarm7 arm
(0/100; D-064). Humanoid g1 has no competent BC control.</li>
</ul></div>
<p class="lede"><b>Bottom line.</b> The scripted teacher solves the tasks and edits shown here on single-arm, dual-arm, legged and most
humanoid bodies (weaker on g1, h1 and one procedural arm, §2b), and the pipeline runs within the latency budget (p95 overhead
1.014×, D-058). <b>Plain behaviour cloning on the same data is competent</b> ({bc_lo}–{bc_hi} of 30 on the matched scenes across
checkpoints; 30/30 on both bodies at the end), so data and evaluation are sound. After fixing a train/deploy mismatch (bug B-1) and a
velocity-copy shortcut in system 0, <b>the deployable latent route succeeds sometimes but stays well below BC</b>: best R2
{r2_best}; on held-out source bodies parm5s_tf3 and parm5l_pg2 {ho_r2} and {ho_r2b} vs BC (12k-update checkpoint) {ho_bc} and {ho_bcb}; pooled over the dev and all fresh seed sets, R2 gets {pool_txt}. A stateless oracle diagnostic, which feeds system 0 packets encoded from BC's own chunks, reaches {orc_best}: the gap from BC to
that diagnostic is system 0's, and the gap from the diagnostic to R2 is the generator's (D-052, D-056, D-063, D-066, D-067, D-068, D-070). <b>A semantic advantage of the packet is not shown</b>: goal content in the packet is executed, but
semantic vs capacity-matched no-semantic packets show no difference (D-059, D-062). <b>On the deployable arm route (parm6_tf3), editing the task
context redirects behaviour beyond matched controls</b>: a goal edit puts the cube at the new goal (23/80 vs ≤1/80 per control, pooled over 80 seeds), and a binding edit
redirects the approach to the new cube (original cube lifted 0/80 vs 58/80 unedited), which plain BC ignores. The rebound task is rarely completed,
and there is no nosem counterpart yet (D-074, D-075).
<b>On the go2 quadruped the deployable latent route is competent</b> (nosem 30/30, sem 29/30 vs BC 30/30), and probe-direction edits of the
packet causally halt and turn the robot (D-069, D-070; §2b). hexapod6 is also competent (30/30), and the t1 humanoid only with the no-semantic
packet (26–27/30 vs sem 6–13/30, D-079). The evidence on semantic supervision is mixed and small. <b>On go2's deployable
route, editing only the task context (mirroring the active waypoint) steers the robot toward the new goal: task → packet → behaviour,
shown for one body and one semantic, and not dependent on semantic supervision (D-071).</b></p>
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
