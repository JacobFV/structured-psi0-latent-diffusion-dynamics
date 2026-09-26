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
     "The arm now reaches the cube (6 mm) but drives to the object instead of the packet's pregrasp waypoint 13 cm above "
     "it, so the expert encoded in the packet never advances (D-047, D-049).",
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
    lab = {"teacher": "scripted_teacher", "oracle": "oracle diagnostic", "learned": "learned", "bc": "learned"}[kind]
    if kind == "learned":
        lab = "learned:flow_latent_sem_v2@22k"
    if kind == "bc":
        lab = "learned:" + f.split("_s30000")[1].split("_", 1)[1].rsplit("_", 1)[0]
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

    lat = J("acceptance_latency/sem_v1_interrupted.json")
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
<h3>Legged {badge('teacher')}</h3>
{legt}
<p>Teacher driving the frozen body trackers, dev seeds 10000–10019. No learned legged model finished. {src('legged_vlm_teacher_ref/eval_dev.summary.json', 'D-040')}</p>
<h3>Infrastructure {badge('ok')}</h3>
<ul>
<li>Stage A / Stage B training (standardized, resumable), system-0 runtime at 50 ms, closed-loop eval, disturbance test, labelled renders. {src('D-031')}</li>
<li>Packet contract: latent-space version = config hash + encoder-weights hash; a retrained bundle with the same config is rejected (unit test). {src('D-038')}</li>
<li>Ladder evaluator with matched scenes, shadow teacher labels and failure-stage localization. {src('D-044')}</li>
<li>Acceptance suite: valid semantic edits (rebind_obj, goal_shift) plus irrelevant-edit controls; teacher / oracle / generated routes. {src('D-041')}</li>
<li>Packet-policy GRPO with exact per-step likelihoods (unit-tested); on hold, no signal from a non-competent base. {src('D-042')}</li>
<li>Sealed-protocol baseline runner (direct actions, action-only codec) with exact resume. {src('D-035')}</li>
</ul>
<h3>Latency {badge('ok', 'measured, with caveats')}</h3>
<p>Observation → first native command, interleaved pairs at NFE 8: latent p95 <b>{pi['latent_obs_to_first_command']['p95']:.0f} ms</b>
vs direct-action p95 <b>{pi['direct_obs_to_chunk']['p95']:.0f} ms</b> → overhead ratio <b>{pi['p95_overhead_ratio']:.2f}×</b>
(threshold {pi['threshold_p95_ratio']}×). System-0 tick p95 {lat['system0_tick']['p95']:.1f} ms, {lat['system0_deadline_misses']}/{lat['system0_tick']['n']} misses of the 50 ms deadline.
Caveats: direct path timed with random weights (same compute), peer GPU shared by other jobs; to rerun on final checkpoints in a quiet window.
{src('acceptance_latency/sem_v1_interrupted.json', 'D-041')}</p>
<div class="grid">{''.join(video_card(v) for v in VIDEOS)}</div>
</section>"""


def sec_matrix():
    rows = [
        ["single-arm pick_place (latent_sem)", badge("ok"),
         "R1 best: <b>1/30</b> (panda, jointfix + DAgger-1); 0/30 on parm6_tf3 " + src("D-048"),
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
        ["latency", badge("ok"), "—", "p95 1.12× (≤ 1.25×), caveats " + src("D-041"), "—", "—"],
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
<span class="badge b-oracle">R1</span> uses a packet encoded from the teacher's future actions: it asks whether
<i>system 0</i> can realize a good packet at all. Wilson 95% intervals in brackets.</p>
{lt}
<p>{src('ladder_v1/<robot>/<tag>.summary.json', 'D-044', 'D-046', 'D-047', 'D-048')}. Rows marked n/a were not run with
re-anchoring. Refit rows use the frozen v1 encoder; "jointfix" retrains encoder and system 0 together with the fix.</p>
<h3>D-049: which oracle variant is valid, and what system 0 actually does</h3>
<p>Re-anchoring the oracle expert changes the teacher whose actions are encoded. Check with the teacher itself driving:
no re-anchoring and re-anchoring every 8 ticks both give {kb[0]['success']}/{kb[0]['n']} (panda) and {kb[1]['success']}/{kb[1]['n']} (parm6);
re-anchoring <i>every tick</i> gives {ka[0]['success']}/{ka[0]['n']} and {ka[1]['success']}/{ka[1]['n']}, stuck in pregrasp.
So the every-8-ticks R1 route (the table above) is valid, and the replan-every-tick R1 runs
(<span class="muted">{esc('; '.join(inv))}</span>) are <b>withdrawn</b>. {src('lead_teacher_reanchor/*_K1.json', 'lead_teacher_reanchor/*_K8.json', 'D-049')}</p>
<p>Mechanism in the valid runs: the median closest approach of the tool is 4–6 cm from the cube, not the pregrasp waypoint
13 cm above it. System 0 drives toward the object while the packet encodes “go to pregrasp”, so the encoded expert never
advances. Hypothesis under test: the object-relative semantic loss plus system 0 favour “go to the patient” over the
demonstrated waypoint geometry. {src('D-047', 'D-049')}</p>
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
<span id="sprint-semantic">see the sprint update below when available.</span></p>
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
    for tag, what in (("direct1701_u12000", "direct-action BC, 12k of ~26.3k updates"),
                      ("codec1701_u13152", "action-only codec BC, 13.2k updates")):
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
    hp = "artifacts/runs/baselines_bc_ladder/heldout/direct1701_u12000.summary.json"
    if have(hp):
        h = J(hp)
        hr = [[f"<code>{esc(b)}</code>", frac(v["successes"], v["attempted"]), str(v["infeasible"]),
               esc(", ".join(f"{k} {n}" for k, n in v["outcomes"].items() if k not in ("success", "infeasible")))]
              for b, v in h.items() if not b.startswith("_")]
        k = sum(v["successes"] for b, v in h.items() if not b.startswith("_"))
        n = sum(v["attempted"] for b, v in h.items() if not b.startswith("_"))
        hr.append(["<b>pooled</b>", frac(k, n), "", ""])
        ho = (f"<h3>Held-out source bodies (not in BC training) {badge('bc', 'learned:direct1701_u12000')}</h3>"
              + table(["body", "success / feasible", "infeasible", "failures"], hr)
              + f"<p>Protocol source-competence harness, seeds 2,000,000+, 50 episodes per body. These are held-out "
              f"<i>source</i> bodies, not the sealed target bodies (xarm7_pg2, xarm7_tf3, panda_tf3). {src(hp)}</p>")
    vids = "".join(video_card(v) for v in BC_VIDEOS)
    return f"""
<section id="bc"><h2>6 · Positive control: plain behaviour cloning with the fix {badge('ok', 'competent')}</h2>
<p class="lede"><b>Plain behaviour cloning with deployment-consistent input (B-1 fixed) is a competent controller on the
ladder's matched scenes, already at mid-training.</b> Same data, same seed (1701), same 30 dev seeds, same tracker and
privileged evaluator, prev-action input 0 as deployed. So the latent route's closed-loop failure comes from the latent
architecture/training (Stage A + system 0), <i>not</i> from the data, the demonstrations or the simulator setup.</p>
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
<li><b>Make one route competent on source bodies.</b> System 0 must track the packet's waypoint geometry, not head for the
object: training on learner-visited states with oracle packets (DAgger on the jointly trained model), a packet target it
can servo to. The bar is the competent plain-BC control on the same seeds (§6).</li>
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
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15171a;--fg:#e6e7ea;--mut:#9aa1ad;--card:#1d2024;--line:#30343a;--acc:#8fb0ff;--code:#262a30;
--teach:#7fd6a6;--teach-bg:#173325;--orc:#f2c265;--orc-bg:#3a2d10;--lrn:#9db7ff;--lrn-bg:#1d2947;--bc:#c7a8f5;--bc-bg:#2e2342;
--none:#aab0ba;--none-bg:#2a2e34;--run:#7dd3e0;--run-bg:#133a41;--ok:#7fd6a6;--ok-bg:#173325;--fail:#ff9b92;--fail-bg:#43201d}}
:root[data-theme="dark"]{--bg:#15171a;--fg:#e6e7ea;--mut:#9aa1ad;--card:#1d2024;--line:#30343a;--acc:#8fb0ff;--code:#262a30;
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
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;margin:1rem 0}
figure.vid{margin:0;border:1px solid var(--line);border-radius:8px;background:var(--card);overflow:hidden}
figure.vid video{width:100%;display:block;background:#000;aspect-ratio:4/3}
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
    body = (sec_architecture() + sec_works() + sec_matrix() + sec_debug() + sec_semantic() + sec_bc() + sec_next())
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
<nav><a href="#arch">architecture</a><a href="#works">what works</a><a href="#matrix">evidence</a><a href="#debug">debugging</a>
<a href="#semantic">semantic edits</a><a href="#bc">BC control</a><a href="#next">next</a><a href="#sources">sources</a></nav>
</header>
<p class="lede"><b>Bottom line.</b> The scripted teacher solves every task and edit shown here on single-arm, dual-arm and legged
bodies, and the pipeline runs end to end within the latency budget. <b>Plain behaviour cloning on the same data is competent</b> (25–28 of 30 on matched scenes, mid-training), so data and
evaluation are sound. <b>The latent-packet route is not competent yet</b>: its best oracle-diagnostic variant succeeds 1 time
in 30, and causal packet semantics are not shown. We found and fixed a
train/deploy mismatch (bug B-1) and localized the remaining failure to system 0 heading for the object instead of the
packet's waypoint.</p>
{updates_html}
{body}
<section id="sources"><h2>Sources</h2>
<p>Decision ids refer to <code>research/decisions.md</code>. Raw paths are relative to <code>artifacts/runs/</code> on the
shared peer store (small copies under <code>docs/demo/raw/artifacts/runs/</code>) unless they start with <code>artifacts/</code> (git-tracked).
Videos: <code>docs/demo/video/</code>; index lines in <code>artifacts/video/INDEX.md</code>.</p>
<details><summary>raw files read to build this page ({len(USED)})</summary><ul>{used}</ul></details>
</section>
</main><script>{JS}</script></body></html>"""
    OUT.write_text(page)
    print(OUT, len(page), "bytes;", len(USED), "raw files")


if __name__ == "__main__":
    upd = ROOT / "docs/demo/updates.html"
    build(upd.read_text() if upd.exists() else "")
