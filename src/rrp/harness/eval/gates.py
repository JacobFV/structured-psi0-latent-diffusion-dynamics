"""Motion-quality and robustness GATES for trackers and datasets (W6, adopted in D-112; thresholds in GATES).

API (pure functions of recorded outputs; nothing is re-simulated here):
  check_tracker(validation: dict)                         -> report   (rrp.harness.eval.tracker_validation output)
  check_dataset(manifest: dict | None, episodes: list)    -> report   (dispatches on the episodes' family)
  check_legged_dataset(episodes, manifest=None)           -> report   (rrp.harness.data.legged_latent_collect episode metas)
  check_arm_dataset(episodes, manifest=None, reference=)  -> report   (rrp.harness.data.collect metas with `motion`, or
                                                                       rrp.harness.eval.teacher_quality rows)
  check_dual_dataset(episodes, manifest=None)             -> report   (rrp.harness.data.collect_dual metas with record_quality,
                                                                       or rrp.harness.eval.dual_teacher_quality rows)
  policy_flags(rows)                                      -> report   (learned-policy eval rows; REPORTED, never gated)
A report is {gate, version, subject, verdict, criteria: [{name, status, value, threshold, note}], failed: [...]}.
Criterion status: pass | fail | not_evaluated (the input lacks the measurement) | labelled (measured, reported, not
gated: e.g. penetration under grasp_v1). verdict: fail if any criterion fails; else incomplete if any gated criterion
is not_evaluated; else pass. Pipelines treat ONLY "fail" as a failed node (rrp.harness.pipelines.base.GateFailed); "incomplete" is recorded.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

GATES_VERSION = "rrp.evaluation.gates/v1 (D-112)"
GATE_REPORT = "gate_report.json"

GATES = dict(
    tracker=dict(slip_ratio_max=0.15, cot_max=dict(quadruped=1.0, biped=2.0), peak_force_bw_max=dict(quadruped=3.5, biped=3.0),
                 joint_limit_margin_min=0.02, no_fall_min=0.9, robust_forward_drop_max=0.20, robust_no_fall_min=0.9),
    legged_dataset=dict(slip_ratio_max=0.15, slip_ok_frac_min=0.95, falls_at_sigma0_max=0),
    arm_dataset=dict(phase_switch_vel_step_max=0.5, cmd_jerk_rms_ratio_max=2.0, joint_limit_margin_min=0.02,
                     ok_frac_min=0.95, penetration_max_m=0.003, penetration_ok_frac_min=0.99,
                     penetration_gated_grasp=("grasp_v2", "grasp_v2.1")),
    policy=dict(chunk_vel_step_flag=1.5),
    # D-126 #18 (W12 audit, archived; D-126): dual-arm teacher data. Arm criteria per arm as arm_dataset;
    # maintained-contact criteria from the W12 contact metrics (rrp.harness.data.contact_metrics).
    dual_dataset=dict(phase_switch_vel_step_max=0.5, cmd_jerk_rms_ratio_max=2.0, joint_limit_margin_min=0.02,
                      ok_frac_min=0.95, penetration_max_m=0.003, penetration_ok_frac_min=0.99,
                      penetration_gated_grasp=("grasp_v2", "grasp_v2.1"), support_slip_max_m=0.005,
                      grip_drift_pos_max_m=0.003, grip_drift_rot_max_rad=0.1, contact_order_error_max=0.0,
                      success_min=None),
)
BIPED_FAMILIES = ("humanoid", "biped")
# Commanded-jerk reference of the v2 scripted teacher per body (D-112 "jerk RMS <= 2x the v2 teacher"): median over
# 300 seeds of joint_cmd_jerk_rms, grasp_v2 (D-110). Built from artifacts/runs/armexpert_grasp/final_graspv2_18bodies.summary.json
# (sha256 8083c464411268d3); regenerate with rrp.harness.eval.teacher_quality if the teacher changes.
ARM_TEACHER_V2_REFERENCE = dict(
    source="artifacts/runs/armexpert_grasp/final_graspv2_18bodies.summary.json (scripted_teacher:pick_place_v2_minjerk, grasp_v2, 300 seeds per body; D-110)",
    source_sha256_16="8083c464411268d3", metric="median joint_cmd_jerk_rms (rad/s^3)",
    bodies={
        'panda_pg2': 7.3325,
        'panda_tf3': 7.0141,
        'parm5_pg2': 8.5074,
        'parm5_tf3': 8.7963,
        'parm5l_pg2': 7.8955,
        'parm5l_tf3': 8.3123,
        'parm5s_pg2': 9.6447,
        'parm5s_tf3': 10.0774,
        'parm6_pg2': 7.0341,
        'parm6_tf3': 7.2985,
        'parm7_pg2': 7.1542,
        'parm7_tf3': 7.2927,
        'sawyer_pg2': 8.3811,
        'sawyer_tf3': 8.0764,
        'ur5e_pg2': 5.7997,
        'ur5e_tf3': 5.8149,
        'xarm7_pg2': 7.2833,
        'xarm7_tf3': 7.3423})


def _crit(name, value, threshold, ok, note="", status=None):
    if status is None:
        status = "not_evaluated" if ok is None else ("pass" if ok else "fail")
    v = None if value is None or (isinstance(value, float) and not math.isfinite(value)) else value
    return dict(name=name, status=status, value=v, threshold=threshold, note=note)


def _report(gate, subject, crits, extra=None):
    failed = [f"{c['name']}={c['value']} (threshold {c['threshold']})" for c in crits if c["status"] == "fail"]
    gated_missing = [c["name"] for c in crits if c["status"] == "not_evaluated"]
    verdict = "fail" if failed else ("incomplete" if gated_missing else "pass")
    r = dict(gate=gate, version=GATES_VERSION, subject=subject, verdict=verdict, criteria=crits, failed=failed,
             not_evaluated=gated_missing)
    if extra:
        r.update(extra)
    return r


def write_report(report: dict, out_dir: Path) -> Path:
    p = Path(out_dir) / GATE_REPORT
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=1, default=str))
    return p


# ------------------------------------------------------------------ trackers
def check_tracker(v: dict) -> dict:
    """v: the JSON written by rrp.harness.eval.tracker_validation (protocol v2; `robustness` from --robust).
    Criteria (forward trial unless stated): slip ratio < 0.15; CoT <= 1.0 quadruped / 2.0 biped; peak foot force
    <= 3.5 body weights (max over all trials); joint-limit margin >= 0.02 (min over all trials); no-fall >= 0.9 (all
    trials); robustness INSIDE the tracker's own training randomization (`robustness` block): at every condition
    no-fall >= 0.9 and forward ratio >= nominal forward ratio - 0.20 (the sweep's 20-point break criterion)."""
    G = GATES["tracker"]
    fam = v.get("family") or ""
    kind = "biped" if fam in BIPED_FAMILIES else "quadruped"
    s = v.get("summary", {})
    fw = s.get("forward", {})
    g = v.get("gate", {})
    sr = (g.get("contact_gate") or {}).get("slip_ratio", fw.get("slip_ratio"))
    crits = [
        _crit("slip_ratio_forward", sr, f"< {G['slip_ratio_max']}", None if sr is None else sr < G["slip_ratio_max"]),
        _crit("cot_forward", fw.get("cot"), f"<= {G['cot_max'][kind]} ({kind})",
              None if fw.get("cot") is None else fw["cot"] <= G["cot_max"][kind]),
    ]
    pk = [x.get("peak_force_bw") for x in s.values() if x.get("peak_force_bw") is not None]
    pmax = G["peak_force_bw_max"][kind]
    raw = [x.get("peak_force_raw_bw") for x in s.values() if x.get("peak_force_raw_bw") is not None]
    crits.append(_crit("peak_foot_force_bw", max(pk) if pk else None, f"<= {pmax} ({kind}; 20 ms moving average)",
                       (max(pk) <= pmax) if pk else None,
                       "max over trials of the peak 20 ms moving-average contact normal force per foot / body weight"
                       + (f"; unfiltered per-step peak {max(raw):.2f}" if raw else "")))
    mg = [x.get("joint_limit_margin_min") for x in s.values() if x.get("joint_limit_margin_min") is not None]
    crits.append(_crit("joint_limit_margin", min(mg) if mg else None, f">= {G['joint_limit_margin_min']}",
                       (min(mg) >= G["joint_limit_margin_min"]) if mg else None, "min over trials, ticks and joints"))
    nf = g.get("no_fall_rate")
    crits.append(_crit("no_fall_rate", nf, f">= {G['no_fall_min']}", None if nf is None else nf >= G["no_fall_min"]))
    rb = v.get("robustness")
    if not rb:
        crits.append(_crit("robust_in_training_range", None, "no break inside training randomization", None,
                           "run tracker_validation --robust"))
    else:
        nom = rb["nominal_forward_ratio"]
        bad = [f"{k}: no_fall {c['no_fall_rate']:.2f}, fwd {c['forward_ratio']:.2f}" for k, c in rb["conditions"].items()
               if c["no_fall_rate"] < G["robust_no_fall_min"] or c["forward_ratio"] < nom - G["robust_forward_drop_max"]]
        crits.append(_crit("robust_in_training_range", bad or "all conditions ok",
                           f"no-fall >= {G['robust_no_fall_min']} and forward ratio >= nominal ({nom:.2f}) - "
                           f"{G['robust_forward_drop_max']} at {sorted(rb['conditions'])}", not bad,
                           f"training range: {rb.get('training_range')}"))
    return _report("tracker", dict(body=v.get("body"), tracker_version=v.get("tracker_version"),
                                   tracker_sha=v.get("tracker_sha"), contact_model=v.get("contact_model"),
                                   family=fam), crits)


# ------------------------------------------------------------------ datasets
def _family(episodes) -> str:
    for e in episodes:
        m = e.get("motion") or {}
        if m.get("family"):
            return m["family"]
        if "sigma" in e and "tracker_version" in e:
            return "legged"
        if "joint_cmd_jerk_rms" in e or "control_dt" in e or "robot_key" in e:
            return "arm"
    return "unknown"


def check_dataset(manifest: dict | None, episodes: list[dict], **kw) -> dict:
    fam = _family(episodes)
    if fam == "legged":
        return check_legged_dataset(episodes, manifest)
    if fam == "arm":
        return check_arm_dataset(episodes, manifest, **kw)
    if fam == "dual":
        return check_dual_dataset(episodes, manifest, **kw)
    return _report("dataset", dict(n=len(episodes)), [_crit("family", None, "legged|arm", None, "unrecognised episodes")])


def check_legged_dataset(episodes: list[dict], manifest: dict | None = None) -> dict:
    """Legged: stance slip ratio < 0.15 on >= 95% of the episodes that have a slip measurement (motion.slip_ratio:
    loaded feet over moving ticks), and 0 falls among the sigma-0 (noise-free) episodes."""
    G = GATES["legged_dataset"]
    sl = [(e.get("motion") or {}).get("slip_ratio") for e in episodes]
    have = [x for x in sl if x is not None]
    frac = (sum(x < G["slip_ratio_max"] for x in have) / len(have)) if have else None
    s0 = [e for e in episodes if float(e.get("sigma", 0.0)) == 0.0]
    falls0 = sum(e.get("status") == "fell" for e in s0)
    crits = [
        _crit("slip_ok_fraction", None if frac is None else round(frac, 4),
              f">= {G['slip_ok_frac_min']} of episodes with slip < {G['slip_ratio_max']}",
              None if frac is None else frac >= G["slip_ok_frac_min"],
              f"{len(have)}/{len(episodes)} episodes measured; median slip "
              f"{float(np.median(have)) if have else None}; max {max(have) if have else None}"),
        _crit("falls_at_sigma0", falls0 if s0 else None, f"<= {G['falls_at_sigma0_max']}",
              None if not s0 else falls0 <= G["falls_at_sigma0_max"], f"{len(s0)} sigma-0 episodes"),
    ]
    st = {}
    for e in episodes:
        st[e.get("status")] = st.get(e.get("status"), 0) + 1
    body = sorted({str(e.get("body")) for e in episodes})
    return _report("legged_dataset", dict(body=body, n=len(episodes), statuses=st,
                                          tracker_sha256=sorted({str(e.get("tracker_sha256")) for e in episodes}),
                                          name=(manifest or {}).get("name")), crits)


def _arm_row(e: dict) -> dict:
    """Normalise a collector meta (with `motion`) or a teacher_quality row."""
    m = e.get("motion") or {}
    phys = e.get("physics") or {}
    grasp = m.get("grasp_contact_version") or phys.get("grasp_contact_version") or e.get("grasp_contact")
    grasp = ("grasp_v1" if grasp in (None, "v1", "grasp_v1") else
             str(grasp) if str(grasp).startswith("grasp_") else f"grasp_{grasp}")      # v2 -> grasp_v2, v2.1 -> grasp_v2.1
    pick = lambda *ks: next((x for x in ks if x is not None), None)
    return dict(body=pick(e.get("robot_key"), e.get("robot")), status=pick(e.get("status"), e.get("outcome")),
                step=pick(m.get("phase_switch_vel_step_max"), e.get("vel_jump_switch_max")),
                jerk=pick(m.get("cmd_jerk_rms"), e.get("joint_cmd_jerk_rms")),
                margin=pick(m.get("joint_limit_margin_min"), e.get("joint_limit_margin_min")),
                pen=pick(m.get("penetration_max_m"), e.get("pen_max_m")), grasp=grasp,
                noise=float(pick(e.get("exec_noise"), 0.0) or 0.0))


PROCEDURAL_ARM_PREFIXES = ("parm", "pa2s")     # pa2s = procedural_arm_family/v2 (D-137): generated joint ranges, like parm*


def _procedural(body) -> bool:
    return str(body).startswith(PROCEDURAL_ARM_PREFIXES)


def load_arm_reference(path: Path | None = None) -> dict:
    return json.loads(Path(path).read_text()) if path else ARM_TEACHER_V2_REFERENCE


def check_arm_dataset(episodes: list[dict], manifest: dict | None = None, reference: dict | None = None) -> dict:
    """Arm teacher data (feasible episodes; commanded = the clean teacher command, i.e. the LABEL, also in DART episodes):
    phase-switch joint-velocity step <= 0.5 rad/s, joint-limit margin >= 0.02, each on >= 95% of episodes; per body,
    median commanded joint jerk RMS <= 2x the v2 teacher's median on that body (ARM_TEACHER_V2_REFERENCE);
    penetration <= 3 mm on >= 99% of episodes, GATED for grasp_v2 data only (grasp_v1 is measured and LABELLED)."""
    G = GATES["arm_dataset"]
    ref = reference if reference is not None else load_arm_reference()
    R = [_arm_row(e) for e in episodes if e.get("status") not in ("infeasible", "generation_error")
         and e.get("outcome") != "infeasible" and e.get("feasible", True)]
    def frac(key, ok):
        xs = [r[key] for r in R if r[key] is not None]
        return ((sum(ok(x) for x in xs) / len(xs)) if xs else None), xs
    f_step, steps = frac("step", lambda x: x <= G["phase_switch_vel_step_max"])
    crits = [
        _crit("phase_switch_vel_step", None if f_step is None else round(f_step, 4),
              f">= {G['ok_frac_min']} of episodes with step <= {G['phase_switch_vel_step_max']} rad/s",
              None if f_step is None else f_step >= G["ok_frac_min"],
              f"median {float(np.median(steps)) if steps else None}, max {max(steps) if steps else None}"),
    ]
    # joint-limit margin: ENFORCED on menagerie arms; REPORTED (labelled) on the procedural parm* arms, whose joint ranges
    # are invented and which the v2 teacher drives into their limits (lead decision after D-112; W7 backlog: limit-aware IK)
    thr = f">= {G['ok_frac_min']} of episodes with margin >= {G['joint_limit_margin_min']}"
    # D-118 (2): under DART execution noise the arm is perturbed, so touching a joint limit is expected and physical:
    # the margin is ENFORCED on clean (noise-free) menagerie-arm episodes and REPORTED on DART episodes.
    for name, sel, gated in (("joint_limit_margin", lambda r: not _procedural(r["body"]) and r["noise"] == 0.0, True),
                             ("joint_limit_margin_dart", lambda r: not _procedural(r["body"]) and r["noise"] > 0.0, False),
                             ("joint_limit_margin_procedural", lambda r: _procedural(r["body"]), False)):
        xs = [r["margin"] for r in R if r["margin"] is not None and sel(r)]
        if not xs and not gated:
            continue
        f = (sum(x >= G["joint_limit_margin_min"] for x in xs) / len(xs)) if xs else None
        bodies = sorted({str(r["body"]) for r in R if sel(r)})
        note = f"bodies {bodies}; min {min(xs) if xs else None}"
        if gated:
            if not xs:
                continue                                   # no menagerie episodes in this dataset
            crits.append(_crit(name, round(f, 4), thr, f >= G["ok_frac_min"], note))
        else:
            why = "procedural arms" if name.endswith("procedural") else "DART episodes (D-118)"
            crits.append(_crit(name, round(f, 4), thr + f" ({why}: reported, not gated)", None, note,
                               status="labelled"))
    per_body, bad, missing = {}, [], []
    for b in sorted({r["body"] for r in R}):
        js = [r["jerk"] for r in R if r["body"] == b and r["jerk"] is not None]
        rf = (ref.get("bodies") or {}).get(b)
        if not js or rf is None:
            missing.append(b)
            continue
        ratio = float(np.median(js)) / rf
        per_body[b] = round(ratio, 3)
        if ratio > G["cmd_jerk_rms_ratio_max"]:
            bad.append(b)
    crits.append(_crit("cmd_jerk_rms_vs_v2_teacher", per_body or None,
                       f"median per body <= {G['cmd_jerk_rms_ratio_max']}x v2 teacher",
                       None if not per_body else not bad,
                       (f"no reference/measurement for {missing}; " if missing else "") + f"reference {ref.get('source')}"))
    grasps = sorted({r["grasp"] for r in R})
    pens = [r["pen"] for r in R if r["pen"] is not None]
    fp = (sum(x <= G["penetration_max_m"] for x in pens) / len(pens)) if pens else None
    note = f"grasp {grasps}; median {float(np.median(pens)) if pens else None} m, max {max(pens) if pens else None} m"
    if pens and all(g in G["penetration_gated_grasp"] for g in grasps):
        crits.append(_crit("penetration", round(fp, 4), f">= {G['penetration_ok_frac_min']} of episodes <= "
                           f"{G['penetration_max_m'] * 1000:.0f} mm (grasp_v2)", fp >= G["penetration_ok_frac_min"], note))
    else:
        crits.append(_crit("penetration", None if fp is None else round(fp, 4),
                           f"<= {G['penetration_max_m'] * 1000:.0f} mm gated under grasp_v2 only", None, note,
                           status="labelled" if pens else "not_evaluated"))
    return _report("arm_dataset", dict(bodies=sorted({str(r['body']) for r in R}), n=len(R), grasp=grasps,
                                       name=(manifest or {}).get("name")), crits)


# ------------------------------------------------------------------ dual-arm datasets (D-126 #18)
def _dual_row(e: dict) -> dict:
    """Normalise a collect_dual meta (`motion` from rrp.harness.data.dual_quality, record_quality: true) or a
    rrp.harness.eval.dual_teacher_quality row."""
    m = e.get("motion") if isinstance(e.get("motion"), dict) and e["motion"].get("family") == "dual" else e
    pair = e.get("robot_key") or e.get("pair") or ""
    bodies = dict(zip(("left", "right"), str(pair).split("__"))) if "__" in str(pair) else {}
    grasp = m.get("grasp_contact_version") or (e.get("physics") or {}).get("grasp_contact_version") or "grasp_v1"
    return dict(pair=pair, bodies=bodies, status=e.get("status"), per_arm=m.get("per_arm") or {},
                pen=m.get("penetration_max_m"), contact=m.get("contact") or {}, grasp=grasp,
                noise=float(e.get("exec_noise", e.get("noise", 0.0)) or 0.0))


def check_dual_dataset(episodes: list[dict], manifest: dict | None = None, reference: dict | None = None) -> dict:
    """Dual-arm teacher data (feasible episodes with a quality record). Per arm, as the arm gate: phase-switch step
    <= 0.5 rad/s and (clean menagerie arms) joint margin >= 0.02 on >= 95% of episodes; commanded jerk RMS median per
    body <= 2x the ARM v2 teacher on that body. Penetration <= 3 mm on >= 99% (grasp_v2*; labelled under grasp_v1).
    Maintained contact (clean episodes): support-anchor slip <= 5 mm, held-object in-grip drift <= 3 mm and <= 0.1 rad,
    contact-order error 0 vs the task spec, each on >= 95% of the episodes that have the measurement. DART episodes'
    maintained-contact values are reported (labelled), not gated."""
    G = GATES["dual_dataset"]
    ref = reference if reference is not None else ARM_TEACHER_V2_REFERENCE
    R = [_dual_row(e) for e in episodes if e.get("status") not in ("infeasible", "generation_error", "error")
         and e.get("feasible", True) is not False]
    R = [r for r in R if r["per_arm"]]
    if not R:
        return _report("dual_dataset", dict(n=0, name=(manifest or {}).get("name")),
                       [_crit("quality_records", None, "episodes recorded with record_quality: true", None,
                              "no dual quality records (collect with record_quality: true)")])
    crits = []

    def frac_crit(name, vals, ok, thr, gated=True, note=""):
        xs = [v for v in vals if v is not None]
        if not xs:
            crits.append(_crit(name, None, thr, None, note or "no measurement", status="not_evaluated" if gated else "labelled"))
            return
        f = sum(ok(x) for x in xs) / len(xs)
        crits.append(_crit(name, round(f, 4), f">= {G['ok_frac_min']} of episodes {thr}",
                           (f >= G["ok_frac_min"]) if gated else None,
                           note + f" n={len(xs)}, median {float(np.median(xs)):.4g}, max {max(xs):.4g}",
                           status=None if gated else "labelled"))

    steps = [max((p.get("phase_switch_vel_step_max") or 0.0) for p in r["per_arm"].values()) for r in R]
    frac_crit("phase_switch_vel_step", steps, lambda x: x <= G["phase_switch_vel_step_max"],
              f"with max over arms <= {G['phase_switch_vel_step_max']} rad/s")
    clean = [r for r in R if r["noise"] == 0.0]
    mg = [min(p["joint_limit_margin_min"] for e, p in r["per_arm"].items() if p.get("joint_limit_margin_min") is not None
              and not _procedural(r["bodies"].get(e, "parm"))) for r in clean
          if any(p.get("joint_limit_margin_min") is not None and not _procedural(r["bodies"].get(e, "parm"))
                 for e, p in r["per_arm"].items())]
    frac_crit("joint_limit_margin", mg, lambda x: x >= G["joint_limit_margin_min"],
              f"with margin >= {G['joint_limit_margin_min']} (clean episodes, menagerie arms)", gated=bool(mg),
              note="" if mg else "no menagerie arm in the clean episodes (procedural parm* margins are report-only, D-114)")
    per_body, bad, missing = {}, [], set()
    js: dict = {}
    for r in R:
        for e, p in r["per_arm"].items():
            b = r["bodies"].get(e)
            if b is not None and p.get("cmd_jerk_rms") is not None:
                js.setdefault(b, []).append(p["cmd_jerk_rms"])
    for b, xs in sorted(js.items()):
        rf = (ref.get("bodies") or {}).get(b)
        if rf is None:
            missing.add(b)
            continue
        per_body[b] = round(float(np.median(xs)) / rf, 3)
        if per_body[b] > G["cmd_jerk_rms_ratio_max"]:
            bad.append(b)
    crits.append(_crit("cmd_jerk_rms_vs_arm_v2_teacher", per_body or None,
                       f"median per body <= {G['cmd_jerk_rms_ratio_max']}x the arm v2 teacher",
                       None if not per_body else not bad, f"no reference for {sorted(missing)}" if missing else ""))
    grasps = sorted({r["grasp"] for r in R})
    pens = [r["pen"] for r in R if r["pen"] is not None]
    if pens and all(g in G["penetration_gated_grasp"] for g in grasps):
        fp = sum(x <= G["penetration_max_m"] for x in pens) / len(pens)
        crits.append(_crit("penetration", round(fp, 4), f">= {G['penetration_ok_frac_min']} of episodes <= 3 mm",
                           fp >= G["penetration_ok_frac_min"], f"grasp {grasps}; max {max(pens):.4g} m"))
    else:
        crits.append(_crit("penetration", None, "<= 3 mm gated under grasp_v2* only", None, f"grasp {grasps}",
                           status="labelled" if pens else "not_evaluated"))
    for sel, gated, sfx in ((clean, True, ""), ([r for r in R if r["noise"] > 0], False, "_dart")):
        if not sel and not gated:
            continue
        c = [r["contact"] for r in sel]
        frac_crit("support_slip" + sfx, [x.get("cf_support_anchor_slip_max_m") for x in c],
                  lambda x: x <= G["support_slip_max_m"], f"<= {G['support_slip_max_m'] * 1000:.0f} mm", gated)
        frac_crit("grip_drift_pos" + sfx, [x.get("cf_held_pos_drift_grip_max_m") for x in c],
                  lambda x: x <= G["grip_drift_pos_max_m"], f"<= {G['grip_drift_pos_max_m'] * 1000:.0f} mm", gated)
        frac_crit("grip_drift_rot" + sfx, [x.get("cf_held_rot_drift_grip_max_rad") for x in c],
                  lambda x: x <= G["grip_drift_rot_max_rad"], f"<= {G['grip_drift_rot_max_rad']} rad", gated)
        frac_crit("contact_order" + sfx, [x.get("cf_contact_order_error") for x in c],
                  lambda x: x <= G["contact_order_error_max"], "order error 0 vs the task spec", gated)
    succ = sum(r["status"] == "success" for r in R) / len(R)
    crits.append(_crit("success_rate", round(succ, 4), "reported", None, "", status="labelled"))
    return _report("dual_dataset", dict(pairs=sorted({str(r["pair"]) for r in R}), n=len(R), grasp=grasps,
                                        name=(manifest or {}).get("name")), crits)


# ------------------------------------------------------------------ learned policies (reported only)
def policy_flags(rows: list[dict]) -> dict:
    thr = GATES["policy"]["chunk_vel_step_flag"]
    xs = [(r.get("motion") or {}).get("chunk_vel_step_max") for r in rows]
    xs = [x for x in xs if x is not None]
    n_flag = sum(x > thr for x in xs)
    return dict(gate="policy_report", version=GATES_VERSION, verdict="reported", n=len(rows),
                chunk_vel_step_median=float(np.median(xs)) if xs else None, chunk_vel_step_max=max(xs) if xs else None,
                flagged=n_flag, flag_threshold=thr)
