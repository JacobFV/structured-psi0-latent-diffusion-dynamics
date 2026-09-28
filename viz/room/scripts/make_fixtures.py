"""Writes the room's FIXTURE JSON (synthetic, clearly labelled; never results). Run: python viz/room/scripts/make_fixtures.py
Every document carries fixture: true, git_sha "fixture" and source_label/fixture markers so the UI shows it as such."""
import json, math, random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "fixtures"
(OUT / "replays").mkdir(parents=True, exist_ok=True)
rng = random.Random(7)
GEN = "2026-09-28T00:00:00Z"


def env(name, **body):
    return dict(schema=f"rrp-viz/{name}/v1", generated_at=GEN, git_sha="fixture", sources=[f"viz/room/fixtures/{name}.json (FIXTURE)"],
                fixture=True, **body)


def dump(name, doc, sub=""):
    p = OUT / sub / f"{name}.json"
    p.write_text(json.dumps(doc, separators=(",", ":")))
    print(p, p.stat().st_size)


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0, c - h), 4), round(min(1, c + h), 4)


# ---------------------------------------------------------------- results
rows = []
for fam, tasks, bodies in [("arm", ["pick_place", "pick_place_paired"], ["fixture_arm_a", "fixture_arm_b"]),
                           ("legged", ["walk_goal"], ["fixture_quad"]), ("dual", ["handover"], ["fixture_dual"])]:
    for task in tasks:
        for body in bodies:
            for route in ["teacher", "bc", "flow_sem", "flow_nosem"]:
                for gv in (["grasp_v1", "grasp_v2"] if fam != "legged" else ["-"]):
                    n = 30; k = rng.randint(3, 29)
                    lo, hi = wilson(k, n)
                    rows.append(dict(id=f"fx-{fam}-{task}-{body}-{route}-{gv}", family=fam, task=task, body=body, route=route, variant="v1",
                                     seed="1701", grasp_contact_version=gv, contact_version="contact_v2" if fam == "legged" else "contact_v1",
                                     actuator_limits_version="act_v1", metric="success", k=k, n=n, rate=round(k / n, 4), ci_lo=lo, ci_hi=hi,
                                     source_label="fixture:scripted_teacher" if route == "teacher" else f"fixture:learned:{route}",
                                     source_file="viz/room/fixtures/results.json", decision="D-000", interim=route == "flow_nosem",
                                     caveat="FIXTURE row: synthetic" if route == "bc" else None))
dump("results", env("results", rows=rows))

# ---------------------------------------------------------------- overview
dump("overview", env("overview",
    established=[dict(claim="FIXTURE: an established claim appears here with its decision id", decision="D-000"),
                 dict(claim="FIXTURE: a second claim, qualified by a caveat", decision="D-000", caveat="FIXTURE caveat")],
    open=[dict(item="FIXTURE: an open question linked to the roadmap", roadmap="1")],
    caveats=[dict(text="FIXTURE: caveats are listed, never hidden", decision="D-000")],
    key_numbers=[dict(label="FIXTURE success rate", value=0.5, k=15, n=30, ci_lo=0.33, ci_hi=0.67, source_label="fixture", decision="D-000",
                      source_file="viz/room/fixtures/overview.json", interim=True)],
    decisions=[dict(id=f"D-{i:03d}", date="2026-09-28", title=f"FIXTURE decision {i}") for i in range(3, 0, -1)]))

# ---------------------------------------------------------------- live + dags
now = 1790553600
samples = []
for i in range(120):
    t = now - (120 - i) * 10
    samples.append(dict(t=t, level="warn" if 60 <= i < 66 else "ok", reasons=["FIXTURE: psi above threshold"] if 60 <= i < 66 else [],
                        gpu_util=round(50 + 40 * math.sin(i / 9), 1), gpu_temp_c=round(55 + 10 * math.sin(i / 15), 1),
                        cpu_temp_c=round(60 + 8 * math.sin(i / 11), 1), mem_available_gb=round(80 - 20 * math.sin(i / 20), 2),
                        psi_mem_some_avg10=round(max(0, 6 * math.sin(i / 7)), 2)))
dump("live", env("live", stale=False,
    peer=dict(host="fixture-peer", gpu_util=61.0, gpu_temp_c=58.0, cpu_temp_c=63.0, mem_available_gb=74.2, psi_mem_some_avg10=0.4,
              project_mem_gb=21.5, disk_free_gb=512.0, admission=dict(state="open", reason="FIXTURE: headroom available")),
    leases=[dict(id=f"fx-lease-{j}", label=f"FIXTURE job {j}", workstream=w, declared=dict(mem_gb=d, gpu=1 if j < 2 else 0, cpu=4),
                 measured=dict(current_gb=round(d * f, 2), peak_gb=round(d * min(1.1, f + 0.15), 2)), memory_high_events=e, age_s=600 * (j + 1))
            for j, (w, d, f, e) in enumerate([("ladder", 24, 0.62, 0), ("legged", 16, 0.95, 3), ("room", 2, 0.3, 0)])],
    watchdog=samples, host=dict(load_1m=4.2, mem_available_gb=90.1, note="FIXTURE")))
dags = []
for name, ws, states in [("fixture_ladder_dag", "ladder", "cccccrrppp"), ("fixture_legged_dag", "legged", "ccfcbpp"), ("fixture_room_dag", "room", "cccc")]:
    m = dict(c="completed", r="running", p="pending", f="failed", b="blocked")
    nodes = [dict(id=f"{name}.n{i}", stage=f"stage{i // 3}", state=m[s], lease=None, started=None, ended=None, rc=1 if s == "f" else None,
                  caveat=None) for i, s in enumerate(states)]
    counts = {}
    for n in nodes: counts[n["state"]] = counts.get(n["state"], 0) + 1
    dags.append(dict(name=name, workstream=ws, host="peer", nodes=nodes, counts=counts, eta=None))
dump("dags", env("dags", dags=dags))

# ---------------------------------------------------------------- edits
edits = []
for body in ["fixture_arm_a", "fixture_quad"]:
    for seed in ["1701", "1702", "1703"]:
        for edit, eff in [("ctx_halt", -0.4), ("z_turn", 0.3), ("mirror_active", -0.25), ("mirror_inactive", 0.0)]:
            e = eff + rng.uniform(-0.08, 0.08); w = rng.uniform(0.06, 0.15)
            edits.append(dict(body=body, variant="v1", seed=seed, edit=edit, control=edit == "mirror_inactive", effect=round(e, 3),
                              ci=[round(e - w, 3), round(e + w, 3)], n_pairs=64, permutation_p=0.4 if edit == "mirror_inactive" else 0.002,
                              metric="FIXTURE effect", decision="D-000"))
dump("edits", env("edits", rows=edits))

# ---------------------------------------------------------------- training
runs = []
for name, kind in [("fixture_flow_run", "flow"), ("fixture_bc_run", "bc")]:
    steps = list(range(0, 20000, 100))
    loss = [round(2.0 * math.exp(-s / 5000) + 0.1 + rng.uniform(0, 0.03), 4) for s in steps]
    gn = [round(3 * math.exp(-s / 8000) + rng.uniform(0, 0.8) + (4 if 6000 < s < 6800 else 0), 4) for s in steps]
    runs.append(dict(run=name, kind=kind, step=steps, losses=dict(total=loss, flow=[round(x * 0.7, 4) for x in loss]), grad_norm=gn,
                     clip_scale=[round(min(1.0, 2.0 / g), 4) if g > 0 else 1.0 for g in gn], lr=[3e-4 * min(1, s / 1000) for s in steps],
                     alpha=[round(min(1, s / 15000), 4) for s in steps] if kind == "flow" else None,
                     gate="pass" if kind == "flow" else "pending", dagger_rounds=[dict(round=1, step=10000)] if kind == "bc" else [],
                     source_label="fixture", source_file="viz/room/fixtures/training.json", decision="D-000"))
for r in runs:
    if r["alpha"] is None: del r["alpha"]
dump("training", env("training", runs=runs))

# ---------------------------------------------------------------- robustness
sweeps = []
for route in ["flow_sem", "bc"]:
    for factor, levels in [("friction_scale", [0.25, 0.5, 1.0, 1.5, 2.0]), ("push_n", [0, 5, 10, 20, 40])]:
        for li, lv in enumerate(levels):
            n = 20; k = max(0, min(20, int(18 - (abs(li - 2) if factor == "friction_scale" else li) * (4 if route == "bc" else 2.5))))
            lo, hi = wilson(k, n)
            sweeps.append(dict(route=route, factor=factor, level=lv, k=k, n=n, rate=k / n, ci_lo=lo, ci_hi=hi, source_label="fixture"))
dump("robustness", env("robustness", sweeps=sweeps,
    break_points=[dict(route="bc", factor="push_n", level=10, criterion="FIXTURE: rate < 0.5"), dict(route="flow_sem", factor="push_n", level=20, criterion="FIXTURE: rate < 0.5")],
    motion_quality=[dict(route=r, jerk_median=round(rng.uniform(1, 3), 3), foot_slip_median=round(rng.uniform(0, 0.02), 4), source_label="fixture") for r in ["flow_sem", "bc"]],
    variant_diffs=[dict(variant_a="v1", variant_b="v2", factor="push_n", diff=0.12, ci=[0.02, 0.22], p=0.01, n_pairs=40, decision="D-000")]))

# ---------------------------------------------------------------- physics, psi0, knowledge, videos, replays index
dump("physics", env("physics",
    contact_v2=[dict(tracker=f"fixture_tracker_{i}", contact_version=v, slip=round(rng.uniform(0.001, 0.03), 4), cot=round(rng.uniform(0.3, 1.2), 3)) for i in range(3) for v in ["contact_v1", "contact_v2"]],
    grasp_rig=[dict(rig="fixture_rig", grasp_contact_version=v, hold_rate=r, n=20) for v, r in [("grasp_v1", 0.6), ("grasp_v2", 0.95)]],
    tracker_validation=[dict(tracker="fixture_tracker_0", check="FIXTURE check", verdict="pass")],
    gates=[dict(gate=f"FIXTURE gate {i}", decision="D-000", verdict=v) for i, v in enumerate(["pass", "pass", "fail", "pending"])]))
dump("psi0", env("psi0",
    reproduction=[dict(metric="FIXTURE metric", paper=0.8, ours=0.78, n=50)],
    step2=[dict(config="FIXTURE step-2", result=0.5)],
    p_decisions=[dict(id="P-001", date="2026-09-20", title="FIXTURE P-decision", body="FIXTURE body linking **D-000**.")],
    crosswalk=[dict(d="D-000", p="P-001", relation="FIXTURE: informs")]))
dump("knowledge", env("knowledge",
    decisions=[dict(id=f"D-{i:03d}", date=f"2026-09-{20 + i % 8:02d}", title=f"FIXTURE decision {i}",
                    body=f"FIXTURE body for decision {i}. See D-{max(1, i - 1):03d}.\n\n- a list item\n- `code`") for i in range(1, 12)],
    roadmap=[dict(n=1, item="FIXTURE roadmap item", status="open")], backlog=[dict(item="FIXTURE backlog item", status="planned")],
    strategy=[dict(workstream="FIXTURE workstream", goal="FIXTURE goal")], status="# FIXTURE STATUS\n\nThis is fixture text, not the real STATUS.md.",
    crosswalk=[dict(d="D-000", p="P-001", relation="FIXTURE: informs")], docs=["STATUS.md", "README.md"]))
dump("videos", env("videos", videos=[]))


# ---------------------------------------------------------------- replays (synthetic kinematics)
def qz(a): return [math.cos(a / 2), 0, 0, math.sin(a / 2)]
def qy(a): return [math.cos(a / 2), 0, math.sin(a / 2), 0]
def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return [w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2]
def rot(q, v):
    w, x, y, z = q; qv = [0, *v]
    r = qmul(qmul(q, qv), [w, -x, -y, -z]); return r[1:]
def add(a, b): return [a[i] + b[i] for i in range(3)]
def r4(v): return [round(x, 4) for x in v]


def replay(rid, condition, halt):
    fps, n = 30, 120
    bodies = ["world", "base", "link1", "link2", "finger_l", "finger_r", "cube"]
    geoms = [dict(name="floor", body="world", type="plane", size=[1.5, 1.5, 0.1], rgba=[0.82, 0.83, 0.85, 1]),
             dict(name="table", body="world", type="box", size=[0.3, 0.3, 0.02], rgba=[0.55, 0.45, 0.35, 1], pos=[0.45, 0, 0.18], quat=[1, 0, 0, 0]),
             dict(name="goal_zone", body="world", type="cylinder", size=[0.06, 0.002], rgba=[0.1, 0.7, 0.4, 0.5], pos=[0.45, -0.18, 0.202], quat=[1, 0, 0, 0]),
             dict(name="base", body="base", type="cylinder", size=[0.07, 0.1], rgba=[0.2, 0.22, 0.25, 1], pos=[0, 0, 0.1], quat=[1, 0, 0, 0]),
             dict(name="link1", body="link1", type="capsule", size=[0.035, 0.15], rgba=[0.9, 0.9, 0.92, 1], pos=[0.15, 0, 0], quat=qy(math.pi / 2)),
             dict(name="link2", body="link2", type="capsule", size=[0.03, 0.12], rgba=[0.9, 0.9, 0.92, 1], pos=[0.12, 0, 0], quat=qy(math.pi / 2)),
             dict(name="finger_l", body="finger_l", type="box", size=[0.01, 0.008, 0.03], rgba=[0.25, 0.25, 0.28, 1]),
             dict(name="finger_r", body="finger_r", type="box", size=[0.01, 0.008, 0.03], rgba=[0.25, 0.25, 0.28, 1]),
             dict(name="cube", body="cube", type="box", size=[0.02, 0.02, 0.02], rgba=[0.85, 0.2, 0.2, 1]),
             dict(name="marker", body="base", type="mesh", size=[1, 1, 1], rgba=[0.95, 0.7, 0.1, 1], pos=[0, 0, 0.22], quat=[1, 0, 0, 0],
                  mesh=dict(name="tetra", vertices=[0, 0, 0.04, 0.03, 0, 0, -0.015, 0.026, 0, -0.015, -0.026, 0], faces=[0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2]))]
    T, P, Q = [], [], []
    sig = {k: [] for k in ["joint_target", "joint_pos", "contacts", "packet_pca", "phase", "edit_active", "forward_progress", "object_pose", "penetration_mm", "slip"]}
    probe = dict(contact=[], halt=[], goal=[])
    events = []
    cube = [0.45, 0.12, 0.22]
    held = False
    for f in range(n):
        s = f / (n - 1)
        edit = halt and 0.45 < s < 0.75
        yaw_t = 0.26 if s < 0.4 else (0.26 - 0.66 * min(1, (s - 0.4) / 0.4))
        pitch_t = 0.55 if 0.25 < s < 0.4 or s > 0.8 else 0.3
        if edit: yaw_t = 0.26 - 0.66 * 0.1
        q = [yaw_t + 0.02 * math.sin(f / 4), pitch_t, -0.9 + 0.2 * pitch_t]
        grip = 0.012 if 0.33 < s < 0.85 else 0.03
        base_p, base_q = [0, 0, 0.2], [1, 0, 0, 0]
        l1q = qmul(qz(q[0]), qy(q[1])); l1p = add(base_p, [0, 0, 0])
        l2q = qmul(l1q, qy(q[2])); l2p = add(l1p, rot(l1q, [0.3, 0, 0]))
        tip = add(l2p, rot(l2q, [0.25, 0, 0]))
        fl = add(tip, rot(l2q, [0, grip, 0])); fr = add(tip, rot(l2q, [0, -grip, 0]))
        if 0.33 < s < 0.85 and not edit and math.dist(tip, cube) < 0.12: held = True
        if s >= 0.85: held = False
        if held: cube = add(tip, [0, 0, -0.03])
        elif cube[2] > 0.22: cube = [cube[0], cube[1], max(0.22, cube[2] - 0.02)]
        T.append(round(f / fps, 4))
        P.append([r4(base_p), r4(base_p), r4(l1p), r4(l2p), r4(fl), r4(fr), r4(cube)])
        Q.append([[1, 0, 0, 0], r4(base_q), r4(l1q), r4(l2q), r4(l2q), r4(l2q), [1, 0, 0, 0]])
        sig["joint_target"].append(r4([yaw_t, pitch_t, -0.9 + 0.2 * pitch_t]))
        sig["joint_pos"].append(r4(q))
        sig["contacts"].append([held, held])
        sig["packet_pca"].append(r4([math.cos(s * 5) + (0.8 if edit else 0), math.sin(s * 5), s * 2 - 1]))
        ph = "approach" if s < 0.33 else "grasp" if s < 0.45 else "transport" if s < 0.85 else "release"
        sig["phase"].append(ph)
        sig["edit_active"].append(edit)
        sig["forward_progress"].append(round(min(1, s * 1.1) * (0.6 if halt else 1), 4))
        sig["object_pose"].append(r4(cube + [1, 0, 0, 0]))
        sig["penetration_mm"].append(round(abs(math.sin(f / 5)) * (0.4 if held else 0.05), 3))
        sig["slip"].append(round(0.002 * abs(math.sin(f / 3)) if held else 0, 4))
        probe["contact"].append([round(0.9 if held else 0.08, 2), round(0.85 if held else 0.1, 2)])
        probe["halt"].append(round(0.9 if edit else 0.05, 2))
        probe["goal"].append(r4([0.45, -0.18 if not edit else 0.1]))
        if f == 0: events.append(dict(t=0.0, event="reach", status="pending"))
        if f == int(0.4 * n): events.append(dict(t=T[-1], event="grasp", status="success" if held else "failure"))
        if f == n - 1: events.append(dict(t=T[-1], event="place", status="success" if not halt else "failure"))
    sig["probe"] = probe; sig["task_events"] = events
    meta = dict(family="arm", task="pick_place", body="fixture_arm", route="flow_sem", source_label="fixture:learned:synthetic",
                ckpt_sha=None, variant="v1", seed=1701, condition=condition, success=not halt, failure_stage="place" if halt else None,
                physics=dict(contact_version="contact_v1", grasp_contact_version="grasp_v2", actuator_limits_version="act_v1", actuator_mode="position"),
                decision_refs=["D-131"], caveat="FIXTURE: synthetic kinematics, no physics, not a recorded episode", fixture=True,
                contact_bodies=["finger_l", "finger_r"], base_body="link2", joint_names=["yaw", "shoulder", "elbow"],
                packet_pca_basis=dict(explained_variance_ratio=[0.4, 0.2, 0.1], n_fit=0, fit_on="FIXTURE"))
    doc = dict(schema="rrp-viz/replay/v1", id=rid, meta=meta, fps=fps, n_frames=n, geoms=geoms, bodies=bodies,
               frames=dict(t=T, body_pos=P, body_quat=Q), signals=sig, annotations=[dict(t=T[n // 2], text="FIXTURE annotation")])
    dump(rid, doc, "replays")
    return dict(id=rid, family="arm", task="pick_place", body="fixture_arm", route="flow_sem", source_label=meta["source_label"], variant="v1",
                seed=1701, condition=condition, success=not halt, n_frames=n, fps=fps, file=f"{rid}.json")


idx = [replay("fixture-arm-unedited", "unedited", False), replay("fixture-arm-halt", "ctx_halt", True)]
dump("replays", env("replays", n=len(idx), replays=idx))
