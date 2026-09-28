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

# ---------------------------------------------------------------- overview (shape of rrp.viz.export overview)
dump("overview", env("overview", status_updated="FIXTURE", current_state_markdown="**FIXTURE** current state text.",
    claims=[dict(title="FIXTURE claim", text="an established claim appears here with its decision id", decisions=["D-000"], status="established", interim=False, source_file="STATUS.md", line=1),
            dict(title="FIXTURE interim claim", text="a claim still being checked", decisions=["D-000"], status="interim", interim=True, source_file="STATUS.md", line=2)],
    open=[dict(n="1", section="FIXTURE", question="FIXTURE: an open roadmap question", status="running", status_text="running", decisions=[], source_file="docs/experiments_roadmap.md", line=1)],
    caveats=[dict(text="FIXTURE: caveats are listed, never hidden", decisions=["D-000"], source_file="STATUS.md", line=3)],
    key_numbers=[dict(k=15, n=30, text="15/30", context="FIXTURE context", claim="FIXTURE claim", decisions=["D-000"], interim=True, source_file="STATUS.md", line=1)],
    latest_decisions=[dict(id=f"D-{i:03d}", date="2026-09-28", title=f"FIXTURE decision {i}") for i in range(3, 0, -1)],
    workstreams=[dict(id="W0", workstream="FIXTURE workstream", state="running", where="research/tracks/room.md", decisions=["D-000"])],
    results_summary=dict(n_rows=len(rows), n_files=1, n_interim=0, n_with_caveat=0)))

# ---------------------------------------------------------------- live + dags
now = 1790553600
GB = 1024 ** 3
samples = []
for i in range(120):
    t = now - (120 - i) * 10
    samples.append(dict(t=t, level="warn" if 60 <= i < 66 else "ok", reasons=["FIXTURE: psi above threshold"] if 60 <= i < 66 else [],
                        gpu_temp_c=round(55 + 10 * math.sin(i / 15), 1), thermal_c=round(60 + 8 * math.sin(i / 11), 1),
                        memory_available=int((80 - 20 * math.sin(i / 20)) * GB), project_memory=int((30 + 10 * math.sin(i / 13)) * GB),
                        psi_full_avg10=round(max(0, 6 * math.sin(i / 7)), 2), idle_cores=round(12 + 4 * math.sin(i / 9), 2),
                        project_cpu_cores=round(5 + 2 * math.sin(i / 8), 2), project_gpu_bytes=int(20 * GB), throttled_leases=[], gpu_thermal_throttle=False))
dump("live", env("live", peer="fixture-peer", peer_read_at=GEN, stale=False, stale_after_s=60, peer_errors=[], last_error=None,
    node=dict(gpu=[dict(name="FIXTURE GPU", util_pct="61", temp_c="58", power_w="40")], gpu_temp_c=58.0, cpu_temp_c=63.0,
              memory_available=int(74 * GB), mem_total=int(120 * GB), psi_memory="some avg10=0.40 avg60=0.2 avg300=0.1 total=1\nfull avg10=0.10 avg60=0 avg300=0 total=1",
              psi_cpu="some avg10=3.00 avg60=3 avg300=3 total=1", loadavg="6.1 5.9 5.7 5/1000 1", project_memory=int(21 * GB), disk_free=int(500 * GB),
              home_free_bytes=int(90 * GB), shm_free_bytes=int(29 * GB), admission=dict(stopped=False, reason=None),
              limits=dict(cpu_cores=20, memory_bytes=int(108 * GB), gpu_slots=8), watchdog_last=dict(level="ok", reasons=[]), watchdog_heartbeat_age_s=1.0),
    leases=[dict(id=f"fx-lease-{j}", label=f"FIXTURE job {j}", workstream=w, dag="fixture_dag", dag_node=f"n{j}",
                 declared=dict(memory_bytes=int(dm * GB), cpu_cores=2.0, gpu=j < 2, gpu_memory_bytes=int(2 * GB) if j < 2 else 0),
                 measured=dict(memory_current=int(dm * f * GB), memory_peak=int(dm * min(1.1, f + 0.15) * GB), memory_high=str(int(dm * 0.8 * GB)), memory_high_events=e, oom_events=0, oom_kill_events=0),
                 age_s=600 * (j + 1), heartbeat_at=now)
            for j, (w, dm, f, e) in enumerate([("ladder", 24, 0.62, 0), ("legged", 16, 0.95, 3), ("room", 2, 0.3, 0)])],
    n_leases_total=3, broker_events=[dict(kind="lease_acquired", label="FIXTURE job 0", lease_id="fx-lease-0", mem=int(24 * GB), cpu=2.0, gpu=True, t=now - 900),
                                     dict(kind="admission_stopped", reason="FIXTURE: sustained psi", t=now - 400)],
    watchdog=samples, host=dict(loadavg=["4.2", "4.0", "3.9"], cpus=20, mem_available="90 GB", gpu="host GPU work is off (FIXTURE)")))
dags = []
for name, ws, states in [("fixture_ladder_dag", "ladder", "cccccrrppp"), ("fixture_legged_dag", "legged", "ccfcbpp"), ("fixture_room_dag", "room", "cccc")]:
    m = dict(c="completed", r="running", p="pending", f="failed", b="blocked")
    nodes = [dict(id=f"n{i}", stage=f"stage{i // 3}", state=m[s], lease=None, started=None, ended=None, rc=1 if s == "f" else None, caveat=None) for i, s in enumerate(states)]
    counts = {}
    for n in nodes: counts[n["state"]] = counts.get(n["state"], 0) + 1
    dags.append(dict(track=ws, dag=name, location="repo", nodes=nodes, counts=counts, n_nodes=len(nodes), complete=all(n["state"] == "completed" for n in nodes), eta_statements=[], updated=GEN))
dump("dags", env("dags", dags=dags))

# ---------------------------------------------------------------- edits
edits = []
for body in ["fixture_arm_a", "fixture_quad"]:
    for seed in ["0", "1", "2"]:
        for edit, eff in [("halt", -0.4), ("z_turn", 0.3), ("mirror_active", -0.25), ("mirror_inactive", 0.0)]:
            e = eff + rng.uniform(-0.08, 0.08); w = rng.uniform(0.06, 0.15)
            edits.append(dict(body=body, variant="fixsem", seed=seed, edit=edit, role="control" if edit == "mirror_inactive" else "edit", control=edit == "mirror_inactive",
                              effect=round(e, 3), ci=[round(e - w, 3), round(e + w, 3)], n_pairs=64, kind="effect", metric="forward", decision="D-000",
                              source_file="viz/room/fixtures/edits.json", location="fixture"))
edits.append(dict(body="fixture_quad", variant="fixsem", edit="halt", kind="permutation", metric="forward", p_one_sided=0.004, p_two_sided=0.008, n_perm=252, direction="fixsem<nosem"))
dump("edits", env("edits", rows=edits))

# ---------------------------------------------------------------- training (series embedded: fixtures only)
runs = []
for name, kind in [("fixture_flow_run", "flow"), ("fixture_bc_run", "bc")]:
    steps = list(range(0, 20000, 100))
    loss = [round(2.0 * math.exp(-s / 5000) + 0.1 + rng.uniform(0, 0.03), 4) for s in steps]
    gn = [round(3 * math.exp(-s / 8000) + rng.uniform(0, 0.8) + (4 if 6000 < s < 6800 else 0), 4) for s in steps]
    r = dict(id=name, run=f"FIXTURE/{name}", kind=kind, n_points=len(steps), stride=1, step=steps, losses=dict(loss=loss, flow=[round(x * 0.7, 4) for x in loss]),
             grad_norm=gn, grad_norm_key="grad_norm", clip_scale=[round(min(1.0, 2.0 / g), 4) if g > 0 else 1.0 for g in gn], clip_scale_key="clip_scale",
             lr=[3e-4 * min(1, s / 1000) for s in steps], gate_state="pass" if kind == "flow" else None,
             dagger_rounds=[dict(round=1, step=10000)] if kind == "bc" else [], source_file="viz/room/fixtures/training.json", location="fixture", decision="D-000")
    if kind == "flow": r["alpha"] = [round(min(1, s / 15000), 4) for s in steps]; r["has_alpha"] = True
    runs.append(r)
dump("training", env("training", runs=runs, grad_health=[]))

# ---------------------------------------------------------------- robustness
levels = []
reports = []
for route in ["flow_sem", "bc"]:
    for factor, lv in [("friction", [0.25, 0.5, 1.0, 1.5, 2.0]), ("push_Ns", [0, 5, 10, 20, 40])]:
        for li, x in enumerate(lv):
            n = 20; k = max(0, min(20, int(18 - (abs(li - 2) if factor == "friction" else li) * (4 if route == "bc" else 2.5))))
            lo, hi = wilson(k, n)
            levels.append(dict(route=route, robot="fixture_arm", factor=factor, level=x, k=k, n=n, rate=k / n, ci=[lo, hi], drop=round(0.9 - k / n, 3),
                               motion=dict(joint_jerk_rms=round(rng.uniform(20, 40), 2), penetration_max_m=round(rng.uniform(0, 0.01), 4))))
    reports.append(dict(route=route, robot="fixture_arm", nominal=dict(k=18, n=20, rate=0.9, ci=list(wilson(18, 20)), sources=["fixture"]),
                        break_points=dict(push_Ns=dict(high=10.0 if route == "bc" else 20.0), friction=dict(low=None, high=None)),
                        pooled_perturbed=dict(success=150, n=200, rate=0.75, wilson95=list(wilson(150, 200)), lost=30, gained=5), decision="D-000"))
dump("robustness", env("robustness", levels=levels, reports=reports,
    comparisons=[dict(a="flow_sem", b="bc", robot="fixture_arm", metric="privileged_success", n_seeds=20, n_levels=10, level_mean_a=0.7, level_mean_b=0.6,
                      diff_level_mean=0.1, diff_level_mean_ci=[0.02, 0.18], p_level_mean=0.01, diff_drop=0.02, p_drop=0.6, decision="D-000")],
    variant_diffs=[]))

# ---------------------------------------------------------------- physics, psi0, knowledge, videos
dump("physics", env("physics",
    trackers=[dict(body=f"fixture_body_{i}", tracker_version=f"learned_tracker:fixture_body_{i}:v1", synthetic=True, passed=i != 2,
                   gate=dict(no_fall_rate=1.0, forward_ratio=0.95, turn_ratio=0.7, passed=i != 2),
                   contact_gate=dict(slip_ratio=round(0.1 + 0.15 * i, 3), slip_ok=i == 0, duty_min=0.6, duty_max=0.7, stepping_ok=True, clearance_ok=True, passed=i == 0),
                   modes=dict(forward=dict(cot=round(0.8 + 0.1 * i, 2), slip_mps=0.1, duty_min=0.6, duty_max=0.7, swing_apex_m=0.04)), source_file="fixture") for i in range(3)],
    gates=[dict(gate=f"FIXTURE gate {i}", version="fixture", subject=dict(name="fixture", n=100, bodies=["fixture_body_0"]), verdict=v, decision="D-000",
                criteria=[dict(name="criterion", status=v, value=0.9, threshold=">= 0.95", note="FIXTURE")]) for i, v in enumerate(["pass", "pass", "fail"])],
    dataset_gates=[], gate_tables=[dict(heading="FIXTURE table", header=["kind", "verdict"], rows=[["tracker", "pass"]], source_file="fixture.md", line=1)]))
dump("psi0", env("psi0",
    runs=[dict(run="fixture_run", task="FIXTURE task", level=None, k=3, n=10, rate=0.3, ci=list(wilson(3, 10)), interim=True, interim_reason="FIXTURE")],
    p_decisions=[dict(id="P-001", date="2026-09-20", title="FIXTURE P-decision", body="FIXTURE body linking D-000.", refs_d=["D-000"])],
    crosswalk=[dict(rrp=["D-000"], psi1z=["P-001"], topic="FIXTURE: informs", source_file="docs/related_repos.md", line=1)],
    p_to_d_table=[], rrp_w10_decisions=[], notes_tables=[], notes_markdown="FIXTURE notes", readme_markdown="FIXTURE readme", missing=[]))
dump("knowledge", env("knowledge",
    decisions=[dict(id=f"D-{i:03d}", date=f"2026-09-{20 + i % 8:02d}", title=f"FIXTURE decision {i}", refs_p=[], workstreams=[], paths=[],
                    body=f"FIXTURE body for decision {i}. See D-{max(1, i - 1):03d}.\n\n- a list item\n- `code`", source_file="fixture", line=i) for i in range(1, 12)],
    roadmap=[dict(n="1", question="FIXTURE roadmap item", status="running")], backlog=[dict(item="FIXTURE backlog item", status="planned")],
    workstreams=[dict(id="W0", workstream="FIXTURE workstream", status="FIXTURE", decisions=["D-000"], detail_markdown="- FIXTURE scope")],
    status_markdown="# FIXTURE STATUS\n\nThis is fixture text, not the real STATUS.md.", docs=[dict(path="STATUS.md", title="STATUS")]))
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
    # v1.2 run-history signals (synthetic; exercise the panels)
    import math as _m
    q = sig["joint_pos"]
    sig["joint_vel"] = [[round((q[min(i + 1, n - 1)][j] - q[max(i - 1, 0)][j]) * fps / 2, 4) for j in range(3)] for i in range(n)]
    sig["joint_torque"] = [[round(3 * _m.sin(i / 7 + j), 3) for j in range(3)] for i in range(n)]
    sig["contact_force"] = [[round(4.0 if c else 0.0, 2) for c in cs] for cs in sig["contacts"]]
    sig["contact_force_tangential"] = [[round(0.5 if c else 0.0, 2) for c in cs] for cs in sig["contacts"]]
    sig["contact_pos"] = [[P[i][4] if sig["contacts"][i][0] else None, P[i][5] if sig["contacts"][i][1] else None] for i in range(n)]
    sig["probe_truth"] = dict(contact=[[1.0 if c else 0.0 for c in cs] for cs in sig["contacts"]], halt=[1.0 if e else 0.0 for e in sig["edit_active"]])
    sig["packet_z"] = [[round(_m.sin(i / 9 + d) * (1.5 if sig["edit_active"][i] else 1), 3) for d in range(16)] for i in range(n)]
    sig["packet_norm"] = [[[round(abs(_m.cos(i / 11 + a + k)), 3) for a in range(2)] for k in range(2)] for i in range(n)]
    sig["packet_events"] = [dict(t=T[i], z_norm=1.0, edit=bool(sig["edit_active"][i])) for i in range(0, n, 10)]
    sig["power"] = [round(abs(sum(sig["joint_torque"][i][j] * sig["joint_vel"][i][j] for j in range(3))), 3) for i in range(n)]
    sig["energy"] = [round(p / fps, 4) for p in sig["power"]]
    sig["base_vel"] = [[0.0, 0.0, 0.0] for _ in range(n)]
    sig["object_vel"] = [[round((sig["object_pose"][min(i + 1, n - 1)][k] - sig["object_pose"][i][k]) * fps, 3) for k in range(3)] for i in range(n)]
    sig["edit_dz_norm"] = [round(0.8, 3) if e else None for e in sig["edit_active"]]
    sig["gripper_aperture"] = [round(0.024 if 0.33 < i / (n - 1) < 0.85 else 0.06, 3) for i in range(n)]
    sig["grasp_state"] = ["held" if c[0] else "free" for c in sig["contacts"]]
    meta = dict(family="arm", task="pick_place", body="fixture_arm", route="flow_sem", source_label="fixture:learned:synthetic",
                ckpt_sha=None, variant="v1", seed=1701, condition=condition, success=not halt, failure_stage="place" if halt else None,
                physics=dict(contact_version="contact_v1", grasp_contact_version="grasp_v2", actuator_limits_version="act_v1", actuator_mode="position"),
                decision_refs=["D-131"], caveat="FIXTURE: synthetic kinematics, no physics, not a recorded episode", fixture=True,
                contact_bodies=["finger_l", "finger_r"], base_body="link2", joint_names=["yaw", "shoulder", "elbow"],
                packet_pca_basis=dict(explained_variance_ratio=[0.4, 0.2, 0.1], n_fit=0, fit_on="FIXTURE"),
                joint_torque_names=["yaw", "shoulder", "elbow"], packet_shape=[2, 2, 4], edit_onset_t=(1.8 if halt else None),
                waypoints=dict(a=[0.45, 0.12], b=[0.45, -0.18]))
    doc = dict(schema="rrp-viz/replay/v1", id=rid, meta=meta, fps=fps, n_frames=n, geoms=geoms, bodies=bodies,
               frames=dict(t=T, body_pos=P, body_quat=Q), signals=sig, annotations=[dict(t=T[n // 2], text="FIXTURE annotation")])
    dump(rid, doc, "replays")
    return dict(id=rid, family="arm", task="pick_place", body="fixture_arm", route="flow_sem", source_label=meta["source_label"], variant="v1",
                seed=1701, condition=condition, success=not halt, n_frames=n, fps=fps, file=f"{rid}.json")


idx = [replay("fixture-arm-unedited", "unedited", False), replay("fixture-arm-halt", "ctx_halt", True)]
dump("replays", env("replays", n=len(idx), replays=idx))
