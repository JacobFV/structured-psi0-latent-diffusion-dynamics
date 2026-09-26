"""ARM NOSEM COUNTERPART: like-for-like sem vs nosem comparison from raw outputs.
R2 rows: ladder_v1/<robot>/<prefix>_<tag>.summary.json; edit suites: semantic_rows_generated.jsonl shards.
Usage: armnosem_compare.py --ladder DIR --sem-parm6 ROWS... --nosem-parm6 ROWS... --sem-panda ROWS... --nosem-panda ROWS...
       --out OUT.json"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rrp.evaluation import latent_semantic_edits as se  # noqa: E402
from rrp.evaluation.statistics import wilson  # noqa: E402

R2 = {  # (sem tag, nosem tag) per seed set; the sem dev-seed file has no _s suffix for the 3,000,000 set
    "final": ("generated_zero_flowgdag2h_rzgendag3_noqd_s{s}", "generated_zero_flownsgdag2h_rznsgendag3_noqd_s{s}"),
    "flow20k_gendag1": ("generated_zero_ladder_flow_jointfix_snap_final_s20000_rzgendag1noqd", "generated_zero_flowns20k_rznsgendag1_noqd_s3000000"),
    "flowgdag1_gendag3": ("generated_zero_flowgdag1_rzgendag3_noqd", "generated_zero_flownsgdag1_rznsgendag3_noqd_s3000000"),
    "r1_stateless_gendag3": ("oracle_zero_gendag3noqd_orcbc", "oracle_zero_nsjfgendag3noqd_orcbc"),
}


def load(d, robot, name):
    p = Path(d) / robot / f"{name}.summary.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    return dict(k=j["success"], n=j["n"], seeds=j["seeds"], failed_stage=j["failed_stage"], path=str(p))


def newcombe(k1, n1, k2, n2):
    """95% CI for p1 - p2 (Newcombe hybrid score)."""
    l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    return [d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)]


def rate(k, n):
    return dict(k=k, n=n, rate=k / n if n else None, wilson95=wilson(k, n) if n else None)


def r2_table(d):
    out = {}
    for robot in ("panda_pg2", "parm6_tf3", "parm5s_tf3", "parm5l_pg2"):
        sets = (3000000, 3000100, 3000200) if robot in ("panda_pg2", "parm6_tf3") else (3000000,)
        rows = {}
        for s in sets:
            st, nt = R2["final"]
            sem = load(d, robot, st.format(s=s)) or (load(d, robot, "generated_zero_flowgdag2h_rzgendag3_noqd") if s == 3000000 else None)
            rows[s] = dict(sem=sem, nosem=load(d, robot, nt.format(s=s)))
        ok = [r for r in rows.values() if r["sem"] and r["nosem"]]
        ks, ns = sum(r["sem"]["k"] for r in ok), sum(r["sem"]["n"] for r in ok)
        kn, nn = sum(r["nosem"]["k"] for r in ok), sum(r["nosem"]["n"] for r in ok)
        out[robot] = dict(per_set=rows, pooled=dict(sem=rate(ks, ns), nosem=rate(kn, nn),
                                                    diff_nosem_minus_sem=(kn / nn - ks / ns) if ns and nn else None,
                                                    diff95=newcombe(kn, nn, ks, ns) if ns and nn else None))
        for key in ("flow20k_gendag1", "flowgdag1_gendag3", "r1_stateless_gendag3"):
            if robot in ("panda_pg2", "parm6_tf3"):
                out[robot][key] = dict(sem=load(d, robot, R2[key][0]), nosem=load(d, robot, R2[key][1]))
    return out


def rows_of(paths):
    return [json.loads(l) for p in paths for l in open(p) if l.strip()]


def per_seed(rows):
    ok = [r for r in rows if "skipped" not in r]
    by = {}
    for r in ok:
        by.setdefault(r["seed"], {})[r["condition"]] = r
    return {s: v for s, v in by.items() if "control" in v}


def edit_key(summ, rows):
    """Headline metrics of the D-074/075/077 suite."""
    g, rb = summ.get("goal_shift", {}), summ.get("rebind_desc", {})
    c = summ.get("control", {})
    con = summ.get("_contrasts", {})
    ctl_goal = {k: summ[k]["cube_at_shifted_goal"] for k in ("control", "irrelevant_distractor", "orthogonal_matched",
                                                             "control_replay") if k in summ}
    return dict(
        n_feasible_seeds=c.get("n"),
        control_success=c.get("privileged_success"), control_cube_lifted=c.get("cube_lifted"),
        goal_cube_at_new_goal=rate(g.get("cube_at_shifted_goal", 0), g.get("n", 0)) if g else None,
        goal_controls_at_new_goal=ctl_goal,
        goal_effect_beyond_irrelevant_m=con.get("goal_shift-irrelevant_distractor:goal_pref"),
        goal_effect_beyond_orthogonal_m=con.get("goal_shift-orthogonal_matched:goal_pref"),
        goal_effect_beyond_replay_m=con.get("goal_shift-control_replay:goal_pref"),
        rebind_original_cube_lifted=rate(rb.get("cube_lifted", 0), rb.get("n", 0)) if rb else None,
        rebind_new_cube_lifted=rate(rb.get("distractor0_lifted", 0), rb.get("n", 0)) if rb else None,
        rebind_new_cube_in_zone=rb.get("distractor0_in_zone"),
        rebind_first_approach_new=rb.get("approached_first_new_frac"),
        rebind_first_contact_new=rb.get("first_contact_new_frac"),
        control_first_contact_new=c.get("first_contact_new_frac"),
        rebind_mindist_beyond_irrelevant_m=con.get("rebind_desc-irrelevant_distractor:pref_min"),
        rebind_mindist_beyond_orthogonal_m=con.get("rebind_desc-orthogonal_matched:pref_min"),
        rebind_mindist_beyond_replay_m=con.get("rebind_desc-control_replay:pref_min"),
    )


def seed_effects(rows):
    """Per-seed paired effects (edit minus irrelevant control) for sem-vs-nosem bootstrap of the DIFFERENCE."""
    ps = per_seed(rows)
    goal, mind = {}, {}
    for s, v in ps.items():
        q = v["control"]
        if "goal_shift" in v and "irrelevant_distractor" in v:
            a, b, c = se._goal_pref(v["goal_shift"]), se._goal_pref(v["irrelevant_distractor"]), se._goal_pref(q)
            if not (a is None and b is None):
                goal[s] = ((a or 0.0) - (c or 0.0)) - ((b or 0.0) - (c or 0.0))
        if "rebind_desc" in v and "irrelevant_distractor" in v:
            a, b, c = se._pref_min(v["rebind_desc"]), se._pref_min(v["irrelevant_distractor"]), se._pref_min(q)
            if not (a is None and b is None):
                mind[s] = ((a or 0.0) - (c or 0.0)) - ((b or 0.0) - (c or 0.0))
    return goal, mind


def boot_diff(a, b, n=4000, seed=0):
    """mean(a) - mean(b) with a percentile bootstrap (independent resampling; the runs are independent)."""
    a, b = np.asarray(list(a), float), np.asarray(list(b), float)
    if len(a) == 0 or len(b) == 0:
        return None
    r = np.random.default_rng(seed)
    d = [r.choice(a, len(a)).mean() - r.choice(b, len(b)).mean() for _ in range(n)]
    return dict(mean=float(a.mean() - b.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                n_a=len(a), n_b=len(b))


def edit_compare(sem_rows, nosem_rows):
    out = {}
    for name, rows in (("sem", sem_rows), ("nosem", nosem_rows)):
        summ = se.summarize_semantic(rows)
        out[name] = dict(key=edit_key(summ, rows), seeds_complete=len(per_seed(rows)), n_rows=len(rows))
    gs, ms = seed_effects(sem_rows)
    gn, mn = seed_effects(nosem_rows)
    common = sorted(set(per_seed(sem_rows)) & set(per_seed(nosem_rows)))
    out["common_feasible_seeds"] = len(common)
    out["diff_nosem_minus_sem"] = dict(
        goal_effect_beyond_irrelevant_m=boot_diff(gn.values(), gs.values()),
        rebind_mindist_beyond_irrelevant_m=boot_diff(mn.values(), ms.values()))
    ks, kn = out["sem"]["key"], out["nosem"]["key"]
    for m in ("goal_cube_at_new_goal", "rebind_original_cube_lifted", "rebind_new_cube_lifted"):
        a, b = kn.get(m), ks.get(m)
        if a and b and a["n"] and b["n"]:
            out["diff_nosem_minus_sem"][m] = dict(diff=a["rate"] - b["rate"], ci95=newcombe(a["k"], a["n"], b["k"], b["n"]))
    for m in ("rebind_first_contact_new", "rebind_first_approach_new"):
        a, b = kn.get(m), ks.get(m)
        if a and b and a["n"] and b["n"]:
            out["diff_nosem_minus_sem"][m] = dict(diff=a["k"] / a["n"] - b["k"] / b["n"], ci95=newcombe(a["k"], a["n"], b["k"], b["n"]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    for k in ("sem-parm6", "nosem-parm6", "sem-panda", "nosem-panda"):
        ap.add_argument(f"--{k}", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = dict(r2=r2_table(a.ladder))
    if a.sem_parm6 and a.nosem_parm6:
        res["edits_parm6"] = edit_compare(rows_of(a.sem_parm6), rows_of(a.nosem_parm6))
    if a.sem_panda and a.nosem_panda:
        res["edits_panda"] = edit_compare(rows_of(a.sem_panda), rows_of(a.nosem_panda))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:6000])
