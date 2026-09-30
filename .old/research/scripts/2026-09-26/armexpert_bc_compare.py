"""W7 step 2: v2-data BC expert (seeds 1701/1702, u12000 + final) vs the v1-data BC expert direct1701 on the sprint_bc
evaluation sets, plus motion metrics of the BC rollouts. Reads a local copy of the peer store (argv[1] = runs root)."""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
out = Path(sys.argv[2])


def load(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None


def wil(s):
    return f"{s['success']}/{s['n']}" if s else "-"


def cell(c):
    if not c:
        return "-"
    return f"{c['successes']}/{c['attempted']}"


def ho(s):   # evaluate_checkpoint summary (per robot) -> pooled k/n
    if not s:
        return "-"
    parts = {r: (v["successes"], v["attempted"]) for r, v in s.items() if r != "_meta"}
    return " ".join(f"{r} {k}/{n}" for r, (k, n) in parts.items())


rows = []
V1 = dict(label="v1 data: learned:direct1701 (sprint_bc)", ck={
    "u12000": dict(dev={r: load(f"baselines_bc_ladder/{r}/learned_direct1701_u12000.summary.json") for r in ("panda_pg2", "parm6_tf3")},
                   fresh={r: load(f"ladder_v1/{r}/learned_bc_direct1701_u12000_fresh3000200.summary.json") for r in ("panda_pg2", "parm6_tf3")},
                   heldout=load("baselines_bc_ladder/heldout/direct1701_u12000.summary.json")),
    "final": dict(dev={r: load(f"baselines_bc_ladder/{r}/learned_direct1701_ufinal.summary.json") for r in ("panda_pg2", "parm6_tf3")},
                  fresh={}, heldout_cell=load("latent_slice1_b1fix/baseline_direct_action/seed1701/cells/source.json"),
                  b0={t: load(f"latent_slice1_b1fix/baseline_direct_action/seed1701/cells/{t}_b0.json")
                      for t in ("panda_tf3", "xarm7_pg2", "xarm7_tf3")})})
res = {"v1_s1701": V1}
for seed in (1701, 1702):
    E, C = "armexpert_bcv2_eval", f"armexpert_bcv2/baseline_direct_action/seed{seed}"
    d = {}
    for ck in ("u12000", "final"):
        tag = f"bcv2_direct{seed}_{ck}"
        d[ck] = dict(dev={r: load(f"{E}/{r}/learned_{tag}_s3000000.summary.json") for r in ("panda_pg2", "parm6_tf3")},
                     fresh={r: load(f"{E}/{r}/learned_{tag}_s3000200.summary.json") for r in ("panda_pg2", "parm6_tf3")},
                     heldout=load(f"{E}/heldout/{tag}.summary.json") if ck == "u12000" else None,
                     heldout_cell=load(f"{C}/cells/source.json") if ck == "final" else None,
                     b0={t: load(f"{C}/cells/{t}_b0.json") for t in ("panda_tf3", "xarm7_pg2", "xarm7_tf3")} if ck == "final" else {})
    tr = load(f"{C}/source/result.json")
    res[f"v2_s{seed}"] = dict(label=f"v2 data: learned:bcv2_direct{seed}", ck=d, train=tr)
json.dump(res, open(out, "w"), indent=1, default=str)

print("| checkpoint | panda_pg2 dev | parm6_tf3 dev | panda_pg2 fresh 3,000,200 | parm6_tf3 fresh | held-out source (parm5s_tf3, parm5l_pg2) | b0 panda_tf3 / xarm7_pg2 / xarm7_tf3 | cmd_step_rad dev (panda / parm6) |")
print("|---|---|---|---|---|---|---|---|")
for key, e in res.items():
    for ck, d in e["ck"].items():
        hs = ho(d.get("heldout")) if d.get("heldout") else (cell(d.get("heldout_cell")) + " pooled" if d.get("heldout_cell") else "-")
        b0 = " / ".join(cell((d.get("b0") or {}).get(t)) for t in ("panda_tf3", "xarm7_pg2", "xarm7_tf3")) if d.get("b0") else "-"
        cs = " / ".join(f"{d['dev'][r]['cmd_step_rad']:.3f}" if d["dev"].get(r) else "-" for r in ("panda_pg2", "parm6_tf3"))
        print(f"| {e['label']} {ck} | {wil(d['dev'].get('panda_pg2'))} | {wil(d['dev'].get('parm6_tf3'))} | "
              f"{wil(d['fresh'].get('panda_pg2'))} | {wil(d['fresh'].get('parm6_tf3'))} | {hs} | {b0} | {cs} |")
