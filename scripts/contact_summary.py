"""v1 vs v2 summary table (contact track). usage: python3 scripts/contact_summary.py"""
import json
V = "artifacts/runs/contact_v2/val/"
ROWS = [("t1", "v2trk"), ("h1", "v2trk-r2"), ("g1", "v2trk-r1"), ("go2", "v2trk"), ("anymal_c", "v2trk"),
        ("hexapod6", "v2trk-v2c-rejected")]


def load(p):
    r = json.load(open(p)); g = r["gate"]; s = r["summary"]["forward"]
    return dict(nf=g["no_fall_rate"], fwd=g["forward_ratio"], turn=g["turn_ratio"], slip=s["slip_ratio"], duty=(g["contact_gate"]["duty_min"], g["contact_gate"]["duty_max"]),
                apex=s["swing_apex_m"], cot=s["cot"], gate=g["passed"], cgate=g["contact_gate"]["passed"])


f = lambda x, n=2: "-" if x is None else f"{x:.{n}f}"
print("| body | v2 tracker | no-fall v1 -> v2 | fwd ratio v1 -> v2 | turn ratio v1 -> v2 | **slip ratio** v1 -> v2 | duty v1 -> v2 | "
      "swing apex cm v1 -> v2 | CoT v1 -> v2 | legacy gate v1 / v2 | contact gate v2 |")
print("|" + "---|" * 11)
for b, tag in ROWS:
    a = load(f"{V}{b}_v1trk_physv1.json"); c = load(f"{V}{b}_{tag}_physv2.json")
    print(f"| {b} | {tag} | {f(a['nf'])} -> {f(c['nf'])} | {f(a['fwd'])} -> {f(c['fwd'])} | {f(a['turn'])} -> {f(c['turn'])} | "
          f"{f(a['slip'])} -> **{f(c['slip'])}** | {f(a['duty'][0])}-{f(a['duty'][1])} -> {f(c['duty'][0])}-{f(c['duty'][1])} | "
          f"{f(a['apex'] * 100, 1)} -> {f(c['apex'] * 100, 1)} | {f(a['cot'])} -> {f(c['cot'])} | {a['gate']} / {c['gate']} | {c['cgate']} |")
