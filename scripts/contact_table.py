"""Markdown table of tracker validations (contact track). usage: python scripts/contact_table.py file.json ..."""
import json, sys


def f(x, n=2):
    return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


print("| body | tracker | physics | no-fall | fwd ratio | turn ratio | fwd legacy slip | fwd cp slip | body speed | "
      "slip ratio | duty min/max | air s | apex m | CoT fwd | gate | contact gate |")
print("|" + "---|" * 16)
for p in sys.argv[1:]:
    r = json.load(open(p))
    g, s = r["gate"], r["summary"]["forward"]
    cg = g.get("contact_gate", {})
    tag = p.split("/")[-1].replace(".json", "").replace(r["body"] + "_", "")
    print(f"| {r['body']} | {tag.rsplit('_phys', 1)[0]} | {r.get('contact_model', 'contact_v1')} | {f(g['no_fall_rate'])} | "
          f"{f(g['forward_ratio'])} | {f(g['turn_ratio'])} | {f(s['slip_mps'])} | {f(s.get('slip_cp_mps'))} | "
          f"{f(s.get('body_speed_mps'))} | {f(s.get('slip_ratio'))} | {f(cg.get('duty_min'))}/{f(cg.get('duty_max'))} | "
          f"{f(s.get('air_time_s'))} | {f(s.get('swing_apex_m'), 3)} | {f(s.get('cot'))} | {g['passed']} | {cg.get('passed')} |")
