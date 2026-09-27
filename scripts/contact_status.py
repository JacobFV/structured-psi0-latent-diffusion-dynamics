"""Progress of contact_v2 tracker runs: last gate windows per run. usage: python3 scripts/contact_status.py <train_log.jsonl>..."""
import json, sys
for p in sys.argv[1:]:
    try:
        L = [json.loads(l) for l in open(p)]
    except FileNotFoundError:
        print(p, "missing"); continue
    g = [r for r in L if r.get("gate")]
    print(f"{p}: iter {L[-1]['iter']} alpha {L[-1].get('alpha')} s/iter {L[-1]['iter_s']:.2f}")
    for r in g[-3:]:
        x = r["gate"]
        print(f"   it{r['iter']} {x['action']:8s} fall {x['fall_rate']:.2f} trk_err {x.get('track_rel_err', 0):.2f} "
              f"slip {x.get('slip_ratio', 0):.2f} cot {x.get('cot', 0):.2f} ep_len {r['ep_len']}")
