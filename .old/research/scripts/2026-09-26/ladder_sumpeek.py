"""Print compact ladder summaries: ladder_sumpeek.py <summary.json>..."""
import json, sys
for f in sys.argv[1:]:
    d = json.load(open(f))
    oc = {}
    print(f"{f}: {d['success']}/{d['n']} wilson={[round(x,2) for x in d.get('wilson95',[0,0])]} fail={d['failed_stage']} "
          f"minTC={d['min_tcp_cube_m']:.3f} trackq={d['track_q_rad']:.3f} trackTCP={d['track_tcp_m']:.3f} step={d['cmd_step_rad']:.3f}")
