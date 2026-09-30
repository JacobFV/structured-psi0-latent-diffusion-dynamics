"""Print localization summaries (+ gate ratio = sys0 arm err / hold-still): ladder_locpeek.py <json>..."""
import json, sys
for f in sys.argv[1:]:
    d = json.load(open(f))["summary"]
    r = d["sys0_bcoracle_err_arm"] / d["hold_still_ref_arm"]
    g = d.get("sys0_gen_err_arm")
    print(f.split("/")[-2], f.split("/")[-1].replace("bc_direct1701_u12000__", "").replace(".json", ""),
          f"gate={r:.2f} arm={d['sys0_bcoracle_err_arm']:.4f} grip={d['sys0_bcoracle_err_grip']:.3f} hold={d['hold_still_ref_arm']:.4f}",
          f"gen_arm={g:.4f} gen_ratio={g / d['hold_still_ref_arm']:.2f} zrel={d['z_gen_vs_bcoracle_rel']:.2f}" if g else "",
          f"bc={d['bc_success']}/{d['n']} replay_ok={d['replay_consistent']}")
