"""Training-side probe: clip fraction / ratio drift per optimizer step vs lr and last-step treatment.
Uses TRAINING seeds only (2000000+), never eval seeds."""
import copy, json, torch
from rrp.learning.adapt import load_policy, scenario_factory, SeedStream
from rrp.learning.flow_sde import SDEConfig
from rrp.learning.grpo import GRPOConfig, GRPOLearner, build_samples
from rrp.learning.rollout import SDEPolicy
from rrp.learning.branching import collect_plain, SHAPING
from rrp.learning.flow_sde import SDEPath, path_log_prob
from rrp.model.batch import collate_inputs
SHAPING["dist"] = 0.5
torch.manual_seed(0)
base, codec, _ = load_policy("artifacts/runs/adapt_start_bc_xarm7pg2_only_v2/policy.pt", "cpu")
make = scenario_factory("xarm7_pg2")
for last in ["stochastic", "deterministic"]:
    sde = SDEConfig(nfe=8, noise_level=0.5, last_step=last)
    m0 = copy.deepcopy(base)
    lr0 = GRPOLearner(m0, GRPOConfig(sde=sde), "cpu", version_prefix="p")
    actor = SDEPolicy(m0, None, "cpu", sde, version=lr0.version, seed=1)
    groups = collect_plain(actor, make, SeedStream(make, 2000000).take(3), 8, 300, prefix_source="teacher")
    samples, info = build_samples([dict(returns=g.returns, members=g.members) for g in groups], GRPOConfig())
    print(last, "samples", len(samples), info, flush=True)
    if not samples: continue
    for lr in [3e-5, 1e-5, 3e-6, 1e-6]:
        m = copy.deepcopy(base)
        L = GRPOLearner(m, GRPOConfig(sde=sde, lr=lr, epochs=1, minibatch=16), "cpu", version_prefix="p")
        mb = samples[:16]
        batch = collate_inputs([s["pi"] for s in mb]); path = SDEPath.cat([s["path"] for s in mb])
        out = []
        for step in range(4):
            L.update([dict(s, path=s["path"]) for s in mb]); L.iteration = 0
            with torch.no_grad():
                cache = m.prepare(batch)
                lp = path_log_prob(lambda z, t: m.velocity(z, t, cache), path)
            r = torch.exp(lp - path.old_log_prob)
            out.append(dict(step=step + 1, clip=round(float(((r < 0.8) | (r > 1.2)).float().mean()), 3),
                            med_abs_logratio=round(float((lp - path.old_log_prob).abs().median()), 3)))
        print(last, "lr", lr, json.dumps(out), flush=True)
