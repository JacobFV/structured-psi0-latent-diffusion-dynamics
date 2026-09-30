"""W4 numerics parity: identical closed-loop outputs before/after the restructure (old import paths, fixed seeds).
Run with cwd = the main checkout (checkpoints), PYTHONPATH = the code tree under test. argv[1] = output json."""
import hashlib, json, random, sys, warnings
warnings.simplefilter("ignore")
import numpy as np, torch
torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0); random.seed(0)


def digest(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return "nd:" + hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]
    if isinstance(x, dict):
        return {k: digest(v) for k, v in x.items() if not any(t in k for t in ("wall", "latenc", "time_s", "elapsed"))}
    if isinstance(x, (list, tuple)):
        return [digest(v) for v in x]
    if isinstance(x, float):
        return repr(x)
    return x if isinstance(x, (int, str, bool, type(None))) else repr(type(x))


import os; os.chdir(sys.argv[3])
out = {}
# ---- arm: frozen jointfix sem flow + its Stage A system 0, 1 episode, 80 ticks
from rrp.policy.latent_runner import LatentPolicy
from rrp.learning.latent_train import load_representation
from rrp.learning.latent_grpo import run_episodes
pol = LatentPolicy.from_checkpoint("artifacts/runs/ladder_flow_jointfix/snap_final_s20000.pt", device="cpu", seed=0)
_, E, R, P, res = load_representation("artifacts/runs/ladder_latent_sem_b1fix_anchor/representation.pt", "cpu")
eps = run_episodes(pol, R, "panda_pg2", [3000100], max_steps=80, record=True)
out["arm"] = digest(eps)
out["arm_ids"] = [pol.lsv, pol.rcv, res["latent_space_version"]]
# ---- legged: W3 smoke checkpoints (tiny), go2, 2 s
from rrp.evaluation.legged_latent_eval import LatentLeggedController, run_episode
D = sys.argv[2]
ctl = LatentLeggedController(f"{D}/legged_flow/policy.pt", "cpu", seed=0, rep=f"{D}/legged_rep/representation.pt")
row, _ = run_episode(ctl, "go2", 10000, max_s=2.0)
out["legged"] = digest(row)
json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True, default=str)
print("arm", eps[0]["outcome"], eps[0]["steps"], "packets", eps[0]["packets"], "| legged", row.get("success"), row.get("stats"))
