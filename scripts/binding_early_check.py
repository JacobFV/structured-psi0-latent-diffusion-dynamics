"""Early read on a Stage-A checkpoint in progress (rep_last.pt): fixed counterexample with the jointly trained probe.
Diagnostic only (jointly trained P; nosem's P is untrained); final numbers use post-hoc probes on representation.pt."""
import json, sys
from pathlib import Path
import torch
from rrp.learning.checkpoint import load_checkpoint
from rrp.model.semantic_latent import LatentConfig, TargetEncoder
from rrp.model.latent_probes import PacketProbe
from rrp.evaluation.latent_counterfactuals import counterexample
st = load_checkpoint(Path(sys.argv[1]), map_location="cpu")
cfg = LatentConfig(**st["config"]["latent"])
E, P = TargetEncoder(cfg), PacketProbe(cfg.dz, cfg.knots)
E.load_state_dict(st["model"]["E"]); P.load_state_dict(st["model"]["P"]); E.eval(); P.eval()
r = counterexample(E, P, Path("artifacts/datasets/pick_place_primary_v3dart"), n=30, per_episode=2)
r.pop("rows")
print(json.dumps(dict(ckpt=sys.argv[1], step=st["step"], **r)))
