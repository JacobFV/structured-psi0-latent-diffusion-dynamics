"""Refit system 0 only on a frozen Stage-A encoder: ladder_refit.py <config.json> (see latent_train.refit_realizer)."""
import json, sys
from pathlib import Path
from rrp.learning.latent_train import refit_realizer
cfg = json.loads(Path(sys.argv[1]).read_text())
if len(sys.argv) > 2:
    cfg.update(json.loads(sys.argv[2]))
print(json.dumps(refit_realizer(cfg, Path(cfg["out_dir"])), indent=1, default=str)[:3000])
