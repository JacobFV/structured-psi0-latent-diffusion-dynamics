import random, tempfile, torch
from pathlib import Path
from rrp.learning.packed import pack_dataset
from rrp.learning.latent_train import LatentData, representation_step
from rrp.model.semantic_latent import LatentConfig, TargetEncoder
from rrp.control.latent_realizer import LatentRealizer
from rrp.model.latent_probes import PacketProbe
ds = Path("/home/brandonin/work/relational-robot-policy/artifacts/datasets/pick_place_primary_v3dart")
with tempfile.TemporaryDirectory() as d:
    print(pack_dataset(ds, Path(d), {"panda_pg2", "parm5_tf3"}, 16, stride=1, limit_per_robot=2, include_dart_failures=True)["n"])
    data = LatentData(Path(d))
    cfg = LatentConfig(width=64, heads=2, ctx_layers=1, enc_layers=1, dz=16)
    E, R, P = TargetEncoder(cfg), LatentRealizer(cfg.dz, width=64), PacketProbe(cfg.dz, cfg.knots, width=64)
    sel, tgt, j = data.sample(8, random.Random(0), cfg.max_phase_ticks)
    b, a, v, lab, r = data.fetch(sel, tgt, "cpu")
    loss, logs, _ = representation_step(E, R, P, (b, a, v, lab, r, torch.as_tensor(j)), cfg)
    loss.backward()
    ok = lambda m: sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
    print("loss", float(loss), "grads E/R/P", ok(E) > 0, ok(R) > 0, ok(P) > 0, {k: round(v, 3) for k, v in logs.items()})
    print("subtask labels", lab["subtask"].tolist(), "local", r["local"][:2].tolist())
