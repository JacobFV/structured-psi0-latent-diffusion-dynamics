"""Build a loadable representation bundle from a realizer refit snapshot (rz_last.pt) + the frozen Stage-A bundle
(for testing a refit mid-training): ladder_bundle.py <rz_last.pt> <representation.pt> <out.pt>"""
import sys
from rrp.learning.checkpoint import load_checkpoint, save_checkpoint
rz, rep, out = sys.argv[1:4]
r = load_checkpoint(rz, map_location="cpu")
st = load_checkpoint(rep, map_location="cpu")
model = dict(st["model"], R=r["model"])


class _B:
    def state_dict(self):
        return model


res = dict(st["extra"]["result"], refit_snapshot=rz, refit_step=r["step"], refit_of=rep)
save_checkpoint(out, model=_B(), optimizer=None, step=r["step"], versions=st.get("versions", {}),
                config=dict(st["config"], refit=r["config"]), extra=dict(result=res))
print(out, r["step"])
