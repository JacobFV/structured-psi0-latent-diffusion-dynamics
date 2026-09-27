"""Checkpoint load audit (W4 gate: "old checkpoints still load after the restructure").

For every `*.pt` under the given roots: `torch.load(weights_only=False)` with the current code, the W3 provenance
(`checkpoint_provenance`, which also verifies stored weight fingerprints), and for the kinds with a loader, the real
loader that builds the models and loads the weights strictly:
  arm representation      -> rrp.controllers.bundles.load_representation
  arm latent flow         -> rrp.controllers.latent_runner.LatentPolicy.from_checkpoint
  arm direct/codec policy -> rrp.controllers.policy_runner.LearnedPolicy.from_checkpoint
  legged representation   -> rrp.controllers.bundles.load_rep
  legged flow / refit R   -> rrp.evaluation.legged_latent_eval.LatentLeggedController
  legged BC               -> rrp.models.legged_bc.load_bc
Training-resume states, trackers and probes are loaded with torch.load only (no rrp classes are pickled in any of them).
Relative paths inside checkpoints (e.g. a flow's `representation`) resolve against the checkout that holds the file
(the directory above its `artifacts/`).

    python -m rrp.evaluation.checkpoint_audit [--per-kind N] [--out report.json] ROOT [ROOT ...]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickletools
import sys
import time
import warnings
import zipfile
from pathlib import Path


def pickled_globals(path: Path) -> set[tuple[str, str]]:
    """(module, name) of every class/function a torch zip checkpoint's pickle references."""
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        data = z.read(name)
    out, strs = set(), []
    for op, arg, _ in pickletools.genops(data):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
            strs.append(arg)
        elif op.name == "STACK_GLOBAL":
            out.add((strs[-2], strs[-1]))
        elif op.name == "GLOBAL":
            out.add(tuple(arg.split(" ", 1)))
    return out


def classify(st) -> str:
    if not isinstance(st, dict):
        return "other"
    k = set(st)
    if {"config", "model", "versions"} <= k:
        cfg = st.get("config") or {}
        if "latent" in cfg and isinstance(st["model"], dict) and {"E", "R", "P"} <= set(st["model"]):
            return "arm_representation" if (st.get("extra") or {}).get("result") else "resume_state"
        if "representation" in cfg and "policy" in cfg:
            return "arm_latent_flow"
        if "policy" in cfg:
            return "arm_policy"
        return "arm_checkpoint"
    if {"E", "R", "P", "cfg", "result"} <= k:
        return "legged_representation"
    if {"cfg", "flow", "result"} <= k:
        return "legged_flow"
    if {"R", "cfg", "representation"} <= k:
        return "legged_refit_realizer"
    if {"cfg", "model"} <= k and "opt" not in k and isinstance(st.get("cfg"), dict) and "latent" not in st["cfg"]:
        return "legged_bc"
    if {"actor", "obs_mean"} <= k:
        return "tracker"
    if "opt" in k or "optimizer" in k:
        return "resume_state"
    return "other"


def _root_of(path: Path) -> Path:
    parts = path.resolve().parts
    return Path(*parts[:parts.index("artifacts")]) if "artifacts" in parts else path.parent


def load_one(path: Path) -> dict:
    import torch
    row = dict(path=str(path))
    t0 = time.time()
    globs = {g for g in pickled_globals(path) if g[0].startswith("rrp") or g[0] == "__main__"}
    row["rrp_pickled_classes"] = sorted(".".join(g) for g in globs)
    st = torch.load(str(path), map_location="cpu", weights_only=False)
    kind = row["kind"] = classify(st)
    with contextlib.chdir(_root_of(path)):
        if kind.startswith("arm"):
            from rrp.models.checkpoint import checkpoint_provenance
            prov = checkpoint_provenance(st, path)
            row["provenance"] = dict(legacy=prov.legacy, notes=prov.notes)
            if kind == "arm_representation":
                from rrp.controllers.bundles import load_representation
                _, E, R, P, res = load_representation(path, "cpu")
                row["loaded"] = dict(latent_space_version=res["latent_space_version"])
            elif kind == "arm_latent_flow":
                from rrp.controllers.latent_runner import LatentPolicy
                pol = LatentPolicy.from_checkpoint(path, device="cpu")
                row["loaded"] = dict(latent_space_version=pol.lsv)
            elif kind == "arm_policy":
                from rrp.controllers.policy_runner import LearnedPolicy
                LearnedPolicy.from_checkpoint(path, device="cpu")
                row["loaded"] = "LearnedPolicy"
        elif kind.startswith("legged"):
            from rrp.controllers.bundles import checkpoint_provenance
            prov = checkpoint_provenance(st, path)
            row["provenance"] = dict(legacy=prov.legacy, notes=prov.notes)
            if kind == "legged_representation":
                from rrp.controllers.bundles import load_rep
                load_rep(path, "cpu")
                row["loaded"] = "load_rep"
            elif kind == "legged_flow":
                from rrp.evaluation.legged_latent_eval import LatentLeggedController
                c = LatentLeggedController(path, "cpu")
                row["loaded"] = dict(flow=c.F is not None)
            elif kind == "legged_refit_realizer":
                from rrp.evaluation.legged_latent_eval import LatentLeggedController
                LatentLeggedController(None, "cpu", rep=st["representation"], realizer=path)
                row["loaded"] = "refit realizer"
            elif kind == "legged_bc":
                from rrp.models.legged_bc import load_bc
                load_bc(path, "cpu")
                row["loaded"] = "load_bc"
    row["seconds"] = round(time.time() - t0, 2)
    return row


def find(roots) -> list[Path]:
    out = set()
    for r in roots:
        for p in Path(r).rglob("*.pt"):
            if p.is_file():                       # skips dangling symlinks
                out.add(p.resolve())
    return sorted(out)


def audit(roots, per_kind: int | None = None) -> dict:
    """Load every checkpoint (or the first `per_kind` of each kind). A row with `error` is a failure."""
    import torch
    rows, taken = [], {}
    for p in find(roots):
        if per_kind is not None:
            try:
                kind = classify(torch.load(str(p), map_location="cpu", weights_only=False))
            except Exception as e:  # noqa: BLE001
                rows.append(dict(path=str(p), error=f"torch.load: {type(e).__name__}: {e}"))
                continue
            if taken.get(kind, 0) >= per_kind:
                continue
            taken[kind] = taken.get(kind, 0) + 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rows.append(load_one(p))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(path=str(p), error=f"{type(e).__name__}: {e}"))
    kinds = {}
    for r in rows:
        kinds[r.get("kind", "error")] = kinds.get(r.get("kind", "error"), 0) + 1
    return dict(n=len(rows), n_errors=sum("error" in r for r in rows), kinds=kinds,
                rrp_pickled_classes=sorted({c for r in rows for c in r.get("rrp_pickled_classes", [])}), rows=rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--per-kind", type=int, default=None, help="load at most N checkpoints of each kind")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rep = audit(a.roots, a.per_kind)
    txt = json.dumps(rep, indent=1, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt)
    print(json.dumps({k: rep[k] for k in ("n", "n_errors", "kinds", "rrp_pickled_classes")}, indent=1))
    for r in rep["rows"]:
        if "error" in r:
            print("ERROR", r["path"], r["error"], file=sys.stderr)
    return 1 if rep["n_errors"] else 0


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.exit(main())
