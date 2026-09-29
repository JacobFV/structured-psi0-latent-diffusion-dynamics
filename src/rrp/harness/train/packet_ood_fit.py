"""Fit / score the per-bundle packet OOD detector (D-126 roadmap #29; model in rrp.controllers.packet_ood).

  fit-rep      TRAINING packets = the frozen encoder's posterior means E(ctx, demonstrated targets) on the training
               split of the bundle's own data (what system 0 was trained to realize); calibration = the held-out split
               (every 20th episode, LeggedData's split). One model per body (assembly count).
  fit-packets  packets recorded by `legged_latent_eval --record-packets DIR` on UNEDITED runs (edit == none).
  score        score recorded packets (e.g. a D-092-style edit suite run with --record-packets) and report, per
               episode, the max score, whether it crossed the thresholds, and the ROC AUC of "max score" for
               predicting falls and for separating edited from unedited episodes.

usage:
  python -m rrp.harness.train.packet_ood_fit fit-rep --rep R/representation.pt --body t1 --out R/packet_ood_t1 [--n-fit 4000]
  python -m rrp.harness.train.packet_ood_fit fit-packets --packets DIR --body t1 --out R/packet_ood_t1_gen
  python -m rrp.harness.train.packet_ood_fit score --model R/packet_ood_t1 --packets DIR --out scores.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def auroc(pos, neg) -> float | None:
    """Probability that a random positive scores above a random negative (ties count 1/2)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def encoder_packets(rep: Path, body: str, n_fit: int, n_cal: int, seed: int = 0, data_root: str | None = None,
                    device: str = "cpu"):
    import torch
    from rrp.policies.bundles import load_rep
    from rrp.policies.legged import legged_bundle_versions
    from rrp.harness.train.legged_latent_train import LeggedData
    dev = torch.device(device)
    rcfg, E, R, P, rres = load_rep(rep, dev)
    lsv, _ = legged_bundle_versions(rres["latent_space_version"], E.state_dict(), R.state_dict())
    data = LeggedData(Path(data_root or rcfg["data"]), [body], dev)
    M = int(data.S["asm_mask"][0].sum())
    rng = np.random.default_rng(seed)
    out = {}
    for split, n in (("fit", n_fit), ("cal", n_cal)):
        pool = data.train_idx if split == "fit" else data.test_idx
        idx = rng.choice(pool, size=min(n, len(pool)), replace=False)
        zs = []
        with torch.no_grad():
            for k in range(0, len(idx), 256):
                i = torch.from_numpy(idx[k:k + 256]).to(dev)
                mu, _ = E(data.ctx_batch(i), data.beh(i))
                zs.append(mu[:, :, :M].cpu().numpy())
        out[split] = np.concatenate(zs)
    info = dict(source="encoder_posterior_mean", representation=str(rep), data=str(data_root or rcfg["data"]),
                body=body, n_rows=int(data.n), seed=seed)
    return out["fit"], out["cal"], lsv, M, info


def load_packet_dir(d: Path, edit: str | None = None):
    eps = []
    for f in sorted(Path(d).glob("*.npz")):
        a = np.load(f, allow_pickle=False)
        if a["z"].ndim != 4:
            continue
        eps.append(dict(file=f.name, t=a["t"], edit=a["edit"], z=a["z"], fell=bool(a["fell"]), lsv=str(a["lsv"]),
                        success=bool(a["success"])))
    if edit is not None:
        eps = [e for e in eps if all(x == edit for x in e["edit"])]
    return eps


def cmd_fit_rep(a):
    from rrp.policies.packet_ood import PacketOODModel
    zf, zc, lsv, M, info = encoder_packets(Path(a.rep), a.body, a.n_fit, a.n_cal, a.seed, a.data)
    m = PacketOODModel.fit(zf, zc, latent_space_version=lsv, feature=a.feature, r=a.r, q_monitor=a.q_monitor,
                           q_enforce=a.q_enforce, body=a.body, fit_info=info)
    p = m.save(Path(a.out))
    print(json.dumps(dict(model=str(p), fingerprint=m.fingerprint(), thresholds=m.thresholds, n_assemblies=M), indent=1))


def cmd_fit_packets(a):
    from rrp.policies.packet_ood import PacketOODModel
    eps = load_packet_dir(Path(a.packets), edit="none")
    if not eps:
        raise SystemExit(f"no unedited packet files in {a.packets}")
    lsvs = {e["lsv"] for e in eps}
    if len(lsvs) != 1:
        raise SystemExit(f"packets from several bundles: {sorted(lsvs)}")
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(eps))                  # split by EPISODE (no leakage between fit and calibration)
    n_cal = max(1, len(eps) // 5)
    cal = np.concatenate([eps[i]["z"] for i in order[:n_cal]])
    fit = np.concatenate([eps[i]["z"] for i in order[n_cal:]])
    m = PacketOODModel.fit(fit, cal, latent_space_version=lsvs.pop(), feature=a.feature, r=a.r,
                           q_monitor=a.q_monitor, q_enforce=a.q_enforce, body=a.body,
                           fit_info=dict(source="recorded_generated_packets", packets=str(a.packets),
                                         episodes_fit=int(len(eps) - n_cal), episodes_cal=int(n_cal)))
    p = m.save(Path(a.out))
    print(json.dumps(dict(model=str(p), fingerprint=m.fingerprint(), thresholds=m.thresholds), indent=1))


def cmd_score(a):
    from rrp.policies.packet_ood import PacketOODModel
    m = PacketOODModel.load(Path(a.model))
    eps = load_packet_dir(Path(a.packets))
    rows = []
    for e in eps:
        if e["lsv"] != m.latent_space_version:
            raise SystemExit(f"{e['file']}: packets of {e['lsv']}, model fitted for {m.latent_space_version}")
        s = m.scores(e["z"])
        edited = np.array([x != "none" for x in e["edit"]])
        rows.append(dict(file=e["file"], fell=e["fell"], success=e["success"], edit=sorted(set(e["edit"].tolist())),
                         n=len(s), max_score=float(s.max()), max_score_edited=float(s[edited].max()) if edited.any() else None,
                         max_score_unedited=float(s[~edited].max()) if (~edited).any() else None,
                         over_monitor=int((s > m.thresholds["monitor"]).sum()),
                         over_enforce=int((s > m.thresholds["enforce"]).sum()),
                         first_over_enforce_t=float(e["t"][np.argmax(s > m.thresholds["enforce"])])
                         if (s > m.thresholds["enforce"]).any() else None))
    ed = [r for r in rows if any(x != "none" for x in r["edit"])]
    un = [r for r in rows if r["edit"] == ["none"]]
    summ = dict(model=str(a.model), fingerprint=m.fingerprint(), thresholds=m.thresholds, n_episodes=len(rows),
                auroc_fall=auroc([r["max_score"] for r in rows if r["fell"]], [r["max_score"] for r in rows if not r["fell"]]),
                auroc_edited_vs_unedited=auroc([r["max_score_edited"] for r in ed], [r["max_score"] for r in un]),
                flagged_edited=sum(r["over_enforce"] > 0 for r in ed), n_edited=len(ed),
                flagged_unedited=sum(r["over_enforce"] > 0 for r in un), n_unedited=len(un),
                flagged_falls=sum(r["over_enforce"] > 0 for r in rows if r["fell"]),
                n_falls=sum(r["fell"] for r in rows))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(summary=summ, episodes=rows), indent=1))
    print(json.dumps(summ, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit-rep", "fit-packets"):
        c = sp.add_parser(name)
        c.add_argument("--body", required=True)
        c.add_argument("--out", required=True, help="model path stem; writes <out>.json + <out>.npz")
        c.add_argument("--feature", default="packet", choices=["packet", "token"])
        c.add_argument("--r", type=int, default=32)
        c.add_argument("--q-monitor", type=float, default=0.99)
        c.add_argument("--q-enforce", type=float, default=0.999)
        c.add_argument("--seed", type=int, default=0)
        if name == "fit-rep":
            c.add_argument("--rep", required=True)
            c.add_argument("--data", default=None, help="dataset root (default: the representation's cfg)")
            c.add_argument("--n-fit", type=int, default=4000)
            c.add_argument("--n-cal", type=int, default=1000)
        else:
            c.add_argument("--packets", required=True)
    c = sp.add_parser("score")
    c.add_argument("--model", required=True)
    c.add_argument("--packets", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dict(**{"fit-rep": cmd_fit_rep, "fit-packets": cmd_fit_packets, "score": cmd_score})[a.cmd](a)


if __name__ == "__main__":
    main()
