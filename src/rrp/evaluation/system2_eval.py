"""System II (psi0 Qwen3-VL) -> public task binding for the legged latent policy: offline grounding + closed loop.

stage `ground`: for scene seeds x instruction families, render the declared public `front` camera at the initial
    state, ask system II which marker comes first. Records zero-shot answer scores (real / blank / shuffled image),
    4-token greedy generations, and pooled frozen features. A logistic readout ("probe", trained head) is fit on
    TRAIN seeds x TRAIN templates and evaluated on held-out seeds (held-out templates reported separately).
stage `closed_loop`: go2 dev seeds, color-family instructions with random order. Conditions differ ONLY in the
    public task binding given to system i:
        oracle   : binding from the instruction truth (upper reference)
        default  : the task file's fixed binding (orange first), instruction ignored (what system i had before)
        system2  : binding from system II's answer (zero-shot scoring, and the probe readout)
    Controller = learned legged latent policy (system i flow + system 0); privileged evaluator on the instruction
    truth.

usage: python -m rrp.evaluation.system2_eval ground --out artifacts/runs/legged_vlm_system2_v1
       python -m rrp.evaluation.system2_eval closed_loop --flow F/policy.pt --out artifacts/runs/legged_vlm_system2_v1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rrp.model.system2 import (COLORS, HELD_OUT_TEMPLATES, TEMPLATES, make_instruction, render_public,
                               scenario_with_truth)

TRAIN_SEEDS = range(0, 160)
TEST_SEEDS = range(5000, 5080)


def ground(out: Path, body="go2", batch=8):
    from rrp.model.system2 import System2
    from rrp.sim.legged import build_waypoint_contact
    s2 = System2()
    rows, feats = [], []
    t0 = time.time()
    items = []
    for split, seeds in (("train", TRAIN_SEEDS), ("test", TEST_SEEDS)):
        for sd in seeds:
            sc = build_waypoint_contact(body, sd)
            img = render_public(sc)
            for fam in TEMPLATES:
                text, truth, ti = make_instruction(sd, fam, sc.meta["waypoints"])
                items.append(dict(split=split, seed=sd, family=fam, template=ti,
                                  held_out_template=ti in HELD_OUT_TEMPLATES[fam], text=text, truth=list(truth),
                                  img=img))
    if items:
        import imageio
        out.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out / "example_front_camera.png", items[0]["img"])
    blank = np.full_like(items[0]["img"], 128)
    for k in range(0, len(items), batch):
        chunk = items[k:k + batch]
        texts = [c["text"] for c in chunk]
        real = s2.run([c["img"] for c in chunk], texts)
        bl = s2.run([blank for _ in chunk], texts)
        shuf = s2.run([items[(k + j + 37) % len(items)]["img"] for j in range(len(chunk))], texts)
        for j, c in enumerate(chunk):
            r = {kk: v for kk, v in c.items() if kk != "img"}
            r.update(score=float(real["score"][j]), score_blank=float(bl["score"][j]), score_shuffled=float(shuf["score"][j]),
                     gen=real["gen"][j], top1=real["top1"][j])
            rows.append(r)
            feats.append(real["feat"][j])
        print(f"{k + len(chunk)}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    F = np.stack(feats).astype(np.float32)
    np.save(out / "features.npy", F)
    with open(out / "ground_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = analyze(rows, F)
    res.update(provenance=s2.provenance(), wall_s=time.time() - t0, body_for_scene=body, camera="system2 (declared static virtual camera) 384x384",
               n=len(rows))
    (out / "ground_result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=1))
    return res


def fit_probe(F, y, steps=2000, wd=1e-2, seed=0):
    torch.manual_seed(seed)
    X = torch.from_numpy(F)
    mu, sd = X.mean(0), X.std(0) + 1e-4
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=1e-2)
    Y = torch.from_numpy(y.astype(np.float32))
    for _ in range(steps):
        lg = ((X - mu) / sd) @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lg, Y) + wd * (w ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda Z: ((((torch.from_numpy(Z) - mu) / sd) @ w + b) > 0).numpy(), dict(mu=mu, sd=sd, w=w.detach(), b=b.detach())


def analyze(rows, F):
    y = np.array([r["truth"][0] == "orange" for r in rows])
    res = {}
    for fam in TEMPLATES:
        idx = [i for i, r in enumerate(rows) if r["family"] == fam]
        tr = [i for i in idx if rows[i]["split"] == "train" and not rows[i]["held_out_template"]]
        te = [i for i in idx if rows[i]["split"] == "test"]
        te_ho = [i for i in te if rows[i]["held_out_template"]]
        te_in = [i for i in te if not rows[i]["held_out_template"]]
        acc = lambda key, ii: float(np.mean([(rows[i][key] > 0) == y[i] for i in ii])) if ii else None
        gen_acc = lambda ii: float(np.mean([(("orange" in rows[i]["gen"].lower()) and ("cyan" not in rows[i]["gen"].lower()))
                                            == y[i] for i in ii])) if ii else None
        pred, _ = fit_probe(F[tr], y[tr])
        pacc = lambda ii: float(np.mean(pred(F[ii]) == y[ii])) if ii else None
        res[fam] = dict(n_train=len(tr), n_test=len(te), base_rate_orange_first=float(y[te].mean()),
                        zero_shot_score_acc=dict(test=acc("score", te), blank_image=acc("score_blank", te),
                                                 shuffled_image=acc("score_shuffled", te)),
                        zero_shot_generation_acc=gen_acc(te),
                        probe_acc=dict(test_all=pacc(te), test_heldout_templates=pacc(te_ho),
                                       test_train_templates=pacc(te_in)),
                        example_generations=[rows[i]["gen"] for i in te[:6]])
    return res


def closed_loop(flow: Path, out: Path, seeds=range(10000, 10020), body="go2"):
    """Legged latent policy with three public task bindings (see module doc)."""
    from rrp.model.system2 import System2
    from rrp.evaluation.legged_latent_eval import LatentLeggedController, run_episode
    from rrp.learning.legged_latent_train import _dev
    dev = _dev()
    s2 = System2()
    # probe readout fit on the stage-`ground` training rows (color family)
    rows = [json.loads(l) for l in open(out / "ground_rows.jsonl")]
    F = np.load(out / "features.npy")
    tr = [i for i, r in enumerate(rows) if r["family"] == "color" and r["split"] == "train" and not r["held_out_template"]]
    y = np.array([r["truth"][0] == "orange" for r in rows])
    pred, _ = fit_probe(F[tr], y[tr])
    res_rows = []
    with open(out / "closed_loop.jsonl", "w") as f:
        for sd in seeds:
            from rrp.sim.legged import build_waypoint_contact
            sc0 = build_waypoint_contact(body, sd)
            text, truth, ti = make_instruction(sd, "color", sc0.meta["waypoints"])
            img = render_public(sc0)
            o = s2.run([img], [text])
            zs = ("orange", "cyan") if o["score"][0] > 0 else ("cyan", "orange")
            pr = ("orange", "cyan") if bool(pred(o["feat"])[0]) else ("cyan", "orange")
            for cond, order in (("oracle", tuple(truth)), ("default", ("orange", "cyan")),
                                ("system2_zero_shot", zs), ("system2_probe", pr)):
                sc = scenario_with_truth(body, sd, tuple(truth), order)
                ctl = LatentLeggedController(flow, dev, seed=sd)
                row, _ = run_episode(ctl, body, sd, 60.0, scenario=sc)
                r = dict(seed=sd, condition=cond, instruction=text, truth=list(truth), binding=list(order),
                         binding_correct=list(order) == list(truth), success=row["success"], fell=row["fell"],
                         sim_time=row["sim_time"], source=row["source"], events=row["events"])
                res_rows.append(r)
                f.write(json.dumps(r) + "\n"); f.flush()
                print(json.dumps({k: r[k] for k in ("seed", "condition", "binding_correct", "success", "fell")}), flush=True)
    summ = {}
    for cond in ("oracle", "default", "system2_zero_shot", "system2_probe"):
        rs = [r for r in res_rows if r["condition"] == cond]
        summ[cond] = dict(n=len(rs), binding_correct=sum(r["binding_correct"] for r in rs),
                          success=sum(r["success"] for r in rs), fell=sum(r["fell"] for r in rs))
    (out / "closed_loop_summary.json").write_text(json.dumps(dict(flow=str(flow), body=body, per_condition=summ),
                                                             indent=1))
    print(json.dumps(summ, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ground", "closed_loop"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--flow", default=None)
    ap.add_argument("--seeds", default="10000-10019")
    ap.add_argument("--small", action="store_true", help="smoke: 8 train + 8 test seeds")
    ap.add_argument("--body", default="go2")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "ground":
        if a.small:
            global TRAIN_SEEDS, TEST_SEEDS
            TRAIN_SEEDS, TEST_SEEDS = range(0, 8), range(5000, 5008)
        ground(out, body=a.body)
    else:
        lo, hi = a.seeds.split("-")
        closed_loop(Path(a.flow), out, range(int(lo), int(hi) + 1), body=a.body)


if __name__ == "__main__":
    main()
