"""Object-QA training/evaluation on top of a trained (VLM-backed) flow policy.

Trains readout + projector (policy frozen by default) with answer NLL through the frozen
Qwen2.5-0.5B decoder. Held-out evaluation uses DEPLOY-mode hidden states (the policy's own
sampled chunk; no demonstration) with real / blank / shuffled-object controls, per question type,
against the majority-answer baseline. QA pairs come from privileged labels (training/eval only).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from rrp.learning.checkpoint import load_checkpoint
from rrp.model.flow import PolicyConfig
from rrp.model.qa import ObjectQA, QTYPES, OBJECT_DEPENDENT, load_decoder, make_questions, policy_hidden


def _slots_by_episode(ds_dir: Path, eids):
    from rrp.data.collect import read_episode
    return {e: read_episode(ds_dir / "episodes" / f"{e}.private.pkl.gz")["slots"] for e in eids}


def _items(chunk, slots_of, rng, per_sample):
    items = []
    for bi, s in enumerate(chunk):
        for it in make_questions(s.labels, slots_of[s.meta["episode"]], rng, per_sample):
            it["bi"] = bi
            items.append(it)
    return items


def grad_report(qa: ObjectQA, policy) -> dict:
    g = lambda ps: float(sum(p.grad.norm() ** 2 for p in ps if p.grad is not None) ** 0.5)  # noqa: E731
    return dict(projector=g(qa.proj.parameters()), readout=g(qa.readout.parameters()),
                policy_blocks=g(policy.policy.blocks.parameters()) if hasattr(policy, "policy") else g(policy.blocks.parameters()),
                decoder_params_with_grad=sum(p.grad is not None for p in qa.lm.parameters()),
                decoder_requires_grad=sum(p.requires_grad for p in qa.lm.parameters()))


def run(cfg: dict) -> dict:
    from rrp.learning.behavior import device_setup
    from rrp.learning.data import collate_samples
    from rrp.learning.vlm_train import VLMFlowPolicy, load_split
    dev, ginfo = device_setup()
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    st = load_checkpoint(Path(cfg["policy_checkpoint"]), map_location=dev)
    pc = st["config"]
    policy = VLMFlowPolicy(PolicyConfig(**pc["policy"]), pc.get("resampler_cfg")).to(dev)
    policy.load_state_dict(st["model"])
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(bool(cfg.get("train_policy", False)))
    split_cfg = dict(pc, feature_cache=cfg.get("feature_cache", pc["feature_cache"]))
    store, tr, held = load_split(split_cfg)
    nv = store.n_visual
    slots_of = _slots_by_episode(Path(pc["dataset"]), {s.meta["episode"] for s in tr + held})
    t0 = time.time()
    lm, tok = load_decoder(dev)
    dec_load_s = time.time() - t0
    lm_bytes = sum(p.numel() * p.element_size() for p in lm.parameters())
    results = dict(policy_checkpoint=cfg["policy_checkpoint"], decoder=dict(repo_id="Qwen/Qwen2.5-0.5B-Instruct",
                   revision="7ae557604adf67be50417f59c2c2f167def9a775", frozen=True, bytes=lm_bytes, load_s=dec_load_s),
                   train_samples=len(tr), heldout_samples=len(held), gpu=ginfo, sources={})
    for source in cfg.get("sources", ["action", "action+slot"]):
        torch.manual_seed(cfg["seed"])
        rng = random.Random(cfg["seed"])
        gen = torch.Generator(device=dev).manual_seed(cfg["seed"])
        D = policy.cfg.D
        qa = ObjectQA(D, policy.cfg.heads, lm, tok, n_prefix=cfg.get("n_prefix", 4), source=source).to(dev)
        params = list(qa.parameters()) + ([p for p in policy.parameters()] if cfg.get("train_policy") else [])
        opt = torch.optim.AdamW(params, lr=cfg.get("lr", 3e-4), weight_decay=1e-4)
        log = open(out / f"train_log.{source}.jsonl", "w")
        step, gr = 0, None
        bs = cfg.get("batch_size", 64)
        for epoch in range(cfg["epochs"]):
            idx = list(range(len(tr)))
            rng.shuffle(idx)
            for i in range(0, len(idx) - bs + 1, bs):
                ch = [tr[j] for j in idx[i:i + bs]]
                batch, a, v, lab, eff = collate_samples(ch)
                feats = store.get([(s.meta["episode"], s.meta["t"]) for s in ch]).to(dev)
                batch, a, v = batch.to(dev), a.to(dev), v.to(dev)
                policy.attach(batch, feats, nv)
                hidden, cache = policy_hidden(policy.policy, batch, a, v, mode="teacher", gen=gen)
                slots = qa.readout(hidden, cache, batch)
                items = _items(ch, slots_of, rng, cfg.get("q_per_sample", 4))
                r = qa(slots, items)
                loss = r["nll"].mean()
                opt.zero_grad()
                loss.backward()
                if gr is None:
                    gr = grad_report(qa, policy)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                step += 1
                if step % 50 == 0:
                    log.write(json.dumps(dict(step=step, epoch=epoch, nll=float(loss), acc=float(r["correct"].float().mean()))) + "\n")
                    log.flush()
        ev = evaluate(qa, policy, held, store, slots_of, dev, nv, cfg)
        ev["train_steps"] = step
        ev["first_step_grads"] = gr
        results["sources"][source] = ev
        torch.save(dict(qa={k: v for k, v in qa.state_dict().items()}, source=source, config=cfg),
                   out / f"qa_head.{source.replace('+', '_')}.pt")
        print(source, json.dumps(ev["summary"]), flush=True)
    results["peak_gpu_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    (out / "result.json").write_text(json.dumps(results, indent=1, default=str))
    return results


@torch.no_grad()
def evaluate(qa, policy, held, store, slots_of, dev, nv, cfg) -> dict:
    from rrp.learning.data import collate_samples
    qa.eval()
    rng = random.Random(12345)
    gen = torch.Generator(device=dev).manual_seed(12345)
    recs = {m: {q: [[], []] for q in QTYPES} for m in ("real", "blank", "shuffled")}
    answers = {q: [] for q in QTYPES}
    bs = 64
    for i in range(0, min(len(held), cfg.get("eval_samples", 2000)), bs):
        ch = held[i:i + bs]
        batch, a, v, lab, eff = collate_samples(ch)
        feats = store.get([(s.meta["episode"], s.meta["t"]) for s in ch]).to(dev)
        batch = batch.to(dev)
        policy.attach(batch, feats, nv)
        hidden, cache = policy_hidden(policy.policy, batch, mode="deploy", gen=gen)
        slots = qa.readout(hidden, cache, batch)
        items = _items(ch, slots_of, rng, 6)
        for it in items:
            answers[it["qtype"]].append(it["ans"])
        for m in recs:
            r = qa(slots, items, substitute=None if m == "real" else m)
            for q, n, c in zip(r["qtypes"], r["nll"].tolist(), r["correct"].tolist()):
                recs[m][q][0].append(n)
                recs[m][q][1].append(c)
    qa.train()
    per = {}
    for q in QTYPES:
        maj = max(set(answers[q]), key=answers[q].count) if answers[q] else None
        per[q] = dict(n=len(answers[q]), majority_answer=maj,
                      majority_acc=answers[q].count(maj) / len(answers[q]) if answers[q] else None,
                      object_dependent=q in OBJECT_DEPENDENT,
                      **{f"{m}_acc": float(np.mean(recs[m][q][1])) if recs[m][q][1] else None for m in recs},
                      **{f"{m}_nll": float(np.mean(recs[m][q][0])) if recs[m][q][0] else None for m in recs})
    summ = {m: dict(acc=float(np.mean([c for q in QTYPES for c in recs[m][q][1]])),
                    nll=float(np.mean([n for q in QTYPES for n in recs[m][q][0]]))) for m in recs}
    return dict(per_type=per, summary=summ)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    run(json.loads(Path(a.config).read_text()))


if __name__ == "__main__":
    main()
