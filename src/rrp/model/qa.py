"""Object QA (R19): system-i slot readout -> learned projector -> prefix tokens -> SEPARATE frozen
text decoder -> answer NLL.

Slot readout: canonical scene-slot handles (clean context tokens) query the ACTION-EXPERT hidden
states of the flow policy (a velocity pass at flow time tau) -> r_s. Source "action" uses only
the attention output (object content must come through the action stream); "action+slot" also
adds a projection of the slot's clean context token.
Decoder: frozen Qwen/Qwen2.5-0.5B-Instruct (Apache-2.0), requires_grad False, eval mode.
Gradients flow THROUGH its operations into projector / readout / (optionally) policy, never into
its weights. The QA decoder is never called inside the sampler.
QA pairs are generated from PRIVILEGED labels and used for training/evaluation only.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rrp.model.attention import MHA

DECODER = dict(repo_id="Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775",
               license="apache-2.0")
COLORS = ["red", "blue", "yellow", "purple", "green"]
DISTRACTOR_COLORS = ("blue", "yellow", "purple")          # rrp.sim.scenario.build_pick_place defaults
QTYPES = {
    "color": ("What color is this object?", COLORS),
    "is_color": ("Is this object {c}?", ["yes", "no"]),
    "held": ("Is the gripper holding this object?", ["yes", "no"]),
    "visible": ("Is this object visible in the front camera?", ["yes", "no"]),
    "closer": ("Which object is closer to the gripper, the first or the second?", ["first", "second"]),
}
OBJECT_DEPENDENT = ("color", "is_color", "closer")


def slot_color(sim_body: str, cube_color: str = "red") -> str:
    if sim_body == "cube":
        return cube_color
    if sim_body == "target_zone":
        return "green"
    if sim_body.startswith("distractor"):
        return DISTRACTOR_COLORS[int(sim_body[len("distractor"):]) % len(DISTRACTOR_COLORS)]
    raise KeyError(sim_body)


def make_questions(labels: dict, slots: list[str], rng: random.Random, per_sample: int = 4) -> list[dict]:
    """labels: privileged per-step sample labels (rrp.learning.data.episode_samples): held [S] (first
    manipulator), visible [S], rel_tcp [S,3] (object minus TCP, world frame)."""
    S = len(slots)
    out = []
    for _ in range(per_sample):
        qt = rng.choice(list(QTYPES))
        a = rng.randrange(S)
        if qt == "color":
            out.append(dict(qtype=qt, a=a, b=-1, q=QTYPES[qt][0], ans=slot_color(slots[a])))
        elif qt == "is_color":
            true_c = slot_color(slots[a])
            c = true_c if rng.random() < 0.5 else rng.choice([x for x in COLORS if x != true_c])
            out.append(dict(qtype=qt, a=a, b=-1, q=QTYPES[qt][0].format(c=c), ans="yes" if c == true_c else "no"))
        elif qt == "held":
            out.append(dict(qtype=qt, a=a, b=-1, q=QTYPES[qt][0], ans="yes" if bool(labels["held"][a]) else "no"))
        elif qt == "visible":
            out.append(dict(qtype=qt, a=a, b=-1, q=QTYPES[qt][0], ans="yes" if bool(labels["visible"][a]) else "no"))
        else:
            if S < 2:
                continue
            b = rng.choice([j for j in range(S) if j != a])
            da = float(np.linalg.norm(labels["rel_tcp"][a]))
            db = float(np.linalg.norm(labels["rel_tcp"][b]))
            out.append(dict(qtype=qt, a=a, b=b, q=QTYPES[qt][0], ans="first" if da < db else "second"))
    return out


def load_decoder(device, dtype=torch.bfloat16, local_dir=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    src = local_dir or DECODER["repo_id"]
    kw = {} if local_dir else dict(revision=DECODER["revision"])
    tok = AutoTokenizer.from_pretrained(src, **kw)
    lm = AutoModelForCausalLM.from_pretrained(src, dtype=dtype, attn_implementation="sdpa", **kw).to(device)
    return lm, tok


class SlotReadout(nn.Module):
    def __init__(self, D: int, heads: int, source: str = "action", layer: int | None = None):
        super().__init__()
        assert source in ("action", "action+slot")
        self.source, self.layer = source, layer
        self.q = nn.Linear(D, D)
        self.att = MHA(D, heads)
        self.slot = nn.Linear(D, D) if source == "action+slot" else None
        self.norm = nn.LayerNorm(D)

    def forward(self, hidden: list, cache, batch) -> torch.Tensor:
        layer = len(hidden) // 2 if self.layer is None else self.layer
        h = hidden[layer]
        B, Hh, N, D = h.shape
        keys = h.reshape(B, Hh * N, D)
        km = cache.node_mask[:, None, :].expand(B, Hh, N).reshape(B, Hh * N)
        so = batch.bank_offset["scene"]
        S = batch.bank_tokens["scene"].shape[1]
        sc = cache.ctx[:, so:so + S]
        r = self.att(self.q(sc), kv=keys, key_mask=km)
        if self.slot is not None:
            r = r + self.slot(sc)
        return self.norm(r)                                       # [B, S, D]


class ObjectQA(nn.Module):
    """Trainable: readout + projector. Frozen: decoder (kept outside the module's parameters)."""

    def __init__(self, D: int, heads: int, lm, tok, n_prefix: int = 4, source: str = "action"):
        super().__init__()
        self.readout = SlotReadout(D, heads, source)
        self.n_prefix = n_prefix
        E = lm.get_input_embeddings().weight.shape[1]
        self.proj = nn.Sequential(nn.Linear(D, 2 * E), nn.GELU(), nn.Linear(2 * E, n_prefix * E))
        self.role = nn.Parameter(torch.zeros(2, n_prefix, E))     # first/second object prefix marker
        self._lm = [lm]            # list => not registered as a submodule (never in state_dict / optimizer)
        self.tok = tok
        lm.eval()
        for p in lm.parameters():
            p.requires_grad_(False)
        emb_scale = lm.get_input_embeddings().weight.float().norm(dim=-1).mean()
        self.register_buffer("emb_scale", emb_scale.detach().clone())
        self.cand_ids = {}
        for qt, (_, cands) in QTYPES.items():
            ids = []
            for c in cands:
                t = tok(" " + c, add_special_tokens=False)["input_ids"]
                if len(t) != 1:
                    raise ValueError(f"answer {c!r} is not a single token: {t}")
                ids.append(t[0])
            self.cand_ids[qt] = ids

    @property
    def lm(self):
        return self._lm[0]

    def prefix(self, slot_vec: torch.Tensor) -> torch.Tensor:
        E = self.role.shape[-1]
        p = self.proj(slot_vec).view(*slot_vec.shape[:-1], self.n_prefix, E)
        p = F.normalize(p, dim=-1) * self.emb_scale                # match decoder embedding scale
        return p

    def forward(self, slots: torch.Tensor, items: list[dict], substitute: str | None = None, gen=None) -> dict:
        """slots [B, S, D] readout; items: dicts with bi (batch idx), a, b, q, ans, qtype.
        substitute: None | 'blank' (zero slot vectors) | 'shuffled' (a different object's vector)."""
        dev = slots.device
        lm, tok = self.lm, self.tok
        emb = lm.get_input_embeddings()
        dt = emb.weight.dtype

        def vec(bi, s):
            return slots[bi, s]
        if substitute == "blank":
            vec = lambda bi, s: torch.zeros_like(slots[0, 0])  # noqa: E731
        elif substitute == "shuffled":
            pool = [(i["bi"], s) for i in items for s in (i["a"], i["b"]) if s >= 0]
            rng = random.Random(0)

            def vec(bi, s):  # noqa: F811 - another object from another scene/slot
                cand = [p for p in pool if p != (bi, s) and p[0] != bi]
                cand = cand or [p for p in pool if p != (bi, s)]
                return slots[rng.choice(cand)]
        seqs, ans_pos, ans_ids = [], [], []
        for it in items:
            parts = [self.prefix(vec(it["bi"], it["a"])) + self.role[0]]
            if it["b"] >= 0:
                parts.append(self.prefix(vec(it["bi"], it["b"])) + self.role[1])
            q_ids = tok(f" Question: {it['q']} Answer:", add_special_tokens=False)["input_ids"]
            a_id = tok(" " + it["ans"], add_special_tokens=False)["input_ids"]
            assert len(a_id) == 1
            ids = torch.tensor(q_ids + a_id, device=dev)
            e = torch.cat([p.to(dt) for p in parts] + [emb(ids)], 0)
            seqs.append(e)
            ans_pos.append(e.shape[0] - 2)              # logits at position predicting the answer token
            ans_ids.append(a_id[0])
        L = max(s.shape[0] for s in seqs)
        E = seqs[0].shape[1]
        x = torch.zeros(len(seqs), L, E, dtype=dt, device=dev)
        m = torch.zeros(len(seqs), L, dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            x[i, :s.shape[0]] = s
            m[i, :s.shape[0]] = 1
        out = lm(inputs_embeds=x, attention_mask=m, use_cache=False)
        idx = torch.tensor(ans_pos, device=dev)
        logits = out.logits[torch.arange(len(seqs), device=dev), idx].float()      # [Q, V]
        tgt = torch.tensor(ans_ids, device=dev)
        nll = F.cross_entropy(logits, tgt, reduction="none")
        correct = []
        for i, it in enumerate(items):
            c = self.cand_ids[it["qtype"]]
            correct.append(c[int(logits[i, c].argmax())] == ans_ids[i])
        return dict(nll=nll, correct=torch.tensor(correct), qtypes=[i["qtype"] for i in items])


# ------------------------------------------------------------------ system-i hidden states for QA
def policy_hidden(policy, batch, a=None, v=None, mode: str = "teacher", nfe: int = 8, gen=None):
    """Action-expert hidden states. teacher: z_tau from the demonstration at random tau (training);
    deploy: the policy's own sampled chunk at tau = 1 - 1/nfe (no demonstration, no future truth)."""
    from rrp.model.flow import interpolate_target
    cache = policy.prepare(batch)
    B, N = batch.node_mask.shape
    H = policy.cfg.horizon
    if mode == "teacher":
        eps = torch.randn(a.shape, generator=gen, device=a.device, dtype=a.dtype)
        tau = torch.rand(B, generator=gen, device=a.device, dtype=a.dtype)
        z, _ = interpolate_target(eps, a, tau)
        z = z * (v & batch.node_mask[:, None, :]).to(a.dtype)[..., None]
    else:
        with torch.no_grad():
            z = policy.sample(cache, H, nfe=nfe, generator=gen)
        tau = torch.full((B,), 1 - 1.0 / nfe, device=z.device, dtype=z.dtype)
    _, hidden = policy.velocity(z, tau, cache, return_hidden=True)
    return hidden, cache
