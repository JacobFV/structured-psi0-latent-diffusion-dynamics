"""Ψ₀ + structure networks (W10; moved from psi1z `structured.py` + `system_i.py`, D-140). Math unchanged (golden test).

Structured latent packet for Ψ₀ on the G1 (+ Dex3 hands): z[K=5 knots, M=6 assemblies, DZ=64]; assemblies =
`rrp.bodies.g1_simple.ASSEMBLIES` (left_hand, right_hand, left_arm, right_arm, torso, base); knots at chunk steps
KNOT_STEPS (20 ms ticks; packet horizon 0.6 s = Ψ₀'s 30-step chunk). K * M = 30 tokens = the token count of Ψ₀'s action
chunk, so system i (built from the upstream Ψ₀ action transformer) uses exactly the token budget Ψ₀ uses for actions.

    E (training only): demonstrated chunk a[t:t+30] (normalized) + morphology tokens + current state -> q(z)
    R (system 0):      z + morphology tokens + CURRENT state (any tick j inside the packet) + phase j -> native
                       36-dim commands for the packet's 30 ticks. R never sees the image, the instruction, object
                       information or system-i state: task meaning arrives only through z.
    P (probe):         z + opaque assembly/knot codes -> semantics (hand-target distance, contact, lift, target position
                       in the robot frame, active hand (binding), base displacement, base command; optional grasp region).
                       metadata_only=True gives the no-z control probe.

System i for the matched fine-tunes: Ψ₀'s own action transformer (upstream `ActionTransformerModel`, ~500M), initialized
like Ψ₀'s SIMPLE fine-tune (only `transformer_blocks.*` from the post-trained action header).
  DirectHead      ("Ψ₀ direct"): flow over the 30x36 action chunk (the upstream objective).
  StructuredHead  ("Ψ₀ + structure"): flow over the packet z (K*M = 30 tokens x 64), with context tokens appended to the
                  VLM tokens: 36 morphology-relation tokens (DimEncoder over the command dims and the CURRENT state) and
                  one object-entity token (VLM states pooled over the target-object name; a learned null token when the
                  instruction names none). Actions come from system 0 (Realizer) applied to the sampled z.
Both see the same information (image + instruction through the same frozen VLM features, state).
Flow convention (upstream FinetuneTrainer.forward_and_loss): sigma ~ U(0,1), x = (1-sigma) a + sigma eps, v = eps - a,
timestep = sigma * 1000; sampling: Euler from sigma=1 to 0 in `nfe` steps.
Naming: Ψ₀/psi0 = the upstream model; system i / system 0 = rrp's packet generator / realizer.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rrp.bodies import g1_simple as G
from rrp.policies.nets.attention import MHA, StructuralBias
from rrp.policies.nets.flow import MLP, sinusoidal
from rrp.policies.nets.legged_latent import gnll

K = 5
KNOT_STEPS = (5, 11, 17, 23, 29)
M = len(G.ASSEMBLIES)
DZ = 64
TP = 30                     # Psi0 chunk length
DA = G.ACTION_DIM           # 36
N_REL = len(G.RELATIONS)

# which assemblies' knots each assembly's command dims may read in system 0 (own + kinematic neighbours)
_A = G.ASM_INDEX
READS = {
    "left_hand": ("left_hand", "left_arm"),
    "right_hand": ("right_hand", "right_arm"),
    "left_arm": ("left_arm", "left_hand", "torso"),
    "right_arm": ("right_arm", "right_hand", "torso"),
    "torso": ("torso", "base", "left_arm", "right_arm"),
    "base": ("base", "torso"),
}


def read_mask() -> torch.Tensor:
    """[DA, K*M] bool: command dim i may attend to knot token (k, m)."""
    allow = np.zeros((DA, M), dtype=bool)
    for i in range(DA):
        own = G.ASSEMBLIES[G.DIM_ASM[i]]
        for a in READS[own]:
            allow[i, _A[a]] = True
    return torch.from_numpy(np.tile(allow, (1, K)))          # token order k-major: (k0,m0..m5),(k1,...)


class Morph(nn.Module):
    """Constant morphology buffers (node features, relations, assembly tokens)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("node_static", torch.from_numpy(G.node_static()))                   # [DA, F]
        self.register_buffer("rel", torch.from_numpy(G.relation_matrix()).permute(1, 2, 0))       # [DA, DA, R]
        self.register_buffer("asm_static", torch.from_numpy(G.asm_static()))                      # [M, M+3]
        self.register_buffer("dim_asm", torch.from_numpy(G.DIM_ASM))
        self.register_buffer("read_mask", read_mask())
        sm = torch.zeros(DA, dtype=torch.bool)
        sm[:G.STATE_DIM] = True
        self.register_buffer("state_valid", sm)                                                   # dims with a state


class DimEncoder(nn.Module):
    """Morphology-relation tokens: one token per command dim (static morphology + current state value + optional
    per-dim extra features), mixed by self-attention with a learned per-relation structural bias
    (self/parent/child/same-assembly/mirror). With the bias weights at 0 it is plain attention."""

    def __init__(self, D, heads=4, layers=2, extra=0):
        super().__init__()
        self.inp = MLP(G.NODE_STATIC_DIM + 2 + extra, D)
        self.layers = nn.ModuleList([nn.ModuleDict(dict(n=nn.LayerNorm(D), a=MHA(D, heads), sb=StructuralBias(heads, N_REL),
                                                        n2=nn.LayerNorm(D), m=MLP(D, D, 4 * D)))
                                     for _ in range(layers)])

    def forward(self, morph: Morph, state, extra=None):
        B = state.shape[0]
        sv = morph.state_valid.to(state.dtype)
        x = torch.cat([morph.node_static[None].expand(B, -1, -1), (state[:, :DA] * sv)[..., None],
                       sv[None, :, None].expand(B, -1, -1)] + ([extra] if extra is not None else []), -1)
        t = self.inp(x)
        rel = morph.rel[None].expand(B, -1, -1, -1)
        for L in self.layers:
            t = t + L["a"](L["n"](t), bias=L["sb"](rel))
            t = t + L["m"](L["n2"](t))
        return t                                                                                   # [B, DA, D]


def block(D, heads):
    return nn.ModuleDict(dict(n1=nn.LayerNorm(D), x=MHA(D, heads), n2=nn.LayerNorm(D), s=MHA(D, heads),
                              n3=nn.LayerNorm(D), m=MLP(D, D, 4 * D)))


def run_block(L, q, kv, bias=None, q_bias=None):
    q = q + L["x"](L["n1"](q), kv=kv, bias=bias)
    q = q + L["s"](L["n2"](q), bias=q_bias)
    return q + L["m"](L["n3"](q))


class KnotQueries(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.asm = MLP(M + 3, D)
        self.knot = nn.Embedding(K, D)

    def forward(self, morph: Morph, B):
        a = self.asm(morph.asm_static)                                                             # [M, D]
        q = a[None] + self.knot.weight[:, None]                                                     # [K, M, D]
        return q.reshape(1, K * M, -1).expand(B, -1, -1)


class PacketEncoder(nn.Module):
    """E: demonstrated normalized chunk [B, TP, DA] + state -> Gaussian posterior over z [B, K, M, DZ]."""

    def __init__(self, D=256, heads=4, layers=3):
        super().__init__()
        self.dims = DimEncoder(D, heads, 2, extra=TP)
        self.q = KnotQueries(D)
        self.blocks = nn.ModuleList([block(D, heads) for _ in range(layers)])
        self.out = nn.Linear(D, 2 * DZ)

    def forward(self, morph, state, actions):
        B = state.shape[0]
        t = self.dims(morph, state, extra=actions.transpose(1, 2))                                   # [B, DA, D]
        q = self.q(morph, B)
        for L in self.blocks:
            q = run_block(L, q, t)
        mu, lv = self.out(q).reshape(B, K, M, 2, DZ).unbind(3)
        return mu, lv.clamp(-8, 4)


class Realizer(nn.Module):
    """R (system 0): z + morphology + CURRENT state at tick j (+ phase j) -> 30-tick command rows for the packet.
    Cross-attention: each command dim reads packet knots DIRECTLY only from its own assembly and its kinematic neighbours
    (READS). Self-attention between command dims then mixes information across assemblies, so the locality is a routing
    prior, not an information barrier (tests/unit/test_psi0.py measures both)."""

    def __init__(self, D=256, heads=4, layers=3):
        super().__init__()
        self.dims = DimEncoder(D, heads, 2, extra=2)
        self.z_in = nn.Linear(DZ, D)
        self.asm = MLP(M + 3, D)
        self.kt = MLP(D, D)
        self.blocks = nn.ModuleList([block(D, heads) for _ in range(layers)])
        self.out = nn.Linear(D, TP)
        self.D = D

    def forward(self, morph, z, state, phase):
        """z [B,K,M,DZ]; state [B,>=32] normalized CURRENT state; phase [B] = tick index j / TP."""
        B = z.shape[0]
        ph = torch.stack([torch.sin(math.pi * phase), torch.cos(math.pi * phase)], -1)            # [B, 2]
        x = self.dims(morph, state, extra=ph[:, None].expand(-1, DA, -1))
        kt = torch.as_tensor(KNOT_STEPS, dtype=z.dtype, device=z.device) / TP
        rel = kt[None] - phase[:, None]                                                            # [B, K]
        tok = self.z_in(z) + self.kt(sinusoidal(rel, self.D))[:, :, None] + self.asm(morph.asm_static)[None, None]
        tok = tok.reshape(B, K * M, -1)
        bias = torch.zeros(1, 1, DA, K * M, device=z.device, dtype=tok.dtype).masked_fill(~morph.read_mask[None, None],
                                                                                           float("-inf"))
        for L in self.blocks:
            x = run_block(L, x, tok, bias=bias)
        return self.out(x).transpose(1, 2)                                                         # [B, TP, DA]


# ---------------------------------------------------------------- probe
HANDS = ("left_hand", "right_hand")
PROBE_QUERIES = ("hand_dist", "contact", "lift", "target_pos", "active_hand", "base_disp", "base_cmd")
N_FACES = 6
CMD_DIMS = (32, 34)          # vx, yaw-rate command (normalized action units) — packet LABELS, never inputs


def add_cmd_labels(b):
    """base_cmd[k] = demonstrated (vx, vyaw) command at knot k, normalized units (label only)."""
    if "labels" in b:
        b["labels"]["base_cmd"] = b["actions"][:, list(KNOT_STEPS)][..., list(CMD_DIMS)].float()
    return b


class PacketProbe(nn.Module):
    """P: reads z with opaque random assembly/knot codes only (no scene features, no labels).

    Outputs per knot k: hand_dist[k, h] (Gaussian, m, hand-to-target distance), contact[k, h] (logit),
    lift[k] (logit, target above its initial height by >= 3 cm), target_pos[k] (Gaussian, 3, target in the
    robot base frame); per packet: active_hand (2-way logits: which hand grasps; binding), base_disp (Gaussian, 3:
    dx, dy, dyaw over the packet in the start base frame)."""

    def __init__(self, D=192, heads=4, metadata_only=False, seed=1234, grasp=False):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("asm_code", F.normalize(torch.randn(M, 16, generator=g), dim=-1))
        self.register_buffer("knot_code", F.normalize(torch.randn(K, 16, generator=g), dim=-1))
        self.metadata_only = metadata_only
        self.z_in = nn.Linear(DZ, D)
        self.code = nn.Linear(16, D)
        self.kcode = nn.Linear(16, D)
        self.qtype = nn.Embedding(len(PROBE_QUERIES), D)
        self.const = nn.Parameter(torch.zeros(1, 1, D))
        self.a1, self.a2 = MHA(D, heads), MHA(D, heads)
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.mlp = MLP(D, D, 2 * D)
        self.heads = nn.ModuleDict(dict(hand_dist=nn.Linear(D, 2), contact=nn.Linear(D, 1), lift=nn.Linear(D, 1),
                                        target_pos=nn.Linear(D, 6), active_hand=nn.Linear(D, 2),
                                        base_disp=nn.Linear(D, 6), base_cmd=nn.Linear(D, 4)))
        # grasp-region affordance (roadmap #24, rrp D-126; default OFF): per hand, the first-contact point on the target
        # in the target's object frame (Gaussian, 3-d) + contact-face class (6: +-x, +-y, +-z). Built last and only when
        # enabled, so default-off models keep exactly the same parameters and initialisation.
        self.grasp = grasp
        if grasp:
            self.grasp_q = nn.Embedding(1, D)
            self.grasp_head = nn.Linear(D, 6 + N_FACES)

    def forward(self, z):
        B = z.shape[0]
        tpos = self.kcode(self.knot_code)[:, None] + self.code(self.asm_code)[None]               # [K, M, D]
        t = (self.const.expand(B, K * M, -1) + tpos.reshape(1, K * M, -1)) if self.metadata_only else \
            (self.z_in(z) + tpos[None]).reshape(B, K * M, -1)

        def read(q, key_mask=None):
            r = q + self.a1(self.n1(q), kv=t, key_mask=key_mask)
            r = r + self.a2(self.n2(r), kv=t, key_mask=key_mask)
            return r + self.mlp(r)
        hi = torch.tensor([G.ASM_INDEX[h] for h in HANDS], device=z.device)
        qh = tpos[:, hi]                                                                           # [K, 2, D]
        qd = read((self.qtype.weight[0] + qh).reshape(1, K * 2, -1).expand(B, -1, -1))
        qc = read((self.qtype.weight[1] + qh).reshape(1, K * 2, -1).expand(B, -1, -1))
        kk = self.kcode(self.knot_code)                                                            # [K, D]
        ql = read((self.qtype.weight[2] + kk)[None].expand(B, -1, -1))
        qt = read((self.qtype.weight[3] + kk)[None].expand(B, -1, -1))
        qa = read(self.qtype.weight[4][None, None].expand(B, 1, -1))
        qb = read(self.qtype.weight[5][None, None].expand(B, 1, -1))
        # base command is read ONLY from the tokens that system 0's base dims read (READS["base"]): the packet must
        # carry locomotion intent where the realizer can use it
        bm = torch.zeros(K, M, dtype=torch.bool, device=z.device)
        for an in READS["base"]:
            bm[:, G.ASM_INDEX[an]] = True
        qm = read((self.qtype.weight[6] + kk + self.code(self.asm_code[G.ASM_INDEX["base"]]))[None].expand(B, -1, -1),
                  key_mask=bm.reshape(1, K * M).expand(B, -1))
        return dict(hand_dist=self.heads["hand_dist"](qd).reshape(B, K, 2, 2),
                    contact=self.heads["contact"](qc).reshape(B, K, 2),
                    lift=self.heads["lift"](ql)[..., 0],
                    target_pos=self.heads["target_pos"](qt),
                    active_hand=self.heads["active_hand"](qa)[:, 0],
                    base_disp=self.heads["base_disp"](qb)[:, 0],
                    base_cmd=self.heads["base_cmd"](qm),
                    **({"grasp": self._grasp(read, qh, B)} if self.grasp else {}))

    def _grasp(self, read, qh, B):
        q = read((self.grasp_q.weight[0] + qh[-1]).reshape(1, 2, -1).expand(B, -1, -1))     # last knot, per hand
        return self.grasp_head(q)                                                            # [B, 2, 6 + N_FACES]



def probe_loss(out, lab, lv_min=-4.0, w_grasp=0.0):
    """lab: hand_dist [B,K,2], contact [B,K,2] {0,1}, lift [B,K], target_pos [B,K,3], active_hand [B] (long, -1 = none),
    base_disp [B,3]; *_valid masks optional. lv_min=-4 is the bounded-NLL setting (rrp D-085)."""
    L = {}
    L["hand_dist"] = gnll(out["hand_dist"], lab["hand_dist"][..., None], lv_min=lv_min)
    L["contact"] = F.binary_cross_entropy_with_logits(out["contact"], lab["contact"].float())
    L["lift"] = F.binary_cross_entropy_with_logits(out["lift"], lab["lift"].float())
    L["target_pos"] = gnll(out["target_pos"], lab["target_pos"], lv_min=lv_min)
    ah = lab["active_hand"]
    L["active_hand"] = F.cross_entropy(out["active_hand"], ah.clamp(min=0), reduction="none")[ah >= 0].mean() \
        if (ah >= 0).any() else out["active_hand"].sum() * 0
    L["base_disp"] = gnll(out["base_disp"], lab["base_disp"], lv_min=lv_min)
    if "base_cmd" in lab:
        L["base_cmd"] = gnll(out["base_cmd"], lab["base_cmd"], lv_min=lv_min)
    if w_grasp > 0 and "grasp" in out and "grasp_pt" in lab:
        m = lab["grasp_valid"].bool()                                                    # [B, 2]
        if m.any():
            g = out["grasp"][m]
            L["grasp_pt"] = w_grasp * gnll(g[:, :6], lab["grasp_pt"][m], lv_min=lv_min)
            L["grasp_face"] = w_grasp * F.cross_entropy(g[:, 6:], lab["grasp_face"][m])
    return sum(L.values()), {f"probe_{k}": float(v.detach()) for k, v in L.items()}


@torch.no_grad()
def probe_metrics(out, lab):
    """(sum, count) pairs."""
    res = {}
    d = (out["hand_dist"][..., 0] - lab["hand_dist"]).abs()
    res["hand_dist_mae"] = (float(d.sum()), d.numel())
    c = (out["contact"] > 0) == lab["contact"].bool()
    res["contact_acc"] = (int(c.sum()), c.numel())
    pos = lab["contact"].bool()
    res["contact_pos_recall"] = (int((c & pos).sum()), int(pos.sum()))
    li = (out["lift"] > 0) == lab["lift"].bool()
    res["lift_acc"] = (int(li.sum()), li.numel())
    te = (out["target_pos"][..., :3] - lab["target_pos"]).norm(dim=-1)
    res["target_pos_err"] = (float(te.sum()), te.numel())
    ah = lab["active_hand"]
    ok = (out["active_hand"].argmax(-1) == ah) & (ah >= 0)
    res["active_hand_acc"] = (int(ok.sum()), int((ah >= 0).sum()))
    if "base_cmd" in lab:
        ce = (out["base_cmd"][..., :2] - lab["base_cmd"]).abs()
        res["base_cmd_vx_mae_norm"] = (float(ce[..., 0].sum()), ce[..., 0].numel())
    if "grasp" in out and "grasp_pt" in lab and lab["grasp_valid"].any():
        m = lab["grasp_valid"].bool()
        e = (out["grasp"][m][:, :3] - lab["grasp_pt"][m]).norm(dim=-1)
        res["grasp_pt_err"] = (float(e.sum()), e.numel())
        f = out["grasp"][m][:, 6:].argmax(-1) == lab["grasp_face"][m]
        res["grasp_face_acc"] = (int(f.sum()), f.numel())
    be = (out["base_disp"][:, :2] - lab["base_disp"][:, :2]).norm(dim=-1)
    res["base_disp_xy_err"] = (float(be.sum()), be.numel())
    return res


def kl_std_normal(mu, lv):
    return 0.5 * (mu.pow(2) + lv.exp() - 1 - lv).sum(-1).mean()


class StageA(nn.Module):
    """E + R + P trained jointly (the packet's semantics are supervised ON z; R realizes the SAME z)."""

    def __init__(self, D=256, probe_D=192, metadata_only_probe=False, grasp=False):
        super().__init__()
        self.morph = Morph()
        self.E = PacketEncoder(D)
        self.R = Realizer(D)
        self.P = PacketProbe(probe_D, metadata_only=metadata_only_probe, grasp=grasp)

    rec_dim_w = None      # optional [DA] reconstruction weights (set by the trainer)

    def loss(self, b, w_kl=1e-3, w_sem=1.0, lv_min=-4.0, z_noise=0.0, w_grasp=0.0):
        """b: state0 [B,36] (packet start), actions [B,TP,DA] normalized, amask [B,TP,DA],
        j [B] realization tick, state_j [B,36] (state at tick j), labels (see probe_loss)."""
        add_cmd_labels(b)
        mu, lv = self.E(self.morph, b["state0"], b["actions"])
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        zr = z + z_noise * torch.randn_like(z) if z_noise > 0 else z
        pred = self.R(self.morph, zr, b["state_j"], b["j"].to(z.dtype) / TP)
        tmask = (torch.arange(TP, device=z.device)[None] >= b["j"][:, None]).to(z.dtype)[..., None]  # rows >= j
        m = b["amask"] * tmask
        if self.rec_dim_w is not None:
            m = m * self.rec_dim_w.to(m.device)
        rec = (((pred - b["actions"]) ** 2) * m).sum() / m.sum().clamp(min=1)
        kl = kl_std_normal(mu, lv)
        pl, plog = probe_loss(self.P(z), b["labels"], lv_min=lv_min, w_grasp=w_grasp) if w_sem > 0 else (z.sum() * 0, {})
        loss = rec + w_kl * kl + w_sem * pl
        return loss, dict(rec=float(rec.detach()), kl=float(kl.detach()), **plog), dict(rec=rec, kl=kl, sem=pl)


def load_stage_a(path, map_location="cpu"):
    """Load a stage-A checkpoint, tolerating v1 checkpoints (6 probe queries, no base_cmd head): the extra query row
    and head keep their fresh init (only used when base_cmd labels are present)."""
    sd = torch.load(path, weights_only=False, map_location=map_location)["model"]
    A = StageA(grasp=any(k.startswith("P.grasp_") for k in sd))
    own = A.state_dict()
    for k, v in list(sd.items()):
        if k in own and own[k].shape != v.shape and k.endswith("qtype.weight"):
            w = own[k].clone(); w[: v.shape[0]] = v; sd[k] = w
    A.load_state_dict(sd, strict=False)
    return A


def load_tolerant(module, sd):
    """load_state_dict for modules containing a v1 stage A (pads probe qtype, ignores the missing base_cmd head)."""
    own = module.state_dict()
    for k, v in list(sd.items()):
        if k in own and own[k].shape != v.shape and k.endswith("qtype.weight"):
            w = own[k].clone(); w[: v.shape[0]] = v; sd[k] = w
    missing, unexpected = module.load_state_dict(sd, strict=False)
    bad = [m for m in missing if "base_cmd" not in m] + list(unexpected)
    if bad:
        raise RuntimeError(f"unexpected key mismatch: {bad[:5]}")
    return module

# ---------------------------------------------------------------- system i heads (Ψ₀ action transformer)
VIEW_DIM = 2048


def build_header(model_cfg, action_dim: int, horizon: int):
    from psi.models.psi0 import ActionTransformerModel
    return ActionTransformerModel(
        resnet_store_path=model_cfg.resnet_store_path, odim=model_cfg.odim, action_dim=action_dim,
        action_pred_horizon=horizon, view_feature_dim=model_cfg.view_feature_dim, use_film=model_cfg.use_film,
        combined_temb=model_cfg.combined_temb, action_hidden_dim=model_cfg.hidden_dim,
        action_num_blocks=model_cfg.num_blocks, final_layer_norm=model_cfg.final_layer_norm, qk_norm=model_cfg.qk_norm,
        pooled_projection_dim=model_cfg.pooled_projection_dim, layerwise_vlm_fusion=False,
        state_drop_prob=model_cfg.state_drop_prob, state_as_action_token=getattr(model_cfg, "state_as_action_token", False),
        state_null_token=getattr(model_cfg, "state_null_token", False), dropout=model_cfg.dropout,
        state_feature_dropout=getattr(model_cfg, "state_feature_dropout", 0.2))


def load_pretrained_blocks(header, path):
    from safetensors.torch import load_file
    sd = load_file(f"{path}/action_header.safetensors")
    blocks = {k: v for k, v in sd.items() if k.startswith("transformer_blocks")}
    missing, unexpected = header.load_state_dict(blocks, strict=False)
    n_loaded = len(blocks) - len(unexpected)
    return dict(loaded=n_loaded, total_ckpt=len(sd), missing_non_block=len([m for m in missing if not m.startswith("transformer_blocks")]))


def header_forward(header, x, sigma, views, mask, state):
    return header(hidden_states=None, timestep=sigma * 1000.0,
                  joint_attention_kwargs=dict(action_hidden_embeds=x, views=views[:, None], obs=state[:, None], traj2ds=None),
                  vlm_attn_mask=mask, return_dict=True).action


def flow_loss(pred_v, a, eps, mask=None):
    err = (pred_v - (eps - a)) ** 2
    if mask is None:
        return err.mean()
    return (err * mask).sum() / mask.sum().clamp(min=1)


class DirectHead(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.header = build_header(model_cfg, 36, 30)

    def loss(self, b):
        a = b["actions"]
        eps = torch.randn_like(a)
        sigma = torch.rand(a.shape[0], device=a.device)
        x = (1 - sigma[:, None, None]) * a + sigma[:, None, None] * eps
        v = header_forward(self.header, x, sigma, b["hidden"], b["mask"], b["state0"])
        fl = flow_loss(v.float(), a, eps, b["amask"])
        return fl, dict(flow=float(fl.detach())), dict(flow=fl)

    @torch.no_grad()
    def sample(self, b, nfe=10, generator=None):
        B = b["state0"].shape[0]
        x = torch.randn(B, 30, 36, device=b["state0"].device, generator=generator)
        for i in range(nfe):
            s = torch.full((B,), 1.0 - i / nfe, device=x.device)
            v = header_forward(self.header, x, s, b["hidden"], b["mask"], b["state0"]).float()
            x = x - (1.0 / nfe) * v
        return x


class ContextTokens(nn.Module):
    """Morphology-relation tokens (36) + one object-entity token, in the VLM token space (2048)."""

    def __init__(self, D=512):
        super().__init__()
        self.morph = Morph()
        self.dims = DimEncoder(D, heads=8, layers=2)
        self.dim_out = nn.Linear(D, VIEW_DIM)
        self.ent = nn.Sequential(nn.LayerNorm(VIEW_DIM), nn.Linear(VIEW_DIM, VIEW_DIM))
        self.ent_null = nn.Parameter(torch.randn(VIEW_DIM) * 0.02)
        self.type_emb = nn.Parameter(torch.zeros(2, VIEW_DIM))

    def forward(self, b):
        hid = b["hidden"].float()
        m = self.dim_out(self.dims(self.morph, b["state0"])) + self.type_emb[0]              # [B, 36, 2048]
        em = b["ent"].float()
        has = em.sum(1) > 0
        src = b["ent_hidden"].float() if "ent_hidden" in b else hid        # entity edit: pool from another prompt
        pooled = (src * em[..., None]).sum(1) / em.sum(1, keepdim=True).clamp(min=1)
        e = torch.where(has[:, None], self.ent(pooled), self.ent_null[None].expand_as(pooled)) + self.type_emb[1]
        tok = torch.cat([m, e[:, None]], 1)
        views = torch.cat([b["hidden"], tok.to(b["hidden"].dtype)], 1)
        mask = torch.cat([b["mask"], torch.ones(tok.shape[:2], dtype=b["mask"].dtype, device=tok.device)], 1)
        return views, mask


class StructuredHead(nn.Module):
    def __init__(self, model_cfg, stage_a: StageA, z_mean, z_std, w_sem=0.1, sigma_max_sem=0.4, w_grasp=0.0):
        super().__init__()
        self.header = build_header(model_cfg, DZ, K * M)
        self.ctx = ContextTokens()
        self.A = stage_a                                     # frozen E/R/P (E only used for targets)
        for p in self.A.parameters():
            p.requires_grad_(False)
        self.register_buffer("z_mean", z_mean)
        self.register_buffer("z_std", z_std)
        self.w_sem, self.sigma_max_sem, self.w_grasp = w_sem, sigma_max_sem, w_grasp

    def _flat(self, z):
        return ((z - self.z_mean) / self.z_std).reshape(z.shape[0], K * M, DZ)

    def _unflat(self, x):
        return x.reshape(x.shape[0], K, M, DZ) * self.z_std + self.z_mean

    def loss(self, b):
        with torch.no_grad():
            mu, _ = self.A.E(self.A.morph, b["state0"], b["actions"])
        a = self._flat(mu)
        eps = torch.randn_like(a)
        sigma = torch.rand(a.shape[0], device=a.device)
        x = (1 - sigma[:, None, None]) * a + sigma[:, None, None] * eps
        views, mask = self.ctx(b)
        v = header_forward(self.header, x, sigma, views, mask, b["state0"]).float()
        fl = flow_loss(v, a, eps)
        logs, parts = dict(flow=float(fl.detach())), dict(flow=fl)
        loss = fl
        add_cmd_labels(b)
        if self.w_sem > 0 and "labels" in b:
            zc = self._unflat(x - sigma[:, None, None] * v)                                # clean estimate
            keep = (sigma <= self.sigma_max_sem).float()
            if keep.sum() > 0:
                out = self.A.P(zc)
                sl, slog = probe_loss({k: o[keep.bool()] for k, o in out.items()},
                                        {k: l[keep.bool()] for k, l in b["labels"].items()}, w_grasp=self.w_grasp)
                loss = loss + self.w_sem * sl
                parts["sem"] = sl
                logs.update({f"zhat_{k}": x_ for k, x_ in slog.items()})
        return loss, logs, parts

    @torch.no_grad()
    def sample_z(self, b, nfe=10, generator=None):
        B = b["state0"].shape[0]
        views, mask = self.ctx(b)
        x = torch.randn(B, K * M, DZ, device=b["state0"].device, generator=generator)
        for i in range(nfe):
            s = torch.full((B,), 1.0 - i / nfe, device=x.device)
            x = x - (1.0 / nfe) * header_forward(self.header, x, s, views, mask, b["state0"]).float()
        return self._unflat(x)

    @torch.no_grad()
    def realize(self, z, state, phase):
        return self.A.R(self.A.morph, z, state, phase)

    @torch.no_grad()
    def sample(self, b, nfe=10, generator=None):
        z = self.sample_z(b, nfe, generator)
        return self.realize(z, b["state0"], torch.zeros(z.shape[0], device=z.device))
