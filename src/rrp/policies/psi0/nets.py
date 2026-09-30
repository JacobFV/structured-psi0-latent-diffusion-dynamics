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
from rrp.policies.nets.attention import MHA, RelBlock
from rrp.policies.relations.base import EdgeSet, RelCtx, TokenSet, get_factor, require_factors, resolve
from rrp.policies.relations.ops import FactorSite
from rrp.policies.nets.flow import MLP, sinusoidal
from rrp.policies.nets.probes import ReadoutProbe, gaussian_nll, readout_defs, readout_loss, readout_metrics

K = 5
KNOT_STEPS = (5, 11, 17, 23, 29)
M = len(G.ASSEMBLIES)
DZ = 64
TP = 30                     # Psi0 chunk length
DA = G.ACTION_DIM           # 36

NETS_VERSION = "2"          # D-146 P1: relation-factor lists, constant-input mask, forced packet use (architecture 14.5)
PACKET_MARGIN = 0.05        # m of 14.5, normalized action units: hinge margin in StageA.loss and the heldout gate threshold
MASK_STD = 1e-4             # state dims with training std below this are constant inputs (14.5 a)
# The resolved factor list of a stage A (family "psi0", relations.md 11): the command-dim structure edges of E and R,
# the system-0 routing site of R, and the packet probe. `factors=` replaces the whole list (an ablation).
DEFAULT_FACTORS = ("preset:psi0-dims", "preset:s0-psi0", "preset:probes:psi0-v1")
GRASP_FACTORS = ("probe.psi0.grasp_pt", "probe.psi0.grasp_face")


def stage_a_specs(factors=None, grasp: bool = False):
    items = list(DEFAULT_FACTORS if factors is None else factors) + (list(GRASP_FACTORS) if grasp else [])
    return resolve(items, family="psi0")

# which assemblies' knots each assembly's command dims may read in system 0 (own + kinematic neighbours); canonical
# table: bodies.g1_simple.READS / reads_table() (also the `route.assembly_reads` factor's params.reads, D-144 R5).
_A = G.ASM_INDEX
READS = G.READS


def read_mask() -> torch.Tensor:
    """[DA, K*M] bool: command dim i may attend to knot token (k, m). Legacy shape, kept for its golden digest and
    `test_realizer_direct_reads_follow_reads_table`; the Realizer itself now gets this exact pattern from the
    `route.assembly_reads` FactorSite (test_route_assembly_reads_matches_read_mask)."""
    allow = G.reads_table()[G.DIM_ASM]                        # [DA, M]
    return torch.from_numpy(np.tile(allow, (1, K)))          # token order k-major: (k0,m0..m5),(k1,...)


class Morph(nn.Module):
    """Constant morphology buffers (node features, relations, assembly tokens). `masked=True` (stage A, structured
    head) adds the constant-input mask `state_keep` (trained state, stored in checkpoints)."""

    def __init__(self, masked: bool = False):
        super().__init__()
        self.register_buffer("node_static", torch.from_numpy(G.node_static()))                   # [DA, F]
        self.register_buffer("rel", torch.from_numpy(G.relation_matrix()).permute(1, 2, 0))       # [DA, DA, R]
        self.register_buffer("asm_static", torch.from_numpy(G.asm_static()))                      # [M, M+3]
        self.register_buffer("dim_asm", torch.from_numpy(G.DIM_ASM))
        self.register_buffer("read_mask", read_mask())
        sm = torch.zeros(DA, dtype=torch.bool)
        sm[:G.STATE_DIM] = True
        self.register_buffer("state_valid", sm)                                                   # dims with a state
        self.masked = masked
        if masked:      # 1 = the dim is a live input; 0 = constant in the training cache (D-141: zeroed at training AND inference)
            self.register_buffer("state_keep", torch.ones(DA))

    def keep_state(self, state):
        return state * self.state_keep[:state.shape[-1]].to(state.dtype) if self.masked else state


class DimEncoder(nn.Module):
    """Morphology-relation tokens: one token per command dim (static morphology + current state value + optional
    per-dim extra features), mixed by self-attention with a learned per-relation structural bias
    (self/parent/child/same-assembly/mirror). With the bias weights at 0 it is plain attention."""

    def __init__(self, D, heads=4, layers=2, extra=0, factors=None):
        super().__init__()
        self.inp = MLP(G.NODE_STATIC_DIM + 2 + extra, D)
        specs = self.specs = resolve(factors, default="psi0-dims")
        self.layers = nn.ModuleList([nn.ModuleDict(dict(n=nn.LayerNorm(D), a=MHA(D, heads),
                                                        sb=FactorSite(heads, D, "dims>dims", specs, ("edges:g1-dim-rel-v1",)),
                                                        n2=nn.LayerNorm(D), m=MLP(D, D, 4 * D)))
                                     for _ in range(layers)])

    def forward(self, morph: Morph, state, extra=None):
        B = state.shape[0]
        sv = morph.state_valid.to(state.dtype)
        x = torch.cat([morph.node_static[None].expand(B, -1, -1), (morph.keep_state(state[:, :DA]) * sv)[..., None],
                       sv[None, :, None].expand(B, -1, -1)] + ([extra] if extra is not None else []), -1)
        t = self.inp(x)
        rc = RelCtx(sets={"dims": TokenSet("dims", torch.ones(B, t.shape[1], dtype=torch.bool, device=t.device))},
                    edges={"dims>dims": EdgeSet(G.RELATIONS, morph.rel[None].expand(B, -1, -1, -1))})
        for L in self.layers:
            t = t + L["a"](L["n"](t), bias=L["sb"].bias(rc))
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

    def __init__(self, D=256, heads=4, layers=3, factors=None):
        super().__init__()
        self.dims = DimEncoder(D, heads, 2, extra=TP, factors=factors)
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
    (READS), applied as the `route.assembly_reads` factor (D-144 R5; params.reads = bodies.g1_simple.reads_table())
    through the shared `RelBlock` / `FactorSite` (mask form: same -inf pattern as the legacy `read_mask()` fill,
    tests/unit/test_psi0.py). Self-attention between command dims then mixes information across assemblies, so the
    locality is a routing prior, not an information barrier (tests/unit/test_psi0.py measures both)."""

    def __init__(self, D=256, heads=4, layers=3, factors=None):
        super().__init__()
        self.dims = DimEncoder(D, heads, 2, extra=2, factors=factors)
        self.z_in = nn.Linear(DZ, D)
        self.asm = MLP(M + 3, D)
        self.kt = MLP(D, D)
        self.blocks = nn.ModuleList([RelBlock(D, heads) for _ in range(layers)])
        self.out = nn.Linear(D, TP)
        self.D = D
        self.route = FactorSite(heads, D, "dims>knots", resolve(factors, default="s0-psi0"), ("assembly_id",))
        self.register_buffer("knot_asm", torch.from_numpy(np.tile(np.arange(M, dtype=np.int64), K)))  # k-major, D-144 R5

    def forward(self, morph, z, state, phase):
        """z [B,K,M,DZ]; state [B,>=32] normalized CURRENT state; phase [B] = tick index j / TP."""
        B = z.shape[0]
        ph = torch.stack([torch.sin(math.pi * phase), torch.cos(math.pi * phase)], -1)            # [B, 2]
        x = self.dims(morph, state, extra=ph[:, None].expand(-1, DA, -1))
        kt = torch.as_tensor(KNOT_STEPS, dtype=z.dtype, device=z.device) / TP
        rel = kt[None] - phase[:, None]                                                            # [B, K]
        tok = self.z_in(z) + self.kt(sinusoidal(rel, self.D))[:, :, None] + self.asm(morph.asm_static)[None, None]
        tok = tok.reshape(B, K * M, -1)
        rc = RelCtx(sets={
            "dims": TokenSet("dims", torch.ones(B, DA, dtype=torch.bool, device=z.device),
                             fields={"assembly_id": morph.dim_asm[None, :, None].expand(B, -1, -1)}),
            "knots": TokenSet("knots", torch.ones(B, K * M, dtype=torch.bool, device=z.device),
                              fields={"assembly_id": self.knot_asm[None, :, None].expand(B, -1, -1)})})
        bias = self.route.bias(rc)
        for L in self.blocks:
            x = L(x, tok, bias_x=bias)
        return self.out(x).transpose(1, 2)                                                         # [B, TP, DA]


# ---------------------------------------------------------------- probe
N_FACES = 6          # grasp contact-face class count (+-x, +-y, +-z); catalog.py's probe.psi0.grasp_face duplicates it
CMD_DIMS = (32, 34)          # vx, yaw-rate command (normalized action units) — packet LABELS, never inputs


def add_cmd_labels(b):
    """base_cmd[k] = demonstrated (vx, vyaw) command at knot k, normalized units (label only)."""
    if "labels" in b:
        b["labels"]["base_cmd"] = b["actions"][:, list(KNOT_STEPS)][..., list(CMD_DIMS)].float()
    return b


def _with_lv_min(specs, lv_min):
    """specs with each gauss-readout FactorSpec's `params.lv_min` set to `lv_min` (readout_loss reads it from
    `FactorSpec.params`, not a call-time argument; psi0's CLI configures it per training run, D-085)."""
    from dataclasses import replace as _replace
    out = []
    for s in specs:
        if get_factor(s.name).readout.loss == "gauss":
            p = dict(s.params); p["lv_min"] = lv_min
            s = _replace(s, params=tuple(sorted(p.items())))
        out.append(s)
    return tuple(out)


def probe_specs(grasp: bool = False):
    """`probes:psi0-v1` (+ the optional grasp-region factors); D-144 R5, replaces the former bespoke `PacketProbe`."""
    return ["preset:probes:psi0-v1"] + (["probe.psi0.grasp_pt", "probe.psi0.grasp_face"] if grasp else [])


def new_probe(D=192, heads=4, metadata_only=False, seed=1234, grasp=False, specs=None) -> ReadoutProbe:
    """The psi0 packet probe: a `ReadoutProbe` (docs/relations.md 4) configured by `probes:psi0-v1` (or by the readout
    factors of `specs`, e.g. a stage A's own list). `max_pairs=2` addresses the two hands (`hand_dist`, `contact`, and
    the optional grasp queries read both; `lift` / `target_pos` / `base_cmd` are per-knot only and read pair slot 0 —
    see `run_probe`)."""
    if specs is not None:
        specs = tuple(s for s, _ in readout_defs(specs))
    return ReadoutProbe(DZ, K, specs=probe_specs(grasp) if specs is None else specs, width=D, heads=heads, max_assemblies=M, max_pairs=2,
                        metadata_only=metadata_only, seed=seed)


def probe_has_grasp(P: ReadoutProbe) -> bool:
    return any(r.query == "grasp_pt" for _, r in readout_defs(P.specs))


def run_probe(P: ReadoutProbe, z: torch.Tensor) -> dict:
    """P(z) + the per-query slicing `ReadoutProbe`'s generic `knot×pair` addressing needs for psi0's per-knot-only
    queries (pair slot 0) and the grasp queries (last knot only, both hands)."""
    B = z.shape[0]
    zmask = torch.ones(B, M, dtype=torch.bool, device=z.device)
    out = dict(P(z, zmask, n_pairs=2))
    for q in ("lift", "target_pos", "base_cmd"):
        if q in out:
            out[q] = out[q][:, :, 0]
    for q in ("grasp_pt", "grasp_face"):
        if q in out:
            out[q] = out[q][:, -1]
    return out


def probe_loss(out, lab, specs, lv_min=-4.0, w_grasp=0.0):
    """lab: hand_dist [B,K,2], contact [B,K,2] {0,1}, lift [B,K], target_pos [B,K,3], active_hand [B] (long, -1 = none),
    base_disp [B,3]; *_valid masks optional. lv_min=-4 is the bounded-NLL setting (rrp D-085). `specs`: `P.specs`
    (`ReadoutProbe.specs`, resolved `probes:psi0-v1` (+ grasp)); the shared `readout_loss` (nets.probes) carries every
    query except grasp, which needs a runtime weight (`w_grasp`, not a spec-time one) and a validity mask."""
    lab2 = dict(lab)
    masks = {}
    if "hand_dist" in out:
        lab2["hand_dist"] = lab["hand_dist"][..., None]
    if "active_hand" in out:
        ah = lab["active_hand"]
        lab2["active_hand"], masks["active_hand"] = ah.clamp(min=0), ah >= 0
    core = _with_lv_min(tuple(s for s in specs if get_factor(s.name).readout.query not in ("grasp_pt", "grasp_face")),
                        lv_min)
    total, logs = readout_loss(out, lab2, core, masks)
    if w_grasp > 0 and "grasp_pt" in out and "grasp_valid" in lab:
        m = lab["grasp_valid"].bool()                                                    # [B, 2]
        if m.any():
            gp = gaussian_nll(out["grasp_pt"][m], lab["grasp_pt"][m],
                              torch.ones(int(m.sum()), dtype=torch.bool, device=m.device), lv_min)
            gf = F.cross_entropy(out["grasp_face"][m], lab["grasp_face"][m])
            total = total + w_grasp * gp + w_grasp * gf
            logs["probe_grasp_pt"], logs["probe_grasp_face"] = float(gp.detach()), float(gf.detach())
    return total, logs


@torch.no_grad()
def probe_metrics(out, lab, specs):
    """(sum, count) pairs; see `probe_loss` for the grasp / active_hand handling `readout_metrics` cannot infer
    generically."""
    lab2 = dict(lab)
    masks = {}
    if "hand_dist" in out:
        lab2["hand_dist"] = lab["hand_dist"][..., None]
    if "active_hand" in out:
        ah = lab["active_hand"]
        lab2["active_hand"], masks["active_hand"] = ah.clamp(min=0), ah >= 0
    core = tuple(s for s in specs if get_factor(s.name).readout.query not in ("grasp_pt", "grasp_face"))
    res = readout_metrics(out, lab2, core, masks)
    if "grasp_pt" in out and "grasp_valid" in lab and lab["grasp_valid"].any():
        m = lab["grasp_valid"].bool()
        e = (out["grasp_pt"][m][:, :3] - lab["grasp_pt"][m]).norm(dim=-1)
        res["grasp_pt_err"] = (float(e.sum()), int(e.numel()))
        f = out["grasp_face"][m].argmax(-1) == lab["grasp_face"][m]
        res["grasp_face_acc"] = (float(f.float().sum()), int(f.numel()))
    return res


def kl_std_normal(mu, lv):
    return 0.5 * (mu.pow(2) + lv.exp() - 1 - lv).sum(-1).mean()


class StageA(nn.Module):
    """E + R + P trained jointly (the packet's semantics are supervised ON z; R realizes the SAME z).

    `factors` (resolved with family "psi0", `stage_a_specs`) configures the command-dim edges of E and R, R's routing
    site and the probe; `factor_specs()` is what checkpoints stamp. Structure fix (architecture 14.5, D-141):
    `morph.state_keep` is the constant-input mask (`fit_state_mask`), stored in the checkpoint and applied inside every
    DimEncoder (E, R), and `loss` forces packet use (state dropout on R + permuted-packet hinge)."""

    def __init__(self, D=256, probe_D=192, metadata_only_probe=False, grasp=False, factors=None):
        super().__init__()
        self.specs = stage_a_specs(factors, grasp)
        self.cfg = dict(D=D, probe_D=probe_D, metadata_only_probe=metadata_only_probe,
                        factors=[s.to_dict() for s in self.specs])
        self.morph = Morph(masked=True)
        self.E = PacketEncoder(D, factors=self.specs)
        self.R = Realizer(D, factors=self.specs)
        self.P = new_probe(probe_D, metadata_only=metadata_only_probe, specs=self.specs)

    rec_dim_w = None      # optional [DA] reconstruction weights (set by the trainer)

    def factor_specs(self):
        return self.specs

    @torch.no_grad()
    def fit_state_mask(self, std, min_std=MASK_STD):
        """Constant-input mask: `std` [DA] = per-dim std of R's state input over the feature cache. Dims below `min_std`
        (and the non-state dims) are zeroed from now on, in training and at inference. Returns the masked dim ids."""
        keep = (std.to(self.morph.state_keep) >= min_std) & self.morph.state_valid
        self.morph.state_keep.copy_(keep.to(self.morph.state_keep))
        return [i for i in range(G.STATE_DIM) if not bool(keep[i])]

    def drop_state(self, s, p_dim, p_all):
        """Training-time state dropout on R's input: each dim zeroed with p_dim, the whole state with p_all (zero = the
        normalized mean, the value a masked dim carries at inference)."""
        keep = (torch.rand_like(s) >= p_dim) & (torch.rand(s.shape[0], 1, device=s.device) >= p_all)
        return s * keep.to(s.dtype)

    def recon_err(self, b, z, state=None):
        """Per-sample masked mean squared error of R(z) against the demonstrated chunk (rows >= j), normalized units."""
        pred = self.R(self.morph, z, b["state_j"] if state is None else state, b["j"].to(z.dtype) / TP)
        tmask = (torch.arange(TP, device=z.device)[None] >= b["j"][:, None]).to(z.dtype)[..., None]  # rows >= j
        m = b["amask"] * tmask
        if self.rec_dim_w is not None:
            m = m * self.rec_dim_w.to(m.device)
        return pred, m, (((pred - b["actions"]) ** 2) * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)

    @torch.no_grad()
    def packet_gap(self, b, z_mean):
        """Per-sample (err(R(z_mean)), err(R(E(a)))): R fed the dataset-mean packet vs the packet of the demonstrated
        chunk, same state and tick. The gap is the gate of architecture 14.5 (c): R that ignores the packet has none."""
        mu, _ = self.E(self.morph, b["state0"], b["actions"])
        return (self.recon_err(b, z_mean[None].expand_as(mu).to(mu))[2], self.recon_err(b, mu)[2])

    def loss(self, b, w_kl=1e-3, w_sem=1.0, lv_min=-4.0, z_noise=0.0, w_grasp=0.0, p_state_dim=0.3, p_state_all=0.1,
             w_perm=1.0, perm_margin=PACKET_MARGIN):
        """b: state0 [B,36] (packet start), actions [B,TP,DA] normalized, amask [B,TP,DA],
        j [B] realization tick, state_j [B,36] (state at tick j), labels (see probe_loss).
        Forced packet use (14.5 b): while training, R's state is dropped per dim (p_state_dim) and whole (p_state_all);
        the hinge `relu(margin - (err(R(z_perm)) - err(R(z))))` with z_perm = z of other samples of the batch (same
        state, tick, target) penalizes an R that realizes the chunk without reading its packet."""
        add_cmd_labels(b)
        mu, lv = self.E(self.morph, b["state0"], b["actions"])
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        zr = z + z_noise * torch.randn_like(z) if z_noise > 0 else z
        sj = self.drop_state(b["state_j"], p_state_dim, p_state_all) if self.training else b["state_j"]
        pred, m, err = self.recon_err(b, zr, sj)
        rec = (((pred - b["actions"]) ** 2) * m).sum() / m.sum().clamp(min=1)
        kl = kl_std_normal(mu, lv)
        pl, plog = probe_loss(run_probe(self.P, z), b["labels"], self.P.specs, lv_min=lv_min, w_grasp=w_grasp) \
            if w_sem > 0 else (z.sum() * 0, {})
        loss = rec + w_kl * kl + w_sem * pl
        parts, logs = dict(rec=rec, kl=kl, sem=pl), dict(rec=float(rec.detach()), kl=float(kl.detach()), **plog)
        if w_perm > 0 and z.shape[0] > 1:
            shift = int(torch.randint(1, z.shape[0], (1,)))
            err_perm = self.recon_err(b, zr.roll(shift, 0), sj)[2]
            hinge = F.relu(perm_margin - (err_perm - err)).mean()
            loss = loss + w_perm * hinge
            parts["perm"] = hinge
            logs.update(perm_hinge=float(hinge.detach()), perm_gap=float((err_perm - err).mean().detach()))
        return loss, logs, parts


def load_stage_a(path, map_location="cpu", factors=None, allow_factor_mismatch=False):
    """Load a stage-A checkpoint written by `train --arm stageA` (`nets.checkpoint.save_checkpoint`). The net is rebuilt
    from the checkpoint's own `config["stage_a"]` (widths + resolved factor list) and validated: config present and
    complete, the factor structure hash equals the stamped `versions["factors"]`, every tensor (incl. the constant-input
    mask `morph.state_keep`) present with its shape (strict load). `factors` = the list the caller expects (must hash
    equal unless `allow_factor_mismatch`). A checkpoint written before D-146 P1 has no config and is refused: it has
    no input mask and its R bypasses the packet (D-141)."""
    state = torch.load(path, weights_only=False, map_location=map_location)
    cfg = (state.get("config") or {}).get("stage_a") if isinstance(state, dict) else None
    need = {"D", "probe_D", "metadata_only_probe", "factors"}
    if not isinstance(cfg, dict) or need - set(cfg):
        raise ValueError(f"{path}: not a P1 stage-A checkpoint (config['stage_a'] missing keys "
                         f"{sorted(need - set(cfg or {}))}); retrain stage A (architecture 14.5)")
    A = StageA(cfg["D"], cfg["probe_D"], cfg["metadata_only_probe"], factors=cfg["factors"])
    require_factors(state.get("versions"), A.specs)
    if factors is not None:
        require_factors(state.get("versions"), stage_a_specs(factors), allow_factor_mismatch)
    A.load_state_dict(state["model"], strict=True)
    return A


def load_tolerant(module, sd):
    """load_state_dict for modules containing an older stage A / structured-head checkpoint: pads the v1 probe's
    `qtype` row (pre-base_cmd), and, for a PRE-D-144 checkpoint (probe module keys `P.a1.` / `P.a2.` / `P.kcode.` /
    `P.code.`: the old `PacketProbe`), drops every `P.*` key so the probe keeps its fresh init — the pre-D-144
    `PacketProbe` and the current `ReadoutProbe` (`probes:psi0-v1`, R5) are different architectures (retrained /
    refit, not loaded); `R` / `E` / `morph` are unaffected and always load strictly. A post-D-144 checkpoint's `P.*`
    loads and is checked strictly like everything else."""
    own = module.state_dict()
    for k, v in list(sd.items()):
        if k in own and own[k].shape != v.shape and k.endswith("qtype.weight"):
            w = own[k].clone(); w[: v.shape[0]] = v; sd[k] = w
    old_probe = any(k.startswith(("P.a1.", "P.a2.", "P.kcode.", "P.code.")) for k in sd)
    if old_probe:
        for k in [k for k in sd if k.startswith("P.")]:
            del sd[k]
    missing, unexpected = module.load_state_dict(sd, strict=False)
    bad = [m for m in missing if "base_cmd" not in m and not (old_probe and m.startswith("P."))] + \
        [u for u in unexpected if not (old_probe and u.startswith("P."))]
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

    def __init__(self, D=512, factors=None, masked=False):
        super().__init__()
        self.morph = Morph(masked)
        self.dims = DimEncoder(D, heads=8, layers=2, factors=factors)
        self.specs = self.dims.specs
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
    def __init__(self, model_cfg, stage_a: StageA, z_mean, z_std, w_sem=0.1, sigma_max_sem=0.4, w_grasp=0.0,
                 factors=None):
        super().__init__()
        self.header = build_header(model_cfg, DZ, K * M)
        self.ctx = ContextTokens(factors=factors, masked=True)            # `factors`: the context tokens' command-dim edges
        self.A = stage_a                                     # frozen E/R/P (E only used for targets)
        for p in self.A.parameters():
            p.requires_grad_(False)
        self.ctx.morph.state_keep.copy_(self.A.morph.state_keep)      # the same constant-input mask (D-141, 14.5 a)
        self.register_buffer("z_mean", z_mean)
        self.register_buffer("z_std", z_std)
        self.w_sem, self.sigma_max_sem, self.w_grasp = w_sem, sigma_max_sem, w_grasp

    def factor_specs(self):
        """Context-token factors then the frozen stage A's list (what `save_checkpoint` stamps)."""
        return tuple(self.ctx.specs) + tuple(self.A.specs)

    def state0(self, b):
        """The header's state input: the packet-start state with the stage A's constant dims zeroed (14.5 a)."""
        return self.A.morph.keep_state(b["state0"])

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
        v = header_forward(self.header, x, sigma, views, mask, self.state0(b)).float()
        fl = flow_loss(v, a, eps)
        logs, parts = dict(flow=float(fl.detach())), dict(flow=fl)
        loss = fl
        add_cmd_labels(b)
        if self.w_sem > 0 and "labels" in b:
            zc = self._unflat(x - sigma[:, None, None] * v)                                # clean estimate
            keep = (sigma <= self.sigma_max_sem).float()
            if keep.sum() > 0:
                out = run_probe(self.A.P, zc)                      # all-true assembly mask; the fixed G1 assemblies
                sl, slog = probe_loss({k: o[keep.bool()] for k, o in out.items()},
                                        {k: l[keep.bool()] for k, l in b["labels"].items()}, self.A.P.specs,
                                        w_grasp=self.w_grasp)
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
            x = x - (1.0 / nfe) * header_forward(self.header, x, s, views, mask, self.state0(b)).float()
        return self._unflat(x)

    @torch.no_grad()
    def realize(self, z, state, phase):
        return self.A.R(self.A.morph, z, state, phase)

    @torch.no_grad()
    def sample(self, b, nfe=10, generator=None):
        z = self.sample_z(b, nfe, generator)
        return self.realize(z, b["state0"], torch.zeros(z.shape[0], device=z.device))


def load_structured(head: StructuredHead, path, allow_factor_mismatch=False):
    """Load a structured-head checkpoint into a head built from its stage A: the stamped factor structure (context
    tokens + stage A) must equal the head's, then `load_tolerant`."""
    state = torch.load(path, weights_only=False, map_location="cpu")
    require_factors(state.get("versions"), head.factor_specs(), allow_factor_mismatch)
    return load_tolerant(head, state["model"])
