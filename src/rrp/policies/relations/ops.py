"""Operators, forms and the per-attention-call `FactorSite` (D-144; docs/relations.md 3.2-3.5).

An operator turns fields / edges of a site's two token sets into a relation value; a form says how it enters the
attention logits:
  bias : additive [B,H,Q,K] term (all `edge` factors of a site are ONE stacked einsum `bqkr,rh->bhqk` over `FactorSite.w`)
  aug  : kernel-compatible q/k feature augmentation, <phi_q(i), phi_k(j)> = value(i, j) up to i-only terms
  mask : hard routing, -inf where the value is False
Every learned coefficient starts at zero (enabling a factor on a trained model is a no-op at step 0) and no
constructor draws from the global torch RNG (fixed codes / projections use a generator seeded by the factor name), so
adding factors never shifts the initialization of the rest of a net.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rrp.policies.relations.base import (FactorError, FactorSpec, PrivilegedInput, RelCtx, effective_source,
                                         get_factor)

GRAPH_CONTROLS = ("on", "off", "zero", "rewired", "shuffled", "reversed", "gt", "estimated")
FIELD_CONTROLS = ("on", "off", "zero", "rewired", "shuffled", "gt", "estimated")


def _gen(name: str) -> torch.Generator:
    return torch.Generator().manual_seed(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))


def _key(name: str) -> str:
    return name.replace(".", "__")


# ------------------------------------------------------------------ controls (shared by every operator)
def rewire_keys(x: torch.Tensor, key_mask: torch.Tensor | None, generator, key_dim: int = 2) -> torch.Tensor:
    """Per sample, permute the first n_valid key positions (n_valid = number of valid keys). This is the legacy
    `transform_relations(..., "rewired")` rule, kept bit-exact (goldens rel.flow.rewired)."""
    B, K = x.shape[0], x.shape[key_dim]
    out = torch.zeros_like(x)
    for b in range(B):
        nvalid = int(key_mask[b].sum()) if key_mask is not None else K
        perm = torch.randperm(nvalid, generator=generator)
        idx = torch.arange(K)
        idx[:nvalid] = perm
        out[b] = x[b].index_select(key_dim - 1, idx.to(x.device))
    return out


def control_graph(rel: torch.Tensor, control: str, rc: RelCtx, site: str) -> torch.Tensor:
    """rel [B,Q,K,R] (or [B,Q,K]) under a control. `reversed` transposes square self-attention sites only."""
    if control in ("on", "gt", "estimated"):
        return rel
    if control == "zero":
        return torch.zeros_like(rel)
    if control == "reversed":
        return rel.transpose(1, 2) if rel.shape[1] == rel.shape[2] else rel
    if control == "rewired":
        return rewire_keys(rel, rc.token_sets(site)[1].mask, rc.generator)
    if control == "shuffled":
        if rel.dim() < 4 or rel.shape[-1] < 2:
            return rel
        out = torch.empty_like(rel)
        for b in range(rel.shape[0]):
            out[b] = rel[b][..., torch.randperm(rel.shape[-1], generator=rc.generator)]
        return out
    raise FactorError(f"control {control!r} does not apply to a graph")


def control_field(v: torch.Tensor, control: str, mask: torch.Tensor, rc: RelCtx, side: str) -> torch.Tensor:
    """v [B,T,d] under a control; side 'q' or 'k' (rewired permutes only the key side)."""
    if control in ("on", "gt", "estimated"):
        return v
    if control == "zero":
        return torch.zeros_like(v)
    if control == "rewired":
        return rewire_keys(v, mask, rc.generator, key_dim=1) if side == "k" else v
    if control == "shuffled":
        return rewire_keys(v, mask, rc.generator, key_dim=1)
    raise FactorError(f"control {control!r} does not apply to a field")


def site_field(rc: RelCtx, set_name: str, name: str, s: FactorSpec):
    """(value [B,T,d], var or None) of a field from the spec's effective source."""
    ts = rc.sets[set_name]
    src = effective_source(s)
    if src == "gt":
        if rc.deploy or ts.deploy:
            raise PrivilegedInput(f"factor {s.name}: ground-truth field {name!r} in deploy mode")
        return ts.label(name), None
    if src == "probe":
        if (set_name, name) not in rc.estimates:
            raise FactorError(f"factor {s.name}: no readout estimate of {set_name}.{name} (readout_layer not reached?)")
        return rc.estimates[(set_name, name)]
    return ts.field(name), ts.fields.get(name + ".var")


# ------------------------------------------------------------------ operators
class Op:
    forms: tuple = ()
    graph = False                      # value computed from graphs (bias / mask), else field / hidden features (aug)

    def controls(self, form: str) -> tuple:
        if form == "mask":
            return ("on", "off", "zero", "rewired")
        return GRAPH_CONTROLS if self.graph else FIELD_CONTROLS

    def build(self, d, s, heads: int, dim: int) -> nn.Module | None:
        return None

    def value(self, d, s, mod, rc: RelCtx, site: str) -> torch.Tensor:        # graph ops: [B,Q,K] float / bool
        raise NotImplementedError

    def features(self, d, s, mod, rc: RelCtx, site: str, xq, xk):            # aug ops: ([B,H,Q,A], [B,H,K,A])
        raise NotImplementedError


class EdgeOp(Op):
    """A given / estimated graph channel. As `bias` it is part of the site's stacked typed group."""
    forms = ("bias", "mask")
    graph = True

    def value(self, d, s, mod, rc, site):
        es = rc.edges[site + "@gt"] if effective_source(s) == "gt" else rc.edges[site]
        if effective_source(s) == "gt" and rc.deploy:
            raise PrivilegedInput(f"factor {s.name}: ground-truth edges in deploy mode")
        return es.data[..., es.channel({**d.p, **s.p}["edge"])]


def _closure(A: torch.Tensor) -> torch.Tensor:
    """Transitive closure of bool [B,T,T] (paths of length >= 1)."""
    R = A.clone()
    for _ in range(max(1, math.ceil(math.log2(max(A.shape[-1], 2))) + 1)):
        R2 = R | (R.float() @ R.float() > 0)
        if torch.equal(R2, R):
            break
        R = R2
    return R


class ClosureOp(EdgeOp):
    """ancestor / flow: closure of a directed edge channel. params.rel: closure | inverse | sibling."""
    forms = ("bias",)

    def value(self, d, s, mod, rc, site):
        A = super().value(d, s, mod, rc, site).bool()
        if A.shape[1] != A.shape[2]:
            raise FactorError(f"{s.name}: closure needs a square self site, got {site}")
        C = _closure(A)
        rel = {**d.p, **s.p}.get("rel", "closure")
        if rel == "inverse":
            C = C.transpose(1, 2)
        elif rel == "sibling":
            eye = torch.eye(A.shape[1], dtype=torch.bool, device=A.device)[None]
            C = ((A.float() @ A.transpose(1, 2).float()) > 0) & ~eye
        return C


class HopOp(EdgeOp):
    """1[graph distance(i, j) == hops] over an edge channel (symmetrized when params.symmetric)."""
    forms = ("bias",)

    def value(self, d, s, mod, rc, site):
        p = {**d.p, **s.p}
        A = super().value(d, s, mod, rc, site).bool()
        if p.get("symmetric", True):
            A = A | A.transpose(1, 2)
        k = int(p.get("hops", 2))
        reach = torch.eye(A.shape[1], dtype=torch.bool, device=A.device)[None].expand_as(A).clone()
        seen = reach.clone()
        for _ in range(k):
            reach = (reach.float() @ A.float() > 0) & ~seen
            seen = seen | reach
        return reach


class OrderOp(Op):
    """sign((r_j - r_i) . axis) with a dead zone (above / below, left / right, ...; antisymmetric)."""
    forms = ("bias",)
    graph = True

    def value(self, d, s, mod, rc, site):
        p = {**d.p, **s.p}
        q, k = site.split(">")
        rq, _ = site_field(rc, q, d.field, s)
        rk, _ = site_field(rc, k, d.field, s)
        axis = torch.as_tensor(p.get("axis", (0.0, 0.0, 1.0)), dtype=rq.dtype, device=rq.device)
        dlt = (rk[:, None, :, :] - rq[:, :, None, :]) @ axis
        return torch.sign(dlt) * (dlt.abs() > float(p.get("margin", 0.01)))


class SameOp(Op):
    """1[id_i == id_j] of an integer id field. mask form: hard routing (route.*), optional params.reads = allowed
    [n_ids_q, n_ids_k] table and params.also_key_field = key-set bool field of always-readable keys. aug form: fixed
    random unit codes per id value (exact equality for n_ids <= code dim)."""
    forms = ("bias", "aug", "mask")
    graph = True

    def controls(self, form):
        return ("on", "off", "zero", "rewired") if form == "mask" else FIELD_CONTROLS

    def build(self, d, s, heads, dim):
        if d.form != "aug":
            return None
        p = {**d.p, **s.p}
        n, c = int(p.get("n_ids", 64)), int(p.get("code_dim", 16))
        m = nn.Module()
        r = torch.randn(max(n, c), c, generator=_gen(s.name))
        code = torch.linalg.qr(r.T)[0].T[:n] if n <= c else F.normalize(r[:n], dim=-1)   # orthonormal when n <= c
        m.register_buffer("code", code.contiguous())
        m.g = nn.Parameter(torch.zeros(heads))
        return m

    def _ids(self, rc, set_name, d, s):
        v, _ = site_field(rc, set_name, d.field, s)
        return v[..., 0].long() if v.dim() == 3 else v.long()

    def value(self, d, s, mod, rc, site):
        q, k = site.split(">")
        iq, ik = self._ids(rc, q, d, s), self._ids(rc, k, d, s)
        p = {**d.p, **s.p}
        if "reads" in p:
            tab = torch.as_tensor(p["reads"], dtype=torch.bool, device=iq.device)
            ok = tab[iq.clamp(min=0)[:, :, None], ik.clamp(min=0)[:, None, :]]
        else:
            ok = iq[:, :, None] == ik[:, None, :]
        ok = ok & (iq[:, :, None] >= 0) & (ik[:, None, :] >= 0)
        if "also_key_field" in p:
            ok = ok | rc.sets[k].field(p["also_key_field"]).bool()[:, None, :]
        return ok

    def features(self, d, s, mod, rc, site, xq, xk):
        q, k = site.split(">")
        eq = mod.code[self._ids(rc, q, d, s).clamp(min=0)] * (self._ids(rc, q, d, s) >= 0)[..., None]
        ek = mod.code[self._ids(rc, k, d, s).clamp(min=0)] * (self._ids(rc, k, d, s) >= 0)[..., None]
        return mod.g[None, :, None, None] * eq[:, None], ek[:, None].expand(-1, mod.g.shape[0], -1, -1)


class _Coeff(nn.Module):
    """Zero-init linear map from the query hidden to per-head coefficients (no global-RNG draws)."""

    def __init__(self, dim, heads, width):
        super().__init__()
        self.heads, self.width = heads, width
        self.weight = nn.Parameter(torch.zeros(heads * width, dim))
        self.bias = nn.Parameter(torch.zeros(heads * width))

    def forward(self, x):                                      # [B,T,D] -> [B,H,T,width]
        y = F.linear(x, self.weight, self.bias)
        return y.view(*x.shape[:2], self.heads, self.width).transpose(1, 2)


class PapeOp(Op):
    """PaPE (arXiv 2602.01418, Eq. 9) in the p-dim compact kernel-compatible form:
        l_ij += -(r_j - r_i)^T M_i (r_j - r_i) + bt_i^T (r_j - r_i),  M_i = P_i^T diag(a_i) P_i,  bt_i = P_i^T b_i
        a_i = g_h * softplus(W_a x_i) >= 0,  b_i = W_b x_i,  P_i = W_p (frame world) | W_p R_i^T (frame query)
        phi_q = [-vec(M_i), 2 M_i r_i + bt_i],  phi_k = [vec(r_j r_j^T), r_j]      (p^2 + p dims; i-only terms dropped)
    params: m (projection rows, default p), frame ('world' | 'query'; query needs params.orient = orientation field)."""
    forms = ("aug",)

    def build(self, d, s, heads, dim):
        p = {**d.p, **s.p}
        P, m_ = int(p["p"]), int(p.get("m", p["p"]))
        mod = nn.Module()
        mod.Wp = nn.Parameter(torch.randn(heads, m_, P, generator=_gen(s.name)) / math.sqrt(P))
        mod.a = _Coeff(dim, heads, m_)
        mod.b = _Coeff(dim, heads, m_)
        mod.g = nn.Parameter(torch.zeros(heads))
        return mod

    def features(self, d, s, mod, rc, site, xq, xk):
        q, k = site.split(">")
        p = {**d.p, **s.p}
        rq, _ = site_field(rc, q, d.field, s)
        rk, _ = site_field(rc, k, d.field, s)
        rq = control_field(rq, s.control, rc.sets[q].mask, rc, "q")
        rk = control_field(rk, s.control, rc.sets[k].mask, rc, "k")
        a = mod.g[None, :, None, None] * F.softplus(mod.a(xq))                   # [B,H,Q,m]
        b = mod.b(xq)                                                            # [B,H,Q,m]
        Wp = mod.Wp                                                              # [H,m,p]
        if p.get("frame", "world") == "query":
            R = rc.sets[q].field(p.get("orient", "orient")).view(*rq.shape[:2], 3, 3)   # [B,Q,3,3] columns = axes
            Pi = torch.einsum("hmp,bqsp->bhqms", Wp, R)                          # W_p R_i^T
        else:
            Pi = Wp[None, :, None].expand(rq.shape[0], -1, rq.shape[1], -1, -1)  # [B,H,Q,m,p]
        M = torch.einsum("bhqmp,bhqm,bhqmr->bhqpr", Pi, a, Pi)
        bt = torch.einsum("bhqmp,bhqm->bhqp", Pi, b)
        Mr = torch.einsum("bhqpr,bqr->bhqp", M, rq)
        phi_q = torch.cat([-M.flatten(-2), 2 * Mr + bt], -1)
        rr = torch.einsum("bkp,bkr->bkpr", rk, rk).flatten(-2)
        phi_k = torch.cat([rr, rk], -1)[:, None].expand(-1, Wp.shape[0], -1, -1)
        return phi_q, phi_k


class DiffOp(Op):
    """Direction only: <b_i, r_j - r_i> -> phi_q = b_i, phi_k = r_j (b_i = W_b x_i, zero-init)."""
    forms = ("aug",)

    def build(self, d, s, heads, dim):
        mod = nn.Module()
        mod.b = _Coeff(dim, heads, int({**d.p, **s.p}["p"]))
        return mod

    def features(self, d, s, mod, rc, site, xq, xk):
        q, k = site.split(">")
        rk, _ = site_field(rc, k, d.field, s)
        rk = control_field(rk, s.control, rc.sets[k].mask, rc, "k")
        return mod.b(xq), rk[:, None].expand(-1, mod.b.heads, -1, -1)


class RelRotOp(Op):
    """<B_i, R_i^T R_j>_F = <R_i B_i, R_j>_F; B_i = W_B x_i (per head 3x3, zero-init). Orientation field: 9 = row-major
    3x3 rotation (columns = the token frame's axes in the common frame)."""
    forms = ("aug",)

    def build(self, d, s, heads, dim):
        mod = nn.Module()
        mod.B = _Coeff(dim, heads, 9)
        return mod

    def features(self, d, s, mod, rc, site, xq, xk):
        q, k = site.split(">")
        Rq, _ = site_field(rc, q, d.field, s)
        Rk, _ = site_field(rc, k, d.field, s)
        Rk = control_field(Rk, s.control, rc.sets[k].mask, rc, "k")
        Bq = mod.B(xq).view(*mod.B(xq).shape[:3], 3, 3)                          # [B,H,Q,3,3]
        RB = torch.einsum("bqst,bhqtu->bhqsu", Rq.view(*Rq.shape[:2], 3, 3), Bq)
        return RB.flatten(-2), Rk[:, None].expand(-1, mod.B.heads, -1, -1)


class AlignOp(Op):
    """<b_i, R_i^T n_j> (frame query, needs params.orient) or <b_i, n_j> (world): phi_q = R_i b_i | b_i, phi_k = n_j."""
    forms = ("aug",)

    def build(self, d, s, heads, dim):
        mod = nn.Module()
        mod.b = _Coeff(dim, heads, 3)
        return mod

    def features(self, d, s, mod, rc, site, xq, xk):
        q, k = site.split(">")
        p = {**d.p, **s.p}
        nk, _ = site_field(rc, k, d.field, s)
        nk = control_field(nk, s.control, rc.sets[k].mask, rc, "k")
        b = mod.b(xq)
        if p.get("frame", "world") == "query":
            R = rc.sets[q].field(p.get("orient", "orient")).view(*b.shape[:1], b.shape[2], 3, 3)
            b = torch.einsum("bqst,bhqt->bhqs", R, b)
        return b, nk[:, None].expand(-1, mod.b.heads, -1, -1)


class SimOp(Op):
    """cos(v_i, v_j) of a property vector: phi_q = g_h v_i/|v_i|, phi_k = v_j/|v_j|."""
    forms = ("aug",)

    def build(self, d, s, heads, dim):
        mod = nn.Module()
        mod.g = nn.Parameter(torch.zeros(heads))
        return mod

    def features(self, d, s, mod, rc, site, xq, xk):
        q, k = site.split(">")
        vq, _ = site_field(rc, q, d.field, s)
        vk, _ = site_field(rc, k, d.field, s)
        vk = control_field(vk, s.control, rc.sets[k].mask, rc, "k")
        vq, vk = F.normalize(vq, dim=-1), F.normalize(vk, dim=-1)
        return mod.g[None, :, None, None] * vq[:, None], vk[:, None].expand(-1, mod.g.shape[0], -1, -1)


class UnaryOp(Op):
    """Key-side property prior <w_h, f_j>: phi_q = w_h (constant over queries), phi_k = f_j (w zero-init)."""
    forms = ("aug",)

    def build(self, d, s, heads, dim):
        mod = nn.Module()
        mod.w = nn.Parameter(torch.zeros(heads, int({**d.p, **s.p}["p"])))
        return mod

    def features(self, d, s, mod, rc, site, xq, xk):
        q, k = site.split(">")
        fk, _ = site_field(rc, k, d.field, s)
        fk = control_field(fk, s.control, rc.sets[k].mask, rc, "k")
        B, Q = rc.sets[q].mask.shape
        return mod.w[None, :, None].expand(B, -1, Q, -1), fk[:, None].expand(-1, mod.w.shape[0], -1, -1)


class BilinearOp(Op):
    """Learned kernel g_h <U_h x_i, V_h x_j> on token hiddens; the same score is the factor's pair probe
    (p_ij = sigmoid(<U x_i, V x_j> + c)). U, V from a name-seeded generator; g zero-init."""
    forms = ("aug",)

    def controls(self, form):
        return ("on", "off", "zero", "rewired")

    def build(self, d, s, heads, dim):
        r = int({**d.p, **s.p}.get("rank", 8))
        g = _gen(s.name)
        mod = nn.Module()
        mod.U = nn.Parameter(torch.randn(heads * r, dim, generator=g) / math.sqrt(dim))
        mod.V = nn.Parameter(torch.randn(heads * r, dim, generator=g) / math.sqrt(dim))
        mod.g = nn.Parameter(torch.zeros(heads))
        mod.c = nn.Parameter(torch.zeros(1))
        mod.heads, mod.rank = heads, r
        return mod

    def scores(self, mod, xq, xk):
        u = F.linear(xq, mod.U).view(*xq.shape[:2], mod.heads, mod.rank).transpose(1, 2)
        v = F.linear(xk, mod.V).view(*xk.shape[:2], mod.heads, mod.rank).transpose(1, 2)
        return u, v

    def features(self, d, s, mod, rc, site, xq, xk):
        if xk is None:
            raise FactorError(f"{s.name}: bilinear needs the key hiddens at {site}")
        u, v = self.scores(mod, xq, xk)
        if s.control == "zero":
            v = torch.zeros_like(v)
        elif s.control == "rewired":
            v = rewire_keys(v, rc.token_sets(site)[1].mask, rc.generator, key_dim=2)
        return mod.g[None, :, None, None] * u, v


class InertOp(Op):
    """message / embed / readout forms: implemented by the net family (they do not touch attention logits)."""
    forms = ("message", "embed", "readout")

    def controls(self, form):
        return ("on", "off", "serialized") if form == "message" else ("on", "off")


OPS: dict[str, Op] = {"edge": EdgeOp(), "ancestor": ClosureOp(), "flow": ClosureOp(), "hop": HopOp(),
                      "order": OrderOp(), "same": SameOp(), "sim": SimOp(), "diff": DiffOp(),
                      "sqdiff+diff": PapeOp(), "rel_rot": RelRotOp(), "align": AlignOp(), "unary": UnaryOp(),
                      "bilinear": BilinearOp(), "inert": InertOp()}


# ------------------------------------------------------------------ the site
def _applies(d, s: FactorSpec, site: str, carries: tuple) -> bool:
    from rrp.policies.relations.catalog import VOCABS
    if s.sites is not None and site not in s.sites:
        return False
    if d.form in ("message", "embed", "readout"):
        return False
    if d.field.startswith("edges:"):
        edge = {**d.p, **s.p}.get("edge")
        for c in carries:
            if c.startswith("edges:") and (d.field in ("edges:*", c)) and (edge is None or edge in VOCABS[c[6:]]):
                return True
        return False
    return d.field in carries or (d.op == "bilinear" and "hidden" in carries)


class FactorSite(nn.Module):
    """The factors of one attention call. `site` = "q>k"; `carries` = what the net provides there ("edges:<vocab>",
    field names, "hidden" when key hiddens are passed). Parameters: `w` [R, H] for the stacked `edge`/`bias` group (the
    path of the former StructuralBias.w), per-factor modules under `f.<factor>` for the rest."""

    def __init__(self, heads: int, dim: int, site: str, specs, carries: tuple):
        super().__init__()
        self.heads, self.site = heads, site
        self.specs = tuple(s for s in specs if _applies(get_factor(s.name), s, site, tuple(carries)))
        self.edge_specs = tuple(s for s in self.specs if get_factor(s.name).op == "edge" and get_factor(s.name).form == "bias")
        self.w = nn.Parameter(torch.full((len(self.edge_specs), heads), 0.0)) if self.edge_specs else None
        self.f = nn.ModuleDict()
        for s in self.specs:
            d = get_factor(s.name)
            if s in self.edge_specs:
                continue
            m = OPS[d.op].build(d, s, heads, dim)
            if d.form in ("bias",) and not isinstance(m, nn.Module):
                m = nn.Module()
            if d.form == "bias":
                m.w = nn.Parameter(torch.zeros(heads))
            if d.form in ("bias", "aug") and s.gate is not None:
                m.gate_u = nn.Parameter(torch.zeros(heads, dim))
                m.gate_b = nn.Parameter(torch.zeros(heads))
            if m is not None:
                self.f[_key(s.name)] = m
        if any(s.gate is not None for s in self.edge_specs):
            self.edge_gate_u = nn.Parameter(torch.zeros(len(self.edge_specs), heads, dim))
            self.edge_gate_b = nn.Parameter(torch.zeros(len(self.edge_specs), heads))

    # ---- helpers
    def _head_mask(self, s, device, dtype):
        if s.heads is None:
            return None
        m = torch.zeros(self.heads, device=device, dtype=dtype)
        m[list(s.heads)] = 1
        return m

    def _gate(self, u, b, rc, s):                               # -> [B, ..., H] multiplier
        c = rc.summaries.get(s.gate) if s.gate != "task" else (rc.summaries.get("task", rc.task))
        if c is None:
            raise FactorError(f"{s.name}: gate {s.gate!r} needs rc.summaries[{s.gate!r}]")
        return torch.sigmoid(torch.einsum("...hd,bd->b...h", u, c) + b) * 2   # 1.0 at init (u = 0, b = 0)

    def _conf(self, rc, s):
        q, k = self.site.split(">")
        f = {**get_factor(s.name).p, **s.p}.get("confidence_field", "pos3d")
        out = []
        for n in (q, k):
            var = rc.sets[n].fields.get(f + ".var")
            out.append(torch.ones_like(rc.sets[n].mask, dtype=torch.float32) if var is None else torch.exp(-var.mean(-1)))
        return out

    def _edges(self, rc: RelCtx) -> torch.Tensor | None:
        """Stacked, control-transformed edge channels of the active edge factors (memoized per site and forward)."""
        act = [s for s in self.edge_specs if s.control != "off"]
        if not act:
            return None
        key = (self.site, tuple((s.name, s.control) for s in act))
        if key in rc.memo:
            return rc.memo[key]
        gt = [s for s in act if effective_source(s) == "gt"]
        if gt and rc.deploy:
            raise PrivilegedInput(f"ground-truth edges in deploy mode: {[s.name for s in gt]}")
        es = rc.edges[self.site]
        idx = [es.channel({**get_factor(s.name).p, **s.p}["edge"]) for s in act]
        rel = es.data if idx == list(range(es.data.shape[-1])) else es.data[..., idx]
        if gt:
            g = rc.edges[self.site + "@gt"]
            for j, s in enumerate(act):
                if s in gt:
                    rel = rel.clone() if rel is es.data else rel
                    rel[..., j] = g.data[..., g.channel({**get_factor(s.name).p, **s.p}["edge"])].to(rel.dtype)
        controls = [s.control for s in act]
        if len(set(controls)) == 1:
            rel = control_graph(rel, controls[0], rc, self.site)
        else:
            rel = torch.stack([control_graph(rel[..., j], c, rc, self.site) for j, c in enumerate(controls)], -1)
        rc.memo[key] = rel
        return rel

    # ---- the two entry points
    def bias(self, rc: RelCtx) -> torch.Tensor | None:
        """x-independent additive term [B, H|1, Q, K] (typed edge group + other graph ops + masks). Cacheable."""
        out = None
        rel = self._edges(rc)
        if rel is not None:
            act = [i for i, s in enumerate(self.edge_specs) if s.control != "off"]
            w = self.w if len(act) == len(self.edge_specs) else self.w[act]
            srow = [self.edge_specs[i] for i in act]
            hm = [self._head_mask(s, w.device, w.dtype) for s in srow]
            if any(m is not None for m in hm):
                w = w * torch.stack([m if m is not None else torch.ones_like(w[0]) for m in hm])
            if any(s.confidence for s in srow):
                cq, ck = self._conf(rc, srow[0])
                rel = rel.to(w.dtype) * torch.stack([(cq[:, :, None] * ck[:, None, :]) if s.confidence else
                                                     torch.ones_like(cq[:, :, None] * ck[:, None, :]) for s in srow], -1)
            if any(s.gate is not None for s in srow):
                gu, gb = self.edge_gate_u[act], self.edge_gate_b[act]
                gam = self._gate(gu, gb, rc, next(s for s in srow if s.gate is not None))        # [B,R,H]
                gam = torch.where(torch.tensor([s.gate is not None for s in srow], device=gam.device)[None, :, None],
                                  gam, torch.ones_like(gam))
                out = torch.einsum("bqkr,brh->bhqk", rel.to(w.dtype), w[None] * gam)
            else:
                out = torch.einsum("bqkr,rh->bhqk", rel.to(w.dtype), w)
        for s in self.specs:
            d = get_factor(s.name)
            if s in self.edge_specs or s.control == "off" or d.form not in ("bias", "mask"):
                continue
            v = OPS[d.op].value(d, s, self.f[_key(s.name)] if _key(s.name) in self.f else None, rc, self.site)
            if d.form == "mask":
                v = control_graph(v, s.control, rc, self.site)
                if s.control == "zero":
                    v = torch.ones_like(v, dtype=torch.bool)
                t = torch.zeros(v.shape, dtype=torch.float32, device=v.device).masked_fill(~v.bool(), float("-inf"))[:, None]
            else:
                v = control_graph(v.float(), s.control, rc, self.site)
                m = self.f[_key(s.name)]
                w = m.w * (self._head_mask(s, m.w.device, m.w.dtype) if s.heads is not None else 1)
                if s.gate is not None:
                    w = w[None] * self._gate(m.gate_u, m.gate_b, rc, s)                  # [B,H]
                    t = v[:, None] * w[:, :, None, None]
                else:
                    t = v[:, None] * w[None, :, None, None]
                if s.confidence:
                    cq, ck = self._conf(rc, s)
                    t = t * (cq[:, :, None] * ck[:, None, :])[:, None]
            out = t if out is None else out + t
        return out

    def augment(self, rc: RelCtx, xq: torch.Tensor, xk: torch.Tensor | None = None):
        """Query-dependent q/k augmentation ([B,H,Q,A], [B,H,K,A]) of every `aug` factor, or (None, None)."""
        qs, ks = [], []
        for s in self.specs:
            d = get_factor(s.name)
            if d.form != "aug" or s.control == "off":
                continue
            m = self.f[_key(s.name)]
            fq, fk = OPS[d.op].features(d, s, m, rc, self.site, xq, xk)
            if s.control == "zero" and d.op != "bilinear":
                fk = torch.zeros_like(fk)
            if s.heads is not None:
                fq = fq * self._head_mask(s, fq.device, fq.dtype)[None, :, None, None]
            if s.gate is not None:
                fq = fq * self._gate(m.gate_u, m.gate_b, rc, s)[:, :, None, None]
            if s.confidence:
                cq, ck = self._conf(rc, s)
                fq, fk = fq * cq[:, None, :, None], fk * ck[:, None, :, None]
            qs.append(fq)
            ks.append(fk)
        if not qs:
            return None, None
        return torch.cat(qs, -1), torch.cat(ks, -1)

    @torch.no_grad()
    def contributions(self, rc: RelCtx, xq=None, xk=None) -> dict:
        """Per-factor logit terms [B,H,Q,K] (diagnostics / viz only; recomputes without the memo)."""
        out = {}
        for s in self.specs:
            d = get_factor(s.name)
            if s.control == "off":
                continue
            sub = FactorSite.__new__(FactorSite)
            nn.Module.__init__(sub)
            sub.heads, sub.site, sub.specs = self.heads, self.site, (s,)
            sub.edge_specs = (s,) if s in self.edge_specs else ()
            sub.w = self.w[[self.edge_specs.index(s)]] if s in self.edge_specs else None
            sub.f = self.f
            r = RelCtx(rc.sets, rc.edges, rc.task, rc.summaries, rc.deploy, None, rc.estimates, {})
            if d.form == "aug":
                if xq is None:
                    continue
                fq, fk = sub.augment(r, xq, xk)
                out[s.name] = fq @ fk.transpose(-1, -2)
            else:
                b = sub.bias(r)
                if b is not None:
                    out[s.name] = b
        return out


class FieldReadouts(nn.Module):
    """Linear Gaussian heads for factors whose source is `probe`: at `params.readout_layer` the token hiddens of the
    query set predict the field (mean, log-variance); the estimate replaces the field for later layers (docs 4)."""

    def __init__(self, dim: int, specs):
        super().__init__()
        from rrp.policies.relations.base import FIELDS
        self.items = []
        self.heads = nn.ModuleDict()
        for s in specs:
            d = get_factor(s.name)
            if s.control == "off" or effective_source(s) != "probe" or d.field.startswith("edges:") or d.op == "bilinear":
                continue
            fd = FIELDS[d.field]
            p = {**d.p, **s.p}
            self.items.append((s.name, d.field, int(p.get("readout_layer", 0)), p.get("readout_set")))
            self.heads[_key(s.name)] = _Coeff(dim, 1, 2 * fd.dim)

    def observe(self, layer: int, set_name: str, h: torch.Tensor, rc: RelCtx) -> None:
        for name, fld, L, only in self.items:
            if L == layer and (only is None or only == set_name):
                mu, lv = self.heads[_key(name)](h)[:, 0].chunk(2, -1)
                rc.estimates[(set_name, fld)] = (mu, lv.clamp(-8, 6).exp())
                rc.estimates[(set_name, fld, "logvar")] = lv
