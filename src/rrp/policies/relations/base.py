"""Relation-factor registry core (D-144; design: docs/relations.md sections 1-3, 7, 11).

Token model: `TokenSet` (mask, kind, provenance-tagged `fields`, privileged `labels`), `EdgeSet` (typed / soft relations
between two token sets over a named vocabulary), `RelCtx` (everything one forward needs). Entries: `FactorDef`
(field x operator x form x algebra x conditioning x source + label / data generators), registered in `FACTORS`
by `rrp.policies.relations.catalog`. Run configs hold a list of `FactorSpec`s; `resolve` expands presets / globs /
overrides, `compat_hash` enters checkpoints, `provenance` goes into run records, `assert_deployable` is the deploy guard.
Runtime contract (D-146, unit F1, docs section 11): `FAMILIES` declare what each net family's token sets / sites / labels
provide and `resolve(family=..., env_caps=..., training=...)` refuses a spec the family cannot run (no silent skips);
`estimates_loss` supervises every estimate a forward wrote; `stamp_versions` / `require_factors` guard checkpoints.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal, Sequence

Prov = Literal["public", "estimated", "privileged"]
FORMS = ("bias", "aug", "gate", "mask", "message", "embed", "readout")
CONTROLS = ("on", "off", "zero", "shuffled", "rewired", "reversed", "gt", "estimated", "serialized")
GATES = ("task", "instruction", "goal", "embodiment", "history")
TOKEN_KINDS = ("pad", "morph_node", "passive_joint", "assembly", "entity", "widget", "image_patch", "task_event",
               "task_role", "predicate", "receipt", "sensor", "action_node", "knot", "command_dim", "text")


class PrivilegedInput(RuntimeError):
    """A deployable forward tried to read privileged (supervision-only) values."""


class FactorError(ValueError):
    """Unknown / planned / invalid factor spec."""


# ------------------------------------------------------------------ token model
@dataclass(frozen=True)
class FieldDef:
    name: str
    dim: int
    kind: Literal["position", "frame", "orientation", "direction", "shape", "scalar", "vector", "id", "membership",
                  "graph", "time", "belief", "symbol"]
    prov: Prov = "public"
    frame: str = ""
    units: str = ""
    label: str | None = None


@dataclass
class TokenSet:
    name: str
    mask: Any                                   # [B,T] bool
    kind: Any = None                            # [B,T] long -> TOKEN_KINDS (or a family-local kind id)
    fields: dict = field(default_factory=dict)  # name -> [B,T,dim]; "<name>.valid" [B,T]; "<name>.var" [B,T,dim]
    labels: dict = field(default_factory=dict)  # privileged targets (supervision only)
    deploy: bool = False

    def field(self, name: str):
        if name not in self.fields:
            raise KeyError(f"token set {self.name!r} has no field {name!r}; fields: {sorted(self.fields)}")
        return self.fields[name]

    def label(self, name: str):
        if self.deploy:
            raise PrivilegedInput(f"label {name!r} of token set {self.name!r} read in deploy mode")
        if name not in self.labels:
            raise KeyError(f"token set {self.name!r} has no label {name!r}; labels: {sorted(self.labels)}")
        return self.labels[name]


@dataclass
class EdgeSet:
    vocab: tuple                                # channel names in the featurizer's on-disk order
    data: Any                                   # [B,Q,K,R] bool (given) or float in [0,1] (estimated / soft)
    prov: Prov = "public"

    def channel(self, name: str) -> int:
        try:
            return self.vocab.index(name)
        except ValueError:
            raise FactorError(f"edge {name!r} not in vocabulary {self.vocab}") from None


@dataclass
class RelCtx:
    """Per-forward relational context. Sites are named "q>k" by token-set names; `edges[site]` is that site's EdgeSet
    (`edges[site + "@gt"]` a privileged one). `generator` drives the controls in site order; `memo` caches per-site
    control-transformed edges so every layer of a site sees the same draw (one draw per site and forward)."""
    sets: dict
    edges: dict = field(default_factory=dict)
    task: Any = None
    summaries: dict = field(default_factory=dict)   # gate name -> [B,D] conditioning vector ("task", "goal", ...)
    deploy: bool = False
    generator: Any = None
    estimates: dict = field(default_factory=dict)   # (set, field) -> (value, var) produced by factor readouts
    memo: dict = field(default_factory=dict)

    def token_sets(self, site: str) -> tuple:
        q, k = site.split(">")
        return self.sets[q], self.sets[k]

    def edge_set(self, site: str, edge: str, source: str) -> "EdgeSet":
        """The EdgeSet at `site` whose vocabulary has `edge`, for a factor source. Keys: `site` (the net's primary
        vocabulary), `site#<vocab>` (further vocabularies, incl. estimates emitted by bilinear factors with
        `params.emits`), `+ "@gt"` (privileged). source "gt" -> privileged only (refused in deploy mode); "probe" ->
        estimated only; "given" -> any non-privileged."""
        if source == "gt" and self.deploy:
            raise PrivilegedInput(f"ground-truth edge {edge!r} at {site} in deploy mode")
        for key, es in self.edges.items():
            base = key[:-3] if key.endswith("@gt") else key
            if base != site and not base.startswith(site + "#"):
                continue
            if (source == "gt") != key.endswith("@gt") or edge not in es.vocab:
                continue
            if source == "probe" and es.prov != "estimated":
                continue
            return es
        raise FactorError(f"no {source} EdgeSet with edge {edge!r} at site {site} (keys: {sorted(self.edges)}); "
                          f"estimated edges come from a bilinear factor with params.emits at its readout layer")


# ------------------------------------------------------------------ entries
@dataclass(frozen=True)
class Algebra:
    arity: int = 2
    direction: Literal["directed", "symmetric", "antisymmetric"] = "directed"
    transitive: bool = False
    value: Literal["bool", "signed", "weighted", "prob", "vector"] = "bool"
    dynamic: bool = False
    range: Literal["local", "global"] = "global"


@dataclass(frozen=True)
class ReadoutDef:
    query: str
    address: Literal["entity", "entity×asm", "asm", "knot×asm", "knot×pair", "body", "token", "pair"]
    out: int
    loss: Literal["bce", "gauss", "ce", "cos", "mse", "soft_ce"]
    label: str = ""
    scale: float = 1.0
    lv_min: float = -8.0
    reads: Literal["packet", "hidden", "tokens"] = "packet"


@dataclass(frozen=True)
class FactorDef:
    name: str
    version: str
    field: str                                  # FIELDS key or "edges:<vocab>" / "edges:*" (any vocab)
    op: str                                     # OPS key (rrp.policies.relations.ops)
    form: str
    algebra: Algebra = Algebra()
    sources: tuple = ("given",)
    label: str | None = None
    gen: tuple = ()
    gates: tuple = ()
    readout: ReadoutDef | None = None
    params: tuple = ()                          # frozen ((key, value), ...); see .p
    status: Literal["implemented", "planned"] = "implemented"
    doc: str = ""

    @property
    def p(self) -> dict:
        return dict(self.params)


@dataclass(frozen=True)
class FactorSpec:
    name: str
    control: str = "on"
    source: str | None = None
    sites: tuple | None = None
    heads: tuple | None = None
    gate: str | None = None
    confidence: bool = False
    weight: float | None = None
    mix: float | None = None
    params: tuple = ()                          # frozen ((key, value), ...) overrides of FactorDef.params

    @property
    def p(self) -> dict:
        return dict(self.params)

    def to_dict(self) -> dict:
        d = {"name": self.name}
        for f in fields(self)[1:]:
            v = getattr(self, f.name)
            if v != f.default:
                d[f.name] = dict(v) if f.name == "params" else (list(v) if isinstance(v, tuple) else v)
        return d


def _freeze(v):
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, list):
        return tuple(_freeze(x) for x in v)
    return v


def spec(x) -> FactorSpec:
    """FactorSpec from a FactorSpec, a name / glob / "preset:<name>" string, or a JSON dict."""
    if isinstance(x, FactorSpec):
        return x
    if isinstance(x, str):
        return FactorSpec(name=x)
    if isinstance(x, dict):
        unknown = set(x) - {f.name for f in fields(FactorSpec)}
        if unknown:
            raise FactorError(f"unknown FactorSpec keys {sorted(unknown)} in {x}")
        d = dict(x)
        for k in ("sites", "heads"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        if "params" in d:
            d["params"] = _freeze(d["params"])
        return FactorSpec(**d)
    raise FactorError(f"cannot read a factor spec from {x!r}")


# ------------------------------------------------------------------ registries
FIELDS: dict[str, FieldDef] = {}
FACTORS: dict[str, FactorDef] = {}
PRESETS: dict[str, tuple] = {}


INERT_FORMS = ("message", "embed", "readout")     # forms a net family implements itself (no attention-logit term)


@dataclass(frozen=True)
class FamilyTokens:
    """FAMILIES[family]: the static declaration of what a net family's collate path and sites honour (docs section
    11). `resolve(family=...)` checks factor lists against it; the family's net derives its carries from it."""
    sets: dict            # token set -> public / estimated fields its collate path fills ("pos3d", "orient", ...)
    sites: dict           # site "q>k" -> carries ("edges:<vocab>", field names, "hidden")
    labels: dict          # token set -> label names the TRAINING collate can attach (relgen LABELS names)
    inert: tuple = ()     # globs of the message / embed / readout factors the family's net implements


FAMILIES: dict[str, FamilyTokens] = {}
# The data side (`harness.data.relgen`) hands over its own registries by reference (`register_data`), so policies never
# import harness (layering) and there is one copy of every label / part declaration. None until relgen is imported.
_DATA: dict[str, Any] = {"labels": None, "parts": None, "transforms": None}


def register_family(name: str, ft: FamilyTokens) -> FamilyTokens:
    FAMILIES[name] = ft
    return ft


def register_data(labels: dict, parts: dict, transforms: dict) -> None:
    _DATA["labels"], _DATA["parts"], _DATA["transforms"] = labels, parts, transforms


def register_field(f: FieldDef) -> FieldDef:
    FIELDS[f.name] = f
    return f


def register_factor(d: FactorDef) -> FactorDef:
    if d.name in FACTORS and FACTORS[d.name] != d:
        raise FactorError(f"factor {d.name!r} registered twice with different definitions")
    if d.form not in FORMS:
        raise FactorError(f"{d.name}: unknown form {d.form!r}")
    FACTORS[d.name] = d
    return d


def register_preset(name: str, items: Sequence) -> None:
    PRESETS[name] = tuple(items)


def _ensure_catalog():
    import rrp.policies.relations.catalog  # noqa: F401  (registers entries on import)


def get_factor(name: str) -> FactorDef:
    _ensure_catalog()
    if name not in FACTORS:
        raise FactorError(f"unknown factor {name!r}; `rrp factors list` shows the registry")
    return FACTORS[name]


# ------------------------------------------------------------------ resolution
def resolve(items: Sequence | None, default: str | None = None, *, family: str | None = None, env_caps=None,
            env: str | None = None, training: bool = False) -> tuple[FactorSpec, ...]:
    """Expand presets and globs; later items override the fields they set on earlier matching entries (a glob that
    matches nothing resolved yet adds every registered match). `None` -> the `default` preset. Validates every
    resolved spec against its entry. Order = first appearance (stable; it is the parameter-row order).
    Runtime contract (docs section 11; no silent skips): with `family`, every non-off spec must apply to >= 1 site of
    the family and be runnable there (a `given` field must be filled by the family's token sets, a `probe` / `gt`
    source needs a label the family can attach, an estimated graph needs its emitter); with `env_caps` and
    `training`, a supervising label's `needs` must be covered by the env's `StateView` caps, and `mix > 0` needs a
    scene part for `env` (relgen must be loaded: `harness.data.relgen.load_families()`)."""
    _ensure_catalog()
    if items is None:
        items = [f"preset:{default}"] if default else []
    out: dict[str, FactorSpec] = {}

    def apply(it):
        if isinstance(it, str) and it.startswith("preset:"):
            name = it[len("preset:"):]
            if name not in PRESETS:
                raise FactorError(f"unknown preset {name!r}; presets: {sorted(PRESETS)}")
            for x in PRESETS[name]:
                apply(x)
            return
        s = spec(it)
        given = set(it) - {"name"} if isinstance(it, dict) else \
            ({f.name for f in fields(FactorSpec)[1:] if getattr(s, f.name) != f.default} if not isinstance(it, str) else set())
        if any(ch in s.name for ch in "*?["):
            hits = [n for n in out if fnmatch.fnmatchcase(n, s.name)]
            if not hits:
                hits = sorted(n for n in FACTORS if fnmatch.fnmatchcase(n, s.name) and FACTORS[n].status == "implemented")
                if not hits:
                    raise FactorError(f"glob {s.name!r} matches no registered factor")
                for n in hits:
                    out[n] = replace(s, name=n)
                return
            for n in hits:
                out[n] = replace(out[n], **{k: getattr(s, k) for k in given})
            return
        get_factor(s.name)
        out[s.name] = replace(out[s.name], **{k: getattr(s, k) for k in given}) if s.name in out else s

    for it in items:
        apply(it)
    for s in out.values():
        validate(s)
    specs = tuple(out.values())
    if family is not None:
        _check_family(specs, family)
    if training and env_caps is not None:
        _check_env(specs, frozenset(env_caps), env)
    return specs


def _check_family(specs: Sequence[FactorSpec], family: str) -> None:
    from rrp.policies.relations.ops import applicable_sites
    if family not in FAMILIES:
        raise FactorError(f"unknown net family {family!r}; families: {sorted(FAMILIES)}")
    ft = FAMILIES[family]
    pool = {n for names in ft.labels.values() for n in names}
    for s in specs:
        if s.control == "off":
            continue
        d, src = get_factor(s.name), effective_source(s)
        if d.form in INERT_FORMS:
            if not any(fnmatch.fnmatchcase(s.name, g) for g in ft.inert):
                raise FactorError(f"{s.name}: family {family!r} implements no {d.form} factor of that name "
                                  f"(implements {list(ft.inert)})")
            continue
        sites = applicable_sites(d, s, specs, ft)
        if not sites:
            raise FactorError(f"{s.name}: applies to no site of family {family!r} (sites and carries: {ft.sites}); "
                              f"an estimated graph needs its bilinear emitter (params.emits) in the same list")
        lab = d.label or d.field
        if d.op == "bilinear":
            if d.label is None and src not in ("probe", "gt"):
                continue                     # a purely learned pair bias: trained by the flow loss alone
            if d.label not in pool:
                raise FactorError(f"{s.name}: needs label {d.label!r}, which family {family!r} cannot attach "
                                  f"(labels: {sorted(pool)})")
            continue
        if d.field.startswith("edges:"):
            continue
        for site in sites:
            for name in (site.split(">")[1:] if d.op == "unary" else site.split(">")):
                if src == "given" and d.field not in ft.sets.get(name, ()):
                    raise FactorError(f"{s.name}: field {d.field!r} is not filled on token set {name!r} of family "
                                      f"{family!r} (fills {list(ft.sets.get(name, ()))}); use source 'probe'")
                if src == "gt" and lab not in ft.labels.get(name, ()):
                    raise FactorError(f"{s.name}: source 'gt' needs label {lab!r} on {name!r}; family {family!r} "
                                      f"attaches {list(ft.labels.get(name, ()))}")
            if src == "probe" and (d.readout is None or lab not in ft.labels.get(site.split(">")[0], ())):
                raise FactorError(f"{s.name}: source 'probe' needs a readout and label {lab!r} on set "
                                  f"{site.split('>')[0]!r} of family {family!r} "
                                  f"(attaches {list(ft.labels.get(site.split('>')[0], ()))})")


def data_registries() -> tuple[dict, dict, dict]:
    labels, parts, transforms = _DATA["labels"], _DATA["parts"], _DATA["transforms"]
    if not labels or not parts or not transforms:
        raise FactorError("relgen label / part registries are empty: call "
                          "rrp.harness.data.relgen.load_families() before resolving with env_caps / training")
    return labels, parts, transforms


def _check_env(specs: Sequence[FactorSpec], caps: frozenset, env: str | None) -> None:
    labels, parts, transforms = data_registries()
    for s in specs:
        if s.control == "off":
            continue
        d = get_factor(s.name)
        if effective_source(s) in ("probe", "gt") and d.label in labels:
            miss = labels[d.label].needs - caps
            if miss:
                raise FactorError(f"{s.name}: label {d.label!r} needs StateView caps {sorted(miss)} the env lacks "
                                  f"(caps {sorted(caps)})")
        if s.mix is not None and s.mix > 0:
            if env is None:
                raise FactorError(f"{s.name}: mix {s.mix} > 0 needs resolve(env=...) to check its scene part")
            # `gen` names scene parts (env-bound) and / or generic transforms (run on any env's samples)
            unknown = [g for g in d.gen if g not in parts and g not in transforms]
            ok = [g for g in d.gen if g in transforms or (g in parts and (not parts[g].envs or env in parts[g].envs))]
            if unknown or not ok:
                raise FactorError(f"{s.name}: mix {s.mix} > 0 but nothing of {list(d.gen)} runs in env {env!r}"
                                  + (f" (unregistered: {unknown})" if unknown else ""))


def validate(s: FactorSpec) -> None:
    from rrp.policies.relations.ops import OPS
    d = get_factor(s.name)
    if d.status != "implemented":
        raise FactorError(f"factor {s.name!r} is a planned catalog entry (research/relations_catalog.md)")
    op = OPS[d.op]
    if s.control not in CONTROLS or s.control not in op.controls(d.form):
        raise FactorError(f"{s.name}: control {s.control!r} not supported by op {d.op!r} / form {d.form!r}")
    if s.source is not None and s.source not in d.sources:
        raise FactorError(f"{s.name}: source {s.source!r} not in {d.sources}")
    if s.control == "gt" and "gt" not in d.sources:
        raise FactorError(f"{s.name}: control 'gt' needs a gt source")
    if s.gate is not None and s.gate not in d.gates:
        raise FactorError(f"{s.name}: gate {s.gate!r} not allowed (allowed: {d.gates})")
    if d.form not in op.forms:
        raise FactorError(f"{s.name}: op {d.op!r} cannot realize form {d.form!r}")


def effective_source(s: FactorSpec) -> str:
    d = get_factor(s.name)
    if s.control == "gt":
        return "gt"
    if s.control == "estimated":
        return "probe"
    return s.source or d.sources[0]


def compat_hash(specs: Sequence[FactorSpec]) -> str:
    """Structure hash (parameter shapes / meaning): controls, weights and mix fractions are excluded."""
    rows = []
    for s in specs:
        d = get_factor(s.name)
        rows.append([s.name, d.version, d.op, d.form, s.source, list(s.sites or []), list(s.heads or []), s.gate,
                     s.confidence, sorted([k, repr(v)] for k, v in {**d.p, **s.p}.items())])
    return "fx-" + hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:12]


def provenance(specs: Sequence[FactorSpec]) -> list[dict]:
    out = []
    for s in specs:
        d = get_factor(s.name)
        out.append(dict(s.to_dict(), version=d.version, op=d.op, form=d.form, source=effective_source(s),
                        privileged=effective_source(s) == "gt"))
    return out


def assert_deployable(specs: Sequence[FactorSpec]) -> None:
    """Deploy guard: a deployable policy may not use ground-truth (privileged) sources."""
    bad = [s.name for s in specs if s.control != "off" and effective_source(s) == "gt"]
    if bad:
        raise PrivilegedInput(f"factors with privileged (gt) sources cannot deploy: {bad}")


def control_of(specs: Sequence[FactorSpec], name: str) -> str:
    """Control of a named factor ('off' when it is absent from the resolved list)."""
    for s in specs:
        if s.name == name:
            return s.control
    return "off"


def to_json(specs: Sequence[FactorSpec]) -> list[dict]:
    return [s.to_dict() for s in specs]


# ------------------------------------------------------------------ supervision of the estimates a forward wrote
def gaussian_nll(pred, target, mask, lv_min: float = -8.0, lv_max: float = 6.0):
    """Gaussian NLL of `target` [..., d] under pred[..., :d] (mean) / pred[..., d:2d] (log-variance, clamped to
    [lv_min, lv_max]); mean over the True entries of `mask` (the canonical probe / estimate loss)."""
    d = target.shape[-1]
    mu, lv = pred[..., :d], pred[..., d:2 * d].clamp(lv_min, lv_max)
    nll = 0.5 * (((target - mu) ** 2) / lv.exp() + lv + math.log(2 * math.pi)).sum(-1)
    m = mask.float()
    return (nll * m).sum() / m.sum().clamp(min=1)


def pair_labels() -> frozenset:
    """Names of the labels that are pair targets ([B,Q,K]): the labels of the registered bilinear factors."""
    _ensure_catalog()
    return frozenset(d.label for d in FACTORS.values() if d.op == "bilinear" and d.label)


def _label_valid(ts: TokenSet, name: str, shape) -> Any:
    import torch
    v = ts.labels.get(name + ".valid")
    if v is None:
        return torch.ones(shape, dtype=torch.bool, device=ts.mask.device)
    while v.dim() < len(shape):
        v = v[..., None]
    return v.bool().expand(shape)


def _pair_loss(logit, y, valid, kmask, qmask, kind: str):
    """(loss, acc_sum, n) of one pair estimate. bce: elementwise over valid pairs of two valid tokens. soft_ce: per
    query row a softmax over the valid keys against the row's target distribution (rows with no mass do not count)."""
    import torch.nn.functional as F
    pm = valid & qmask[:, :, None] & kmask[:, None, :]
    if kind == "soft_ce":
        lg = logit.masked_fill(~kmask[:, None, :], float("-inf"))
        logp = F.log_softmax(lg, -1).masked_fill(~kmask[:, None, :], 0.0)
        rows = pm.any(-1) & (y.sum(-1) > 0)
        v = (-(y * logp).sum(-1) * rows).sum() / rows.sum().clamp(min=1)
        hit = ((lg.argmax(-1) == y.argmax(-1)) & rows).sum()
        return v, float(hit), int(rows.sum())
    v = (F.binary_cross_entropy_with_logits(logit, y.to(logit.dtype), reduction="none") * pm).sum() / pm.sum().clamp(min=1)
    hit = (((logit > 0) == (y > 0.5)) & pm).sum()
    return v, float(hit), int(pm.sum())


def estimates_loss(rc: RelCtx, specs: Sequence[FactorSpec]):
    """Supervision of every estimate written during the forward (docs section 11), weighted by `FactorSpec.weight`
    (default 1). Returns (loss, logs, metrics): `logs["probe_<factor>"]` floats, `metrics["<factor>_acc" | "_mae"]`
    (sum, count) pairs (aggregate by summing both sides).
      rc.estimates[("pair", f)]          logits [B,Q,K] -> bce (soft_ce when ReadoutDef.loss says so) vs the pair label
                                         of the query set (arity 2: [B,Q,K]; arity 1 with soft_ce: [B,K,1] indicator
                                         -> the same key distribution for every `sensor` query row), masked by
                                         `<label>.valid` and both token masks
      rc.estimates[(set, field)]         (mu, var) + logvar -> Gaussian NLL vs `TokenSet.label(<label or field>)`
    A factor whose label is absent from the batch is masked (mix rows without it): no term, count 0, never an error.
    Refused in deploy mode (a loss needs privileged labels)."""
    import torch
    if rc.deploy:
        raise PrivilegedInput("estimates_loss reads privileged labels: not available in a deployable forward")
    dev = next(iter(rc.sets.values())).mask.device
    loss, logs, metrics = torch.zeros((), device=dev), {}, {}
    est_by = _estimated_by(rc)
    for s in specs:
        if s.control == "off":
            continue
        d = get_factor(s.name)
        w = 1.0 if s.weight is None else s.weight
        lab = d.label or d.field
        p = {**d.p, **s.p}
        terms = []
        if ("pair", s.name) in rc.estimates:
            site = rc.memo[("pair_site", s.name)]
            qs, ks = rc.token_sets(site)
            logit, kind = rc.estimates[("pair", s.name)], (d.readout.loss if d.readout else "bce")
            if d.label not in qs.labels:
                metrics[f"{s.name}_acc"] = (0.0, 0)
                continue
            y = qs.labels[d.label]
            if y.dim() == 3 and y.shape[-1] == 1 and kind == "soft_ce":
                # arity-1 indicator over tokens ([B,K,1], e.g. next_contact) -> one key distribution shared by every
                # `sensor` query row (the candidate manipulators)
                if qs.kind is None:
                    raise FactorError(f"{s.name}: an arity-1 label needs the query set's token kinds")
                ind = y[..., 0] * _label_valid(qs, d.label, y.shape[:2]) * ks.mask
                y = (ind / ind.sum(-1, keepdim=True).clamp(min=1e-9))[:, None, :] * \
                    (qs.kind == TOKEN_KINDS.index("sensor"))[:, :, None]
                valid = torch.ones_like(y, dtype=torch.bool)
            else:
                y = y[..., 0] if y.dim() == 4 else y
                valid = _label_valid(qs, d.label, y.shape)
            if y.shape != logit.shape:
                raise FactorError(f"{s.name}: label {d.label!r} shape {tuple(y.shape)} != pair logits "
                                  f"{tuple(logit.shape)} at {site}")
            v, hit, n = _pair_loss(logit, y, valid, ks.mask, qs.mask, kind)
            metrics[f"{s.name}_acc"] = (hit, n)
            terms.append(v)
        for (set_name, fld), name in est_by.items():
            if name != s.name:
                continue
            ts = rc.sets[set_name]
            if lab not in ts.labels:
                metrics[f"{s.name}_mae"] = (0.0, 0)
                continue
            y, (mu, _) = ts.labels[lab], rc.estimates[(set_name, fld)]
            m = _label_valid(ts, lab, ts.mask.shape) & ts.mask
            terms.append(gaussian_nll(torch.cat([mu, rc.estimates[(set_name, fld, "logvar")]], -1), y.to(mu.dtype), m,
                                      p.get("lv_min", -8.0)))
            metrics[f"{s.name}_mae"] = (float(((mu.detach() - y).abs() * m[..., None]).sum()), int(m.sum()) * y.shape[-1])
        if terms:
            v = sum(terms)
            loss = loss + w * v
            logs[f"probe_{s.name}"] = float(v.detach())
    return loss, logs, metrics


def _estimated_by(rc: RelCtx) -> dict:
    """(set, field) -> name of the factor whose FieldReadouts head wrote that estimate (`FieldReadouts.observe`)."""
    return {k[1:]: v for k, v in rc.memo.items() if isinstance(k, tuple) and k and k[0] == "est_by"}


# ------------------------------------------------------------------ checkpoints
def stamp_versions(versions: dict | None, specs: Sequence[FactorSpec] | None, ambient: Sequence[str] = ()) -> dict:
    """versions["factors"] = the factor structure hash, "+"-joined with the `ambient` version tags the caller folds in
    (`nets.checkpoint` passes the `feat.base_axes` version when resolved; relations cannot import features).
    Every checkpoint writer stamps through this; `specs=None` = a model with no relation factors."""
    bits = ([compat_hash(specs)] if specs is not None else []) + list(ambient)
    out = dict(versions or {})
    if bits:
        out["factors"] = "+".join(bits)
    return out


def require_factors(saved_versions: dict | None, specs: Sequence[FactorSpec], allow_mismatch: bool = False) -> None:
    """Loader guard: the saved structure hash must equal the requested specs' (`allow_mismatch` = deliberately
    loading into a different factor list, e.g. a controlled ablation). A checkpoint without a hash is refused too."""
    saved = (saved_versions or {}).get("factors")
    want = compat_hash(specs)
    have = saved.split("+")[0] if isinstance(saved, str) else None
    if have != want and not allow_mismatch:
        raise FactorError(f"checkpoint factor structure {have!r} != requested {want!r} "
                          f"(saved versions['factors'] = {saved!r}); pass allow_mismatch to load anyway")


# ------------------------------------------------------------------ CLI (`rrp factors list|show <name>`)
def cli(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="rrp factors", description="relation-factor registry (docs/relations.md)")
    ap.add_argument("what", choices=["list", "show", "presets"])
    ap.add_argument("name", nargs="?")
    a = ap.parse_args(argv)
    _ensure_catalog()
    if a.what == "list":
        for n, d in sorted(FACTORS.items()):
            if a.name is None or fnmatch.fnmatchcase(n, a.name):
                print(f"{n:32s} v{d.version:4s} {d.status:11s} {d.op:12s} {d.form:8s} {d.field:22s} {','.join(d.sources)}")
    elif a.what == "presets":
        for n, items in sorted(PRESETS.items()):
            print(f"{n}: {[i if isinstance(i, str) else spec(i).to_dict() for i in items]}")
    else:
        d = get_factor(a.name)
        print(json.dumps(asdict(d), indent=1, default=str))
    return 0
