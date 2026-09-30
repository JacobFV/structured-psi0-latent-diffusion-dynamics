"""Relation-factor registry core (D-144; design: docs/relations.md sections 1-3, 7).

Token model: `TokenSet` (mask, kind, provenance-tagged `fields`, privileged `labels`), `EdgeSet` (typed / soft relations
between two token sets over a named vocabulary), `RelCtx` (everything one forward needs). Entries: `FactorDef`
(field x operator x form x algebra x conditioning x source + label / data generators), registered in `FACTORS`
by `rrp.policies.relations.catalog`. Run configs hold a list of `FactorSpec`s; `resolve` expands presets / globs /
overrides, `compat_hash` enters checkpoints, `provenance` goes into run records, `assert_deployable` is the deploy guard.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
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
def resolve(items: Sequence | None, default: str | None = None) -> tuple[FactorSpec, ...]:
    """Expand presets and globs; later items override the fields they set on earlier matching entries (a glob that
    matches nothing resolved yet adds every registered match). `None` -> the `default` preset. Validates every
    resolved spec against its entry. Order = first appearance (stable; it is the parameter-row order)."""
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
    return tuple(out.values())


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
