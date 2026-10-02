"""RunConfig: the one run-configuration schema of the unified pipeline (W5; docs/architecture.md "Config schema").

A RunConfig names WHAT is run (family, stage, variant, seed, lineage), WHICH existing runs it consumes (inputs, by run
id, resolved to paths through a RunIndex), the meaning-changing FLAGS (all required, no silent defaults) and the
stage's native hyperparameters (`params`, passed through unchanged to the existing training/evaluation functions).

- `out` is DERIVED: artifacts/runs/{track}/{lineage}/{stage}[-{tag}]_s{seed}. (`tag` is an addition to the audit's
  template: one lineage has several refits / flows / DAgger rounds of the same stage, e.g. refit-bcdag1, refit-gendag3.)
- Flags: which flags apply depends on (family, stage) (FLAG_SPEC). An applicable flag must be stated (not None); a
  non-applicable one must be None. `to_native()` writes each flag at the key the existing code reads.
- Overlays (`overlay`, deep merge) and matrix expansion (`expand_matrix`, e.g. variant x seed) build configs from a base.
Run ids: "<store>/<name>" with store in {runs, packed, datasets, ...} (a path under artifacts/), optionally followed by
":<file>". Ids resolve to artifacts/<id> unless the RunIndex maps them (aliases such as "arm/semfix/s2/rep" for legacy
runs; artifacts/run_index.json). Stdlib + pydantic only (contracts layer).

D-145 P2: the hand-written per-run JSON configs (`configs/`, now archived) and their reader (`load_legacy` and friends)
are gone; a RunConfig is rendered from a recipe (`recipes/`, harness/dag.py). The `legacy` key stays in the serialised
form (always null) because it is part of every existing run's `config_hash` / `config.json`.
"""
from __future__ import annotations

import ast
import copy
from .compute import Compute
from .provenance import json_digest
import json
import operator
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator

SCHEMA_VERSION = "runconfig-1"
BUILTIN_FAMILIES = ("arm", "dual", "legged", "pointer", "psi0")   # pointer / psi0: no meaning-changing flags (D-145 P4c)
Variant = Literal["sem", "nosem", "semfix", "na"]
_CORE_STAGES = ("collect", "pack", "train_rep", "probes", "train_flow", "flow_ft", "dagger_collect", "refit",
                   "eval_r1", "eval_r2", "heldout", "edits", "train_bc", "validate_tracker", "train_tracker", "eval_tracker",
                   "grpo", "target_eval", "target_adapt")   # D-126 (arm): GRPO + anchors; sealed target-body eval / adaptation
# DERIVED, open (docs/architecture.md 14.2): the core stages above plus every stage a family declares through
# `declare_stage` (`harness.pipelines.base.register_stage` calls it). A live list: importers see additions.
PIPELINE_STAGES: list[str] = list(_CORE_STAGES)
# NOTE (D-144 sweep-flags, D-145 P2): no family's `train_rep` maps `probe_lv_min` any more: the probe weights / lv floor
# are a factor set (`params.latent.factors`, a `probe.arm.*` / `probe.legged.*` FactorSpec per query;
# `nets/semantic_latent.py::legacy_latent_factors`), and `_check_variant` reads them there. `probe_lv_min` stays in
# FLAG_NAMES / `Flags` (closed schema, part of every serialised config and config_hash) though no FLAG_SPEC uses it.
FLAG_NAMES = ("zero_prev_action", "realizer_anchor", "realizer_drop_qd", "probe_lv_min", "qd_dropout", "contact_version")
META = "@meta"          # flag recorded in the RunConfig/provenance only (no native key; e.g. contact_version)
CLI = "@cli"            # flag the stage turns into a command-line argument (e.g. ladder --prev-action zero|own)


def _flag_spec() -> dict[tuple[str, str], dict[str, str]]:
    arm_eval = {"zero_prev_action": CLI, "contact_version": META}
    arm = {
        "collect": {"contact_version": META}, "pack": {"contact_version": META},
        "train_rep": {"zero_prev_action": "zero_prev_action", "realizer_anchor": "realizer_anchor",
                      "contact_version": META},   # D-144 sweep-flags: probe_lv_min retired here, `latent.factors` instead
        "probes": {"contact_version": META},
        "train_flow": {"zero_prev_action": "zero_prev_action", "contact_version": META},
        "flow_ft": {"zero_prev_action": "zero_prev_action", "contact_version": META},
        "refit": {"zero_prev_action": "zero_prev_action", "realizer_anchor": "realizer_anchor",
                  "realizer_drop_qd": "realizer_drop_qd", "contact_version": META},
        "dagger_collect": arm_eval, "eval_r1": arm_eval, "eval_r2": arm_eval, "heldout": arm_eval, "edits": arm_eval,
    }
    dual = dict(arm)                  # (D-126: the dual table is the arm table BEFORE the arm-only stages below)
    arm.update({                      # D-126 arm-only stages
        "grpo": arm_eval, "target_eval": arm_eval, "target_adapt": {"contact_version": META},
        "train_bc": {"zero_prev_action": "zero_prev_action", "contact_version": META},
    })
    legged = {s: {"contact_version": META} for s in _CORE_STAGES}
    legged["train_rep"] = {"qd_dropout": "latent.qd_dropout", "contact_version": META}   # sweep-flags follow-up: probe_lv_min retired here too, `latent.factors` instead
    legged["refit"] = {"qd_dropout": "qd_dropout", "contact_version": META}
    spec = {}
    for fam, table in (("arm", arm), ("legged", legged), ("dual", dual)):
        for st in _CORE_STAGES:
            spec[(fam, st)] = dict(table.get(st, {"contact_version": META}))
    for fam in ("pointer", "psi0"):   # UI pointer / Psi0 humanoid VLA: no arm-physics flags apply to any stage
        for st in _CORE_STAGES:
            spec[(fam, st)] = {}
    return spec


FLAG_SPEC = _flag_spec()

# ------------------------------------------------------------------------------------ family registry (W11)
# External packages (e.g. psi1z) add a body family WITHOUT editing rrp: either call register_family(...) at import
# time, or declare an entry point in the group "rrp.families" whose target is a module (registration on import) or a
# zero-argument callable. Entry points are loaded lazily, the first time an unknown family name is looked up.
FAMILY_ENTRY_POINT_GROUP = "rrp.families"
_EXTERNAL_FAMILIES: dict[str, dict] = {}
_PLUGINS_LOADED = False


def register_family(name: str, *, flag_spec: dict[str, dict[str, str]] | None = None,
                    default_stage_flags: dict[str, str] | None = None,
                    doc: str = "") -> None:
    """Register an extension body family.

    flag_spec: {stage: {flag: where}} for the stages where meaning-changing flags apply (`where` is the native config
    path the stage reads, or META / CLI); every other stage gets `default_stage_flags` (default: no flag applies).
    Flags must be names from FLAG_NAMES (Flags is a closed schema). Re-registering the same spec is a no-op;
    a different spec for an existing name, or a built-in name, raises."""
    if name in BUILTIN_FAMILIES:
        raise RunConfigError(f"family {name!r} is built in")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise RunConfigError(f"bad family name {name!r} (lowercase identifier)")
    flag_spec = {k: dict(v) for k, v in (flag_spec or {}).items()}
    default_stage_flags = dict(default_stage_flags or {})
    for st, table in list(flag_spec.items()) + [("*", default_stage_flags)]:
        if st != "*" and st not in PIPELINE_STAGES:
            raise RunConfigError(f"{name}: unknown stage {st!r}")
        bad = [f for f in table if f not in FLAG_NAMES]
        if bad:
            raise RunConfigError(f"{name}/{st}: unknown flags {bad} (known: {FLAG_NAMES})")
    info = dict(flag_spec=flag_spec, default_stage_flags=default_stage_flags, doc=doc)
    if name in _EXTERNAL_FAMILIES:
        if _EXTERNAL_FAMILIES[name] != info:
            raise RunConfigError(f"family {name!r} is already registered with a different spec")
        return
    _EXTERNAL_FAMILIES[name] = info
    for st in _CORE_STAGES:
        FLAG_SPEC[(name, st)] = dict(flag_spec.get(st, default_stage_flags))


def unregister_family(name: str) -> None:
    """Remove an extension family (tests)."""
    if _EXTERNAL_FAMILIES.pop(name, None) is None:
        return
    for k in [k for k in FLAG_SPEC if k[0] == name]:
        del FLAG_SPEC[k]
    _prune_stages()


_DECLARED: set[tuple[str, str]] = set()      # (family, stage) pairs added by declare_stage (not the core table)


def declare_stage(family: str, stage: str, flags=None) -> None:
    """Declare that `family` has a `stage` (the pair a RunConfig may name), adding the stage name to PIPELINE_STAGES.

    flags: None keeps an existing pair's flag table unchanged (a new pair gets no flags); an iterable of FLAG_NAMES
    (each recorded as META) or a {flag: where} dict states the table. A different table for an existing pair raises.
    The family must be built in or registered (`register_family`)."""
    if family not in BUILTIN_FAMILIES and family not in _EXTERNAL_FAMILIES:
        raise RunConfigError(f"declare_stage: unknown family {family!r} (register_family first)")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", stage):
        raise RunConfigError(f"bad stage name {stage!r} (lowercase identifier)")
    table = None if flags is None else (dict(flags) if isinstance(flags, dict) else {f: META for f in flags})
    bad = [f for f in (table or {}) if f not in FLAG_NAMES]
    if bad:
        raise RunConfigError(f"{family}/{stage}: unknown flags {bad} (known: {FLAG_NAMES})")
    have = FLAG_SPEC.get((family, stage))
    if have is not None:
        if table is not None and table != have:
            raise RunConfigError(f"{family}/{stage} is already declared with flags {have}, not {table}")
        return
    FLAG_SPEC[(family, stage)] = table or {}
    _DECLARED.add((family, stage))
    if stage not in PIPELINE_STAGES:
        PIPELINE_STAGES.append(stage)


def undeclare_stage(family: str, stage: str) -> None:
    """Remove a stage declared by `declare_stage` (tests); core pairs are never removed."""
    if (family, stage) in _DECLARED:
        _DECLARED.discard((family, stage))
        FLAG_SPEC.pop((family, stage), None)
        _prune_stages()


def _prune_stages() -> None:
    """Drop non-core stage names no (family, stage) pair uses any more."""
    used = {st for (_, st) in FLAG_SPEC}
    PIPELINE_STAGES[:] = [st for st in PIPELINE_STAGES if st in _CORE_STAGES or st in used]


def ensure_stage(name: str) -> str:
    if name not in PIPELINE_STAGES:
        raise ValueError(f"unknown stage {name!r} (known: {tuple(PIPELINE_STAGES)}; families declare stages with "
                         "harness.pipelines.base.register_stage)")
    return name


def load_family_plugins(force: bool = False) -> list[str]:
    """Import every entry point of the group "rrp.families" (once). Returns the entry point names loaded.
    A broken plugin raises (no silent skip)."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED and not force:
        return []
    _PLUGINS_LOADED = True
    from importlib.metadata import entry_points
    names = []
    for ep in entry_points(group=FAMILY_ENTRY_POINT_GROUP):
        try:
            obj = ep.load()
        except Exception as e:
            raise RunConfigError(f"rrp.families entry point {ep.name!r} ({ep.value}) failed to load: {e}") from e
        if callable(obj):
            obj()
        names.append(ep.name)
    return names


def families() -> tuple[str, ...]:
    """Built-in + registered families (entry points are NOT loaded here; see ensure_family)."""
    return BUILTIN_FAMILIES + tuple(_EXTERNAL_FAMILIES)


def ensure_family(name: str) -> str:
    """Validate a family name, loading entry-point plugins once if it is not known yet."""
    if name in BUILTIN_FAMILIES or name in _EXTERNAL_FAMILIES:
        return name
    load_family_plugins()
    if name not in _EXTERNAL_FAMILIES:
        raise ValueError(f"unknown family {name!r} (known: {families()}; extensions register via "
                         f"rrp.core.runconfig.register_family or the {FAMILY_ENTRY_POINT_GROUP!r} entry points)")
    return name


Family = Annotated[str, AfterValidator(ensure_family)]
Stage = Annotated[str, AfterValidator(ensure_stage)]


class RunConfigError(ValueError):
    pass


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Flags(Strict):
    """Meaning-changing switches. Every field is REQUIRED (no default): None means "not applicable to this
    (family, stage)", and the RunConfig validator enforces that against FLAG_SPEC."""
    zero_prev_action: bool | None
    realizer_anchor: bool | None
    realizer_drop_qd: bool | None
    probe_lv_min: float | None
    qd_dropout: float | None
    contact_version: str | None


class ProductRef(Strict):
    """runs x files, in that order (e.g. 13 DAgger buffers of 3 rounds)."""
    runs: list[str]
    files: list[str]

    def refs(self) -> list[str]:
        return [f"{r}:{f}" for r in self.runs for f in self.files]


InputValue = Union[str, list[str], ProductRef]


class RunConfig(Strict):
    schema_version: Literal["runconfig-1"]
    family: Family
    stage: Stage
    variant: Variant
    seed: int
    lineage: str
    track: str = "pipeline"
    tag: str | None = None
    inputs: dict[str, InputValue] = Field(default_factory=dict)
    flags: Flags
    params: dict[str, Any] = Field(default_factory=dict)     # native config, passed through to the existing code
    options: dict[str, Any] = Field(default_factory=dict)    # pipeline-wrapper arguments (robots, seed sets, workers)
    note: str = ""
    legacy: None = None      # frozen key: always null; kept so config_hash / config.json of every existing run stay valid
    compute: Compute | None = None   # precision / TF32 / compile / CUDA graphs / seeds_per_job / eval_backend (core/compute.py).
    #   None = today's behaviour and is serialised as ABSENT (every existing hash is unchanged); a non-default block hashes in.

    @field_validator("compute")
    @classmethod
    def _compute_default_is_absent(cls, v):
        return None if v is not None and v.is_default else v        # an all-default block IS the absent block

    @model_serializer(mode="wrap")
    def _omit_absent_compute(self, handler):
        d = handler(self)
        if d.get("compute") is None:
            d.pop("compute", None)
        return d

    @model_validator(mode="after")
    def _check(self):
        spec = FLAG_SPEC.get((self.family, self.stage))
        if spec is None:
            raise RunConfigError(f"{self.family} has no stage {self.stage!r} (declared: "
                                 f"{sorted(st for (f, st) in FLAG_SPEC if f == self.family)})")
        for name in FLAG_NAMES:
            v = getattr(self.flags, name)
            if name in spec and v is None:
                raise RunConfigError(f"{self.family}/{self.stage}: flag {name!r} applies and must be stated "
                                     "explicitly (no silent default)")
            if name not in spec and v is not None:
                raise RunConfigError(f"{self.family}/{self.stage}: flag {name!r} does not apply here; set it to None")
        for k in self.inputs:
            if k in self.params:
                raise RunConfigError(f"{k!r} is both an input and a param")
        for name, where in spec.items():
            if where not in (META, CLI) and _has_path(self.params, where):
                raise RunConfigError(f"flag {name!r} must be set in flags, not in params[{where!r}]")
        if "out_dir" in self.params:
            raise RunConfigError("out_dir is derived (artifacts/runs/<track>/<lineage>/<stage>[-tag]_s<seed>)")
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-]+)*", self.lineage) or not re.fullmatch(r"[A-Za-z0-9_.\-]+", self.track):
            raise RunConfigError(f"bad lineage/track {self.lineage!r}/{self.track!r}")
        _check_variant(self)
        return self

    # --------------------------------------------------------------------------------------------- derived
    @property
    def run_id(self) -> str:
        tag = f"-{self.tag}" if self.tag else ""
        return f"runs/{self.track}/{self.lineage}/{self.stage}{tag}_s{self.seed}"

    @property
    def out(self) -> str:
        """Output directory (relative to the repo root)."""
        return "artifacts/" + self.run_id

    def config_hash(self) -> str:
        d = self.model_dump(mode="json")
        d.pop("note", None)
        return json_digest(d)

    def to_native(self, index: "RunIndex | None" = None) -> dict:
        """The dict the stage function reads: params + resolved inputs + flags at their native keys + derived out_dir/name."""
        index = index or RunIndex()
        native = copy.deepcopy(self.params)
        for k, v in self.inputs.items():
            native[k] = index.resolve_value(v)
        for name, where in FLAG_SPEC[(self.family, self.stage)].items():
            if where in (META, CLI):
                continue
            _set_path(native, where, getattr(self.flags, name))
        native.setdefault("out_dir", self.out)
        native.setdefault("name", self.run_id.replace("runs/", "", 1).replace("/", "_"))
        return native

    def input_paths(self, index: "RunIndex | None" = None) -> dict[str, str | list[str]]:
        index = index or RunIndex()
        return {k: index.resolve_value(v) for k, v in self.inputs.items()}

    def flag_dict(self) -> dict:
        return {k: v for k, v in self.flags.model_dump().items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json", exclude_none=False), indent=1)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        return cls.model_validate(d)


_PROBE_PREFIXES = ("probe.arm.", "probe.legged.")   # D-144 sweep-flags follow-up: legged joins arm/dual here too


def _factors_probe_weight_lv(factors) -> tuple[float | None, float | None]:
    """D-144 R2 (+ sweep-flags follow-up, legged): `LatentConfig.factors`-style `params.latent.factors` (list of
    `probe.arm.*` / `probe.legged.*` FactorSpec-shaped dicts / names) reduced to the equivalent (weight, lv_min)
    pair `_check_variant` needs. Deliberately NOT `rrp.policies.relations.base.resolve` (a stdlib-only,
    no-validation reduction: this module stays "stdlib + pydantic only" per its own module docstring --
    `relations.ops` imports torch)."""
    ws, lvs = set(), set()
    for it in factors or ():
        if isinstance(it, str):
            name, weight, params = it, None, {}
        elif isinstance(it, dict):
            name, weight, params = it.get("name", ""), it.get("weight"), it.get("params") or {}
        else:
            continue
        if isinstance(name, str) and name.startswith(_PROBE_PREFIXES):
            ws.add(1.0 if weight is None else weight)
            lvs.add(params.get("lv_min", -8.0))
    if not ws:
        return None, None
    return (ws.pop() if len(ws) == 1 else sorted(ws)[-1]), (lvs.pop() if len(lvs) == 1 else sorted(lvs)[-1])


def _check_variant(rc: RunConfig) -> None:
    """The variant label must match the recipe it names (catches mislabelled lineages).

    `train_rep` configs are `factors:`-shaped for every family (`params.latent.factors`: a `probe.arm.*` /
    `probe.legged.*` FactorSpec per query; D-144 sweep-flags); the variant is read off their (weight, lv_min)."""
    if rc.variant == "na" or rc.stage not in ("train_rep", "train_flow", "flow_ft"):
        return
    p = rc.params
    if rc.family == "pointer":      # `rrp train pointer rep|flow --w-sem`: semfix = probe loss on z, nosem = both weights 0
        w = p.get("w_sem")
        if w is None:
            raise RunConfigError(f"pointer/{rc.stage}: params.w_sem must be explicit")
        if (w == 0) != (rc.variant == "nosem"):
            raise RunConfigError(f"variant {rc.variant!r} does not match params.w_sem={w} of this pointer {rc.stage} config")
        return
    if rc.stage == "train_rep":
        lat = p.get("latent") or {}
        w, lv = _factors_probe_weight_lv(lat.get("factors"))
        if w is None:
            raise RunConfigError("train_rep: params.latent.factors must include an explicit probe.arm.*/"
                                 "probe.legged.* weight")
        ok = {"nosem": w == 0, "sem": w > 0 and lv is not None and lv <= -8.0,
              "semfix": w > 0 and lv is not None and lv > -8.0}[rc.variant]
    else:
        # sweep-flags follow-up: `packet_semantic_weight` may also arrive as a `flow.packet_semantic` factor spec
        # (`nets/semantic_latent.py::packet_semantic_factor`) -- forward-compatible; every config actually on disk
        # today (arm and legged alike) still renders the flat key, which stays the primary read.
        w = p.get("packet_semantic_weight")
        if w is None and p.get("factors"):
            w = next((it.get("weight", 0.0) for it in p["factors"]
                     if isinstance(it, dict) and it.get("name") == "flow.packet_semantic"), None)
        if w is None:
            raise RunConfigError(f"{rc.stage}: params.packet_semantic_weight (or a flow.packet_semantic weight "
                                 "in params.factors) must be explicit")
        ok = (w == 0) == (rc.variant == "nosem")
    if not ok:
        raise RunConfigError(f"variant {rc.variant!r} does not match the recipe of this {rc.stage} config")


# ------------------------------------------------------------------------------------------------ dotted paths
def _has_path(d: dict, path: str) -> bool:
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _set_path(d: dict, path: str, value) -> None:
    parts = path.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


# ---------------------------------------------------------------------------------------------------- run index
class RunIndex:
    """Run id -> path. Default: artifacts/<id>. `aliases` (artifacts/run_index.json) name existing runs canonically,
    e.g. "arm/semfix/s2/rep" -> "artifacts/runs/ladder_latent_semfix_b1fix_anchor_s2"."""

    def __init__(self, aliases: dict[str, str] | None = None, root: Path | None = None):
        self.aliases = dict(aliases or {})
        self.root = root

    @classmethod
    def load(cls, path: Path | str = "artifacts/run_index.json", root: Path | None = None) -> "RunIndex":
        p = Path(path)
        if not p.is_absolute():            # relative to root, else to the rrp checkout / $RRP_HOME (not the cwd)
            from rrp.core.paths import rrp_home
            p = (root if root is not None else rrp_home()) / p
        d = json.loads(p.read_text()) if p.exists() else {}
        return cls(d.get("aliases", {}), root)

    def resolve(self, ref: str) -> str:
        rid, _, file = ref.partition(":")
        if rid in self.aliases:
            base = self.aliases[rid]
        elif re.fullmatch(r"[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-]+)+", rid):
            base = "artifacts/" + rid
        else:
            raise RunConfigError(f"unknown run id {rid!r} (not an alias, not <store>/<name>)")
        return f"{base}/{file}" if file else base

    def resolve_value(self, v: InputValue) -> str | list[str]:
        if isinstance(v, ProductRef):
            return [self.resolve(r) for r in v.refs()]
        if isinstance(v, dict):
            return [self.resolve(r) for r in ProductRef.model_validate(v).refs()]
        if isinstance(v, list):
            return [self.resolve(r) for r in v]
        return self.resolve(v)

    def absolute(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() or self.root is None else self.root / p


# --------------------------------------------------------------------------------------------- overlays, matrix
def overlay(base: dict, over: dict) -> dict:
    """Deep merge: dicts merge recursively, everything else (lists included) replaces. A key "a.b.c" in `over`
    addresses a nested key; a value None deletes the key."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if "." in k and k not in out:
            head, rest = k.split(".", 1)
            out[head] = overlay(out.get(head) or {}, {rest: v})
            continue
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = overlay(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod}


def _eval(expr: str, env: dict):
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, str)):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise RunConfigError(f"unknown template variable {n.id!r} in {{{expr}}}")
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise RunConfigError(f"unsupported template expression {{{expr}}}")
    return ev(ast.parse(expr, mode="eval"))


_TPL = re.compile(r"\{([^{}]+)\}")


def render(obj, env: dict):
    """Substitute {expr} templates (names from env, + - * // % on numbers). A string that is exactly one template
    keeps the value's type (so "{1701 + 1000*(seed-1)}" becomes an int)."""
    if isinstance(obj, str):
        m = _TPL.fullmatch(obj)
        if m:
            return _eval(m.group(1), env)
        return _TPL.sub(lambda mm: str(_eval(mm.group(1), env)), obj)
    if isinstance(obj, list):
        return [render(x, env) for x in obj]
    if isinstance(obj, dict):
        return {render(k, env) if isinstance(k, str) else k: render(v, env) for k, v in obj.items()}
    return obj


def expand_matrix(base: dict, matrix: dict[str, list], per_value: dict[str, dict[Any, dict]] | None = None,
                  env: dict | None = None) -> list[dict]:
    """Cartesian product over `matrix` (e.g. {"variant": [...], "seed": [...]}): each point sets those top-level
    fields, applies per_value[axis][value] overlays (e.g. the variant recipe), then renders templates with the point
    (plus `env`). Order: first axis outermost."""
    points = [{}]
    for axis, values in matrix.items():
        points = [dict(p, **{axis: v}) for p in points for v in values]
    out = []
    for p in points:
        d = overlay(base, {k: v for k, v in p.items()})
        for axis, v in p.items():
            ov = ((per_value or {}).get(axis) or {}).get(v)
            if ov:
                d = overlay(d, ov)
        out.append(render(d, {**(env or {}), **p}))
    return out
