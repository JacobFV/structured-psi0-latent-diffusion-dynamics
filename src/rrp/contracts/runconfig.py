"""RunConfig: the one run-configuration schema of the unified pipeline (W5; docs/repo_structure_audit.md "Config schema").

A RunConfig names WHAT is run (family, stage, variant, seed, lineage), WHICH existing runs it consumes (inputs, by run
id, resolved to paths through a RunIndex), the meaning-changing FLAGS (all required, no silent defaults) and the
stage's native hyperparameters (`params`, passed through unchanged to the existing training/evaluation functions).

- `out` is DERIVED: artifacts/runs/{track}/{lineage}/{stage}[-{tag}]_s{seed}. (`tag` is an addition to the audit's
  template: one lineage has several refits / flows / DAgger rounds of the same stage, e.g. refit-bcdag1, refit-gendag3.)
- Flags: which flags apply depends on (family, stage) (FLAG_SPEC). An applicable flag must be stated (not None); a
  non-applicable one must be None. `to_native()` writes each flag at the key the existing code reads.
- Overlays (`overlay`, deep merge) and matrix expansion (`expand_matrix`, e.g. variant x seed) build configs from a base.
- Legacy: `load_legacy(path)` reads every JSON config under configs/ into a RunConfig WITHOUT changing its meaning:
  `to_native()` returns exactly the original dict (tests/unit/test_runconfig.py round-trips all of configs/). A flag the
  legacy file omits gets the value the code has always defaulted to (LEGACY_FLAG_DEFAULTS) and is listed in
  `legacy.absent_flags`, so it is not written back (the old code keeps applying the same default).

Run ids: "<store>/<name>" with store in {runs, packed, datasets, ...} (a path under artifacts/), optionally followed by
":<file>". Ids resolve to artifacts/<id> unless the RunIndex maps them (aliases such as "arm/semfix/s2/rep" for legacy
runs; configs/run_index.json). Stdlib + pydantic only (contracts layer).
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import operator
import re
from pathlib import Path
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "runconfig-1"
Family = Literal["arm", "dual", "legged"]
Variant = Literal["sem", "nosem", "semfix", "na"]
PIPELINE_STAGES = ("collect", "pack", "train_rep", "probes", "train_flow", "flow_ft", "dagger_collect", "refit",
                   "eval_r1", "eval_r2", "heldout", "edits")
# kinds of legacy configs that are not pipeline stages (read for provenance; the pipeline does not run them)
LEGACY_ONLY_STAGES = ("train_bc", "train_policy", "adapt", "vlm", "protocol")
Stage = Literal[PIPELINE_STAGES + LEGACY_ONLY_STAGES]  # type: ignore[valid-type]
FLAG_NAMES = ("zero_prev_action", "realizer_anchor", "realizer_drop_qd", "probe_lv_min", "qd_dropout", "contact_version")
META = "@meta"          # flag recorded in the RunConfig/provenance only (no native key; e.g. contact_version)
CLI = "@cli"            # flag the stage turns into a command-line argument (e.g. ladder --prev-action zero|own)
NON_RUN_CONFIGS = ("configs/resources.local.json", "configs/run_index.json")
DEFAULT_CONTACT_VERSION = "contact_v1"   # every config in configs/ predates contact v2 (W1)


def _flag_spec() -> dict[tuple[str, str], dict[str, str]]:
    arm_eval = {"zero_prev_action": CLI, "contact_version": META}
    arm = {
        "collect": {"contact_version": META}, "pack": {"contact_version": META},
        "train_rep": {"zero_prev_action": "zero_prev_action", "realizer_anchor": "realizer_anchor",
                      "probe_lv_min": "latent.probe_lv_min", "contact_version": META},
        "probes": {"contact_version": META},
        "train_flow": {"zero_prev_action": "zero_prev_action", "contact_version": META},
        "flow_ft": {"zero_prev_action": "zero_prev_action", "contact_version": META},
        "refit": {"zero_prev_action": "zero_prev_action", "realizer_anchor": "realizer_anchor",
                  "realizer_drop_qd": "realizer_drop_qd", "contact_version": META},
        "dagger_collect": arm_eval, "eval_r1": arm_eval, "eval_r2": arm_eval, "heldout": arm_eval, "edits": arm_eval,
    }
    legged = {s: {"contact_version": META} for s in PIPELINE_STAGES}
    legged["train_rep"] = {"probe_lv_min": "latent.probe_lv_min", "qd_dropout": "latent.qd_dropout", "contact_version": META}
    legged["refit"] = {"qd_dropout": "qd_dropout", "contact_version": META}
    dual = dict(arm)
    spec = {}
    for fam, table in (("arm", arm), ("legged", legged), ("dual", dual)):
        for st in PIPELINE_STAGES + LEGACY_ONLY_STAGES:
            spec[(fam, st)] = dict(table.get(st, {"contact_version": META}))
    return spec


FLAG_SPEC = _flag_spec()
# the value the existing code uses when a legacy config omits the key (where it is read: see the comments)
LEGACY_FLAG_DEFAULTS = {
    ("arm", "realizer_anchor"): False,        # training/latent_train.py cfg_json.get("realizer_anchor", False)
    ("arm", "realizer_drop_qd"): False,       # training/latent_train.py, controllers/bundles.py .get(..., False)
    ("arm", "probe_lv_min"): -8.0,            # models/semantic_latent.py LatentConfig.probe_lv_min
    ("dual", "realizer_anchor"): False, ("dual", "realizer_drop_qd"): False, ("dual", "probe_lv_min"): -8.0,
    ("legged", "probe_lv_min"): -8.0,         # training/legged_latent_train.py lc.get("probe_lv_min", -8.0)
    ("legged", "qd_dropout@train_rep"): 0.0,  # training/legged_latent_train.py lc.get("qd_dropout", 0.0)
    ("legged", "qd_dropout@refit"): 0.5,      # training/legged_dagger.py cfg.get("qd_dropout", 0.5)
}
# zero_prev_action is NOT defaulted here: a legacy config without it keeps it absent, and the W3 reader
# (contracts.provenance.resolve_zero_prev_action) warns and applies the legacy False, exactly as before.
INPUT_KEYS = ("representation", "packed_dir", "dataset", "init_from", "dagger", "gen_dagger", "gen_flow", "data",
              "init_realizer", "checkpoint", "policy_checkpoint", "feature_cache")


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


class LegacyInfo(Strict):
    path: str
    sha256: str
    absent_flags: list[str] = Field(default_factory=list)
    key_order: list[str] = Field(default_factory=list)


class RunConfig(Strict):
    schema_version: Literal["runconfig-1"]
    family: Family
    stage: Stage  # type: ignore[valid-type]
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
    legacy: LegacyInfo | None = None

    @model_validator(mode="after")
    def _check(self):
        spec = FLAG_SPEC[(self.family, self.stage)]
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
        if self.legacy is None:
            if "out_dir" in self.params:
                raise RunConfigError("out_dir is derived for new runs (only legacy configs keep a literal out_dir)")
            if not re.fullmatch(r"[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-]+)*", self.lineage) or not re.fullmatch(r"[A-Za-z0-9_.\-]+", self.track):
                raise RunConfigError(f"bad lineage/track {self.lineage!r}/{self.track!r}")
            _check_variant(self)
        return self

    # --------------------------------------------------------------------------------------------- derived
    @property
    def run_id(self) -> str:
        if self.legacy is not None and isinstance(self.params.get("out_dir"), str):
            return _id_of(self.params["out_dir"])[0]
        tag = f"-{self.tag}" if self.tag else ""
        return f"runs/{self.track}/{self.lineage}/{self.stage}{tag}_s{self.seed}"

    @property
    def out(self) -> str:
        """Output directory (relative to the repo root)."""
        if self.legacy is not None and isinstance(self.params.get("out_dir"), str):
            return self.params["out_dir"]
        return "artifacts/" + self.run_id

    def config_hash(self) -> str:
        d = self.model_dump(mode="json")
        d.pop("note", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def to_native(self, index: "RunIndex | None" = None) -> dict:
        """The dict the existing stage function reads (legacy: exactly the original file)."""
        index = index or RunIndex()
        native = copy.deepcopy(self.params)
        for k, v in self.inputs.items():
            native[k] = index.resolve_value(v)
        absent = set(self.legacy.absent_flags) if self.legacy else set()
        for name, where in FLAG_SPEC[(self.family, self.stage)].items():
            if where in (META, CLI) or name in absent:
                continue
            _set_path(native, where, getattr(self.flags, name))
        if self.legacy is None:
            native.setdefault("out_dir", self.out)
            native.setdefault("name", self.run_id.replace("runs/", "", 1).replace("/", "_"))
        if self.legacy is not None and self.legacy.key_order:
            order = {k: i for i, k in enumerate(self.legacy.key_order)}
            native = dict(sorted(native.items(), key=lambda kv: order.get(kv[0], len(order))))
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


def _check_variant(rc: RunConfig) -> None:
    """New configs: the variant label must match the recipe it names (catches mislabelled lineages)."""
    if rc.variant == "na" or rc.stage not in ("train_rep", "train_flow", "flow_ft"):
        return
    p = rc.params
    if rc.stage == "train_rep":
        w = (p.get("latent") or {}).get("semantic_weight")
        if w is None:
            raise RunConfigError("train_rep: params.latent.semantic_weight must be explicit")
        lv = rc.flags.probe_lv_min
        ok = {"nosem": w == 0, "sem": w > 0 and lv is not None and lv <= -8.0,
              "semfix": w > 0 and lv is not None and lv > -8.0}[rc.variant]
    else:
        w = p.get("packet_semantic_weight")
        if w is None:
            raise RunConfigError(f"{rc.stage}: params.packet_semantic_weight must be explicit")
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


def _get_path(d: dict, path: str):
    cur = d
    for part in path.split("."):
        cur = cur[part]
    return cur


def _set_path(d: dict, path: str, value) -> None:
    parts = path.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _pop_path(d: dict, path: str):
    parts = path.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur[part]
    return cur.pop(parts[-1])


# ---------------------------------------------------------------------------------------------------- run index
def _id_of(path: str) -> tuple[str, str | None]:
    """artifacts/<store>/<name>[/<rest>] -> ("<store>/<name>", "<rest>" | None)."""
    rel = path[len("artifacts/"):]
    parts = rel.split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise RunConfigError(f"not a run path: {path!r}")
    rest = "/".join(parts[2:])
    return f"{parts[0]}/{parts[1]}", (rest or None)


def path_to_ref(path: str) -> str:
    rid, rest = _id_of(path)
    return rid if rest is None else f"{rid}:{rest}"


class RunIndex:
    """Run id -> path. Default: artifacts/<id>. `aliases` (configs/run_index.json) name existing runs canonically,
    e.g. "arm/semfix/s2/rep" -> "artifacts/runs/ladder_latent_semfix_b1fix_anchor_s2"."""

    def __init__(self, aliases: dict[str, str] | None = None, root: Path | None = None):
        self.aliases = dict(aliases or {})
        self.root = root

    @classmethod
    def load(cls, path: Path | str = "configs/run_index.json", root: Path | None = None) -> "RunIndex":
        p = Path(path)
        if not p.is_absolute() and root is not None:
            p = root / p
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


# ---------------------------------------------------------------------------------------------------- legacy load
_SEMFIX = re.compile(r"(semfix|fixsem|sfjf|lv4)")
_NOSEM = re.compile(r"(nosem|nsjf)")
_SEM = re.compile(r"(sem|jointfix|sejf)")
_LINEAGE = re.compile(r"(sfjf2|nsjf2|sejf2|sfjf|nsjf|jointfix|bindv4\w*?(?=_|$)|b1fix)")
DUAL_DATA_TASKS = ("handover", "support_insert", "assign")


def classify_legacy(rel: str, cfg: dict) -> tuple[str, str]:
    """(family, stage) of a legacy config from its directory and file name."""
    parts = rel.split("/")
    d, stem = parts[1], Path(rel).stem
    text = f"{stem} {cfg.get('name', '')}"
    if d in ("legged_latent", "legged_fixsem", "t1_diag", "legged_dagger", "legged_bc"):
        fam = "legged"
        if d == "legged_bc":
            return fam, "train_bc"
        if d == "legged_dagger" or stem.startswith("rz"):
            return fam, "refit"
        return fam, ("train_rep" if stem.startswith("rep") else "train_flow")
    fam = "dual" if ("dualarm" in text or "multi_m" in cfg) else "arm"
    if d == "data":
        return ("dual" if cfg.get("task") in DUAL_DATA_TASKS else "arm"), "collect"
    if d == "adapt":
        return "arm", "adapt"
    if d == "vlm":
        return "arm", "vlm"
    if d == "eval":
        return "arm", "protocol"
    if d == "model":
        return "arm", "train_policy"
    if d in ("latent", "ladder"):
        if stem.startswith("pack"):
            return fam, "pack"
        if stem.startswith("rep"):
            return fam, "train_rep"
        if stem.startswith("rz"):
            return fam, "refit"
        if stem.startswith("flow"):
            return fam, ("flow_ft" if "init_from" in cfg else "train_flow")
    raise RunConfigError(f"cannot classify legacy config {rel}")


def _legacy_variant(stage: str, rel: str, cfg: dict) -> str:
    if stage in ("collect", "pack", "adapt", "vlm", "protocol", "train_policy", "train_bc"):
        return "na"
    rep = cfg.get("representation") if isinstance(cfg.get("representation"), str) else ""
    text = f"{Path(rel).stem} {cfg.get('name', '')} {rep}".lower()
    lat = cfg.get("latent") if isinstance(cfg.get("latent"), dict) else {}
    if _SEMFIX.search(text) or (lat.get("probe_lv_min", -8.0) > -8.0 and lat.get("semantic_weight", 0) > 0):
        return "semfix"
    if _NOSEM.search(text):
        return "nosem"
    if _SEM.search(text):
        return "sem"
    return "na"


def _compress(refs: list[str]) -> InputValue:
    runs, files = [], []
    for r in refs:
        rid, _, f = r.partition(":")
        if rid not in runs:
            runs.append(rid)
    first = runs[0] if runs else None
    files = [r.partition(":")[2] for r in refs if r.partition(":")[0] == first]
    if len(runs) > 1 and all(files) and ProductRef(runs=runs, files=files).refs() == refs:
        return ProductRef(runs=runs, files=files)
    return refs


def load_legacy(path: Path | str, root: Path | None = None) -> RunConfig:
    """Read a legacy JSON config (configs/**) into a RunConfig; `to_native()` returns the original dict."""
    p = Path(path)
    full = p if p.is_absolute() or root is None else root / p
    raw = full.read_bytes()
    cfg = json.loads(raw)
    rel = str(p) if not p.is_absolute() else str(p.relative_to(root)) if root else str(p)
    rel = rel[rel.index("configs/"):] if "configs/" in rel else rel
    family, stage = classify_legacy(rel, cfg)
    params = copy.deepcopy(cfg)
    inputs: dict[str, InputValue] = {}
    for k in INPUT_KEYS:
        v = params.get(k)
        if isinstance(v, str) and v.startswith("artifacts/") and _is_run_path(v):
            inputs[k] = path_to_ref(params.pop(k))
        elif isinstance(v, list) and v and all(isinstance(x, str) and x.startswith("artifacts/") and _is_run_path(x) for x in v):
            inputs[k] = _compress([path_to_ref(x) for x in params.pop(k)])
    flags: dict[str, Any] = {n: None for n in FLAG_NAMES}
    absent: list[str] = []
    for name, where in FLAG_SPEC[(family, stage)].items():
        if where == META:
            flags[name] = DEFAULT_CONTACT_VERSION if name == "contact_version" else None
            continue
        if where == CLI:
            continue
        if _has_path(params, where):
            flags[name] = _pop_path(params, where)
        else:
            key = (family, f"{name}@{stage}") if (family, f"{name}@{stage}") in LEGACY_FLAG_DEFAULTS else (family, name)
            if name == "zero_prev_action":
                # no default: keep absent. RunConfig needs a value, so record the legacy reader's value (False,
                # with the W3 warning at train/eval time); absent_flags keeps it out of the native dict.
                flags[name] = False
            elif key in LEGACY_FLAG_DEFAULTS:
                flags[name] = LEGACY_FLAG_DEFAULTS[key]
            else:
                raise RunConfigError(f"{rel}: no legacy default known for flag {name!r} ({family}/{stage})")
            absent.append(name)
    variant = _legacy_variant(stage, rel, cfg)
    m = _LINEAGE.search(f"{Path(rel).stem} {cfg.get('name', '')}")
    lineage = m.group(1) if m else Path(rel).stem
    seed = cfg.get("seed") if isinstance(cfg.get("seed"), int) else 0
    return RunConfig(schema_version=SCHEMA_VERSION, family=family, stage=stage, variant=variant, seed=seed,
                     lineage=lineage, track="legacy", tag=Path(rel).stem, inputs=inputs, flags=Flags(**flags),
                     params=params, legacy=LegacyInfo(path=rel, sha256=hashlib.sha256(raw).hexdigest(),
                                                      absent_flags=absent, key_order=list(cfg.keys())))


_RUN_PATH = re.compile(r"artifacts/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-/]*)?")


def _is_run_path(v: str) -> bool:
    return bool(_RUN_PATH.fullmatch(v)) and not v.endswith("/")


def iter_legacy_configs(root: Path) -> list[Path]:
    """Every run config under configs/ (NON_RUN_CONFIGS excluded)."""
    return sorted(p for p in (root / "configs").rglob("*.json")
                  if str(p.relative_to(root)) not in NON_RUN_CONFIGS)
