"""Run provenance (W3, docs/strategy.md): one typed record that says which physics, featurizer, code, weights,
training flags and controller source produced a dataset, checkpoint or evaluation.

- `physics_provenance(model, contact_version=...)` extracts the MuJoCo solver settings that change dynamics
  (version, timestep, integrator, cone, impratio, noslip/solver iterations) plus the contact model version string.
  `contact_version` is where the contact track's versioned contact models plug in (rrp.morphology.contact
  `version_str`, recorded as scenario meta["contact_model"]); code that predates them is "contact_v1".
- `FEATURIZER_VERSION` is the single featurizer constant (aliased by rrp.data.collect.FEATURIZER_VERSION and
  the training modules).
- `Source` / `SourceLabel` are the controller-source vocabulary. `parse_source` maps every legacy free string
  (contracts.action.Source values, "oracle", "target_encoder_oracle", tracker_source values, eval row strings)
  onto it; old files are never rewritten.
- `Provenance.legacy=True` marks records reconstructed from runs that were written before this module existed.
- `weights_digest` fingerprints a state_dict (identical to the arm bundle fingerprint, D-038).
- `resolve_zero_prev_action` makes the B-1 flag explicit: a new run without it is an error, a legacy reader warns.

This module imports only the standard library, the pydantic contracts base and (lazily) torch/mujoco.
"""
from __future__ import annotations

import enum
import functools
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from .base import Strict

PROVENANCE_SCHEMA = "provenance-1"
FEATURIZER_VERSION = "feat-v2"          # THE featurizer version (single source of truth)
CONTACT_VERSION_DEFAULT = "contact_v1"  # physics of all code before the contact track's v2 (rrp.morphology.contact)
UNFINGERPRINTED = "unfingerprinted"


# ----------------------------------------------------------------------------------------------------- physics
class PhysicsProvenance(Strict):
    mujoco_version: str
    timestep: float
    integrator: str
    cone: str
    impratio: float
    solver: str
    iterations: int
    ls_iterations: int
    noslip_iterations: int
    contact_version: str = CONTACT_VERSION_DEFAULT
    actuator_limits: str | None = None      # legged bodies: torque-limit version embedded in the model (W1, D-107)
    grasp_contact_version: str | None = None  # arm worlds: finger/object contact model (W7, D-108); None = grasp_v1 (legacy)

    def to_dict(self) -> dict:
        d = self.model_dump(mode="json")
        if d.get("grasp_contact_version") is None:     # legacy records stay byte-identical (no new key)
            d.pop("grasp_contact_version", None)
        return d


def _enum_name(enum_cls, value) -> str:
    try:
        return enum_cls(int(value)).name.removeprefix("mjINT_").removeprefix("mjCONE_").removeprefix("mjSOL_").lower()
    except Exception:  # noqa: BLE001 - unknown enum value: keep the integer
        return str(int(value))


def model_contact_version(model) -> str | None:
    """The `contact_version` text element of a compiled model (written by the contact track's legged_world,
    rrp.morphology.contact.apply_world); None when the model has none."""
    import mujoco
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, "contact_version")
    if tid < 0:
        return None
    adr, n = model.text_adr[tid], model.text_size[tid]
    return bytes(model.text_data[adr:adr + n - 1]).decode()


def physics_provenance(model, contact_version: str | None = None) -> PhysicsProvenance:
    """Extract the dynamics-relevant solver options from an MjModel. contact_version: explicit value, else the
    model's `contact_version` text element, else CONTACT_VERSION_DEFAULT. An explicit value that contradicts the
    model's element is an error (never record a physics version the model was not built with)."""
    import mujoco
    embedded = model_contact_version(model)
    if contact_version is not None and embedded is not None and contact_version != embedded:
        raise ValueError(f"contact_version {contact_version!r} contradicts the model's {embedded!r}")
    contact_version = contact_version or embedded or CONTACT_VERSION_DEFAULT
    o = model.opt
    return PhysicsProvenance(mujoco_version=mujoco.__version__, timestep=float(o.timestep),
                             integrator=_enum_name(mujoco.mjtIntegrator, o.integrator),
                             cone=_enum_name(mujoco.mjtCone, o.cone), impratio=float(o.impratio),
                             solver=_enum_name(mujoco.mjtSolver, o.solver), iterations=int(o.iterations),
                             ls_iterations=int(o.ls_iterations), noslip_iterations=int(o.noslip_iterations),
                             contact_version=contact_version, actuator_limits=_model_actuator_limits(model),
                             grasp_contact_version=_model_text(model, "grasp_contact_version"))


def _model_text(model, name):
    import mujoco
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, name)
    if tid < 0:
        return None
    adr, n = model.text_adr[tid], model.text_size[tid]
    return bytes(model.text_data[adr:adr + n - 1]).decode()


def _model_actuator_limits(model):
    """Legged-body actuator-limit version from the model's `*actuator_limits` text element (attach may prefix it), or None."""
    import mujoco
    for t in range(model.ntext):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXT, t) or "").endswith("actuator_limits"):
            adr, n = model.text_adr[t], model.text_size[t]
            return bytes(model.text_data[adr:adr + n - 1]).decode()
    return None


# ------------------------------------------------------------------------------------------------------ source
class Source(str, enum.Enum):
    SCRIPTED_TEACHER = "scripted_teacher"        # scripted expert (may read privileged state; see privileged flag)
    PRIVILEGED_TEACHER = "privileged_teacher"    # expert that explicitly consumes simulator truth
    ORACLE = "oracle"                            # teacher-encoded packet -> system 0: DIAGNOSTIC, not deployable
    LEARNED = "learned"                          # learned:<ckpt>
    BC = "bc"                                    # bc:<ckpt> (behaviour-cloning baseline)
    RANDOM = "random"
    MOCK = "mock"                                # mock / debug / synthetic commands
    CPG_TRACKER = "cpg_tracker"                  # scripted legged body tracker (was tracker_source "scripted_controller")
    LEARNED_TRACKER = "learned_tracker"          # learned legged body tracker (PPO actor)
    USER = "user"                                # GUI user command (contracts.action legacy value)
    REPLAY = "replay"                            # replayed recorded commands/packets
    UNKNOWN = "unknown"                          # legacy record whose source was not stored (never for new writes)


NEEDS_CKPT = {Source.LEARNED, Source.BC}

# legacy free strings -> canonical kind (the on-disk strings stay as they are)
LEGACY_SOURCE_MAP: dict[str, Source] = {
    "teacher": Source.SCRIPTED_TEACHER,
    "scripted_teacher": Source.SCRIPTED_TEACHER,
    "privileged_teacher": Source.PRIVILEGED_TEACHER,
    "privileged": Source.PRIVILEGED_TEACHER,
    "oracle": Source.ORACLE,
    "target_encoder_oracle": Source.ORACLE,
    "privileged_oracle_packet": Source.ORACLE,
    "oracle_diagnostic": Source.ORACLE,
    "learned": Source.LEARNED,
    "rep": Source.LEARNED,
    "bc": Source.BC,
    "random": Source.RANDOM,
    "mock": Source.MOCK,
    "debug": Source.MOCK,
    "user": Source.USER,
    "replay": Source.REPLAY,
    "scripted_controller": Source.CPG_TRACKER,
    "cpg": Source.CPG_TRACKER,
    "cpg_tracker": Source.CPG_TRACKER,
    "learned_tracker": Source.LEARNED_TRACKER,
    "unknown": Source.UNKNOWN,
}


class SourceLabel(Strict):
    kind: Source
    detail: str | None = None          # checkpoint for learned/bc; variant for others (e.g. "arc_only")

    def __str__(self) -> str:
        return self.kind.value + (f":{self.detail}" if self.detail else "")

    @property
    def deployable(self) -> bool:
        return self.kind in (Source.LEARNED, Source.BC, Source.CPG_TRACKER, Source.LEARNED_TRACKER, Source.RANDOM)


def parse_source(s: str | Source | SourceLabel, *, strict: bool = False) -> SourceLabel:
    """Map a (possibly legacy) source string onto SourceLabel. strict=True (new writes): learned/bc need a
    checkpoint and only canonical kind names are accepted."""
    if isinstance(s, SourceLabel):
        lab = s
    elif isinstance(s, Source):
        lab = SourceLabel(kind=s)
    else:
        if not isinstance(s, str) or not s:
            raise ValueError(f"source must be a non-empty string, got {s!r}")
        head, _, rest = s.partition(":")
        if head not in LEGACY_SOURCE_MAP:
            raise ValueError(f"unknown source {s!r}; known kinds: {sorted(LEGACY_SOURCE_MAP)}")
        kind = LEGACY_SOURCE_MAP[head]
        if strict and head != kind.value:
            raise ValueError(f"legacy source name {head!r}: write {kind.value!r} instead")
        lab = SourceLabel(kind=kind, detail=rest or None)
    if strict and lab.kind is Source.UNKNOWN:
        raise ValueError("'unknown' is only for legacy records; new writes must state their source")
    if strict and lab.kind in NEEDS_CKPT and not lab.detail:
        raise ValueError(f"{lab.kind.value} source needs a checkpoint: '{lab.kind.value}:<ckpt>'")
    return lab


def source_label(kind: Source | str, detail: str | None = None) -> str:
    """Validated canonical string for NEW writes, e.g. source_label('learned', 'runs/x/policy.pt')."""
    k = Source(kind)
    return str(parse_source(SourceLabel(kind=k, detail=detail), strict=True))



# --------------------------------------------------------------------------- source labels on rows (D-126, sl-1)
# Eval/data rows historically carry a free-string `source` ("scripted_teacher(privileged)", "learned:x+edit:y",
# "target_encoder_oracle(ORACLE DIAGNOSTIC: ...)"). Those strings are never rewritten. A default-OFF, versioned
# switch makes NEW rows additionally carry a strict canonical `source_label` plus `source_label_version`.
# Readers use `row_source(row)`, which prefers `source_label` and falls back to parsing the legacy `source`.
SOURCE_LABELS_ENV = "RRP_SOURCE_LABELS"      # "canonical" = on; unset / "" / "legacy" = off (the default)
SOURCE_LABEL_VERSION = "sl-1"


def canonical_source_labels(enabled: bool | None = None) -> bool:
    """Whether new rows get a canonical `source_label`. An explicit argument wins; else $RRP_SOURCE_LABELS
    ("canonical" -> on; unset, "" or "legacy" -> off). Any other value is an error (no silent typo)."""
    if enabled is not None:
        return bool(enabled)
    v = os.environ.get(SOURCE_LABELS_ENV, "").strip().lower()
    if v in ("", "legacy", "off", "0"):
        return False
    if v in ("canonical", "on", "1", SOURCE_LABEL_VERSION):
        return True
    raise ValueError(f"{SOURCE_LABELS_ENV}={v!r}: expected 'canonical' or 'legacy'")


def parse_legacy_source(s) -> SourceLabel:
    """Lenient reader for on-disk row strings: parse_source plus the ladder's annotated forms
    "scripted_teacher(privileged)", "learned(system-i flow)", "target_encoder_oracle(ORACLE DIAGNOSTIC: ...)"
    (the annotation becomes no detail). Raises ValueError for unparseable values."""
    if isinstance(s, str):
        head = s.split("(", 1)[0] if "(" in s.split(":", 1)[0] else s
        return parse_source(head)
    return parse_source(s)


def row_source(row: dict, *, default: SourceLabel | None = None) -> SourceLabel:
    """Canonical source of an eval/data row in either format: the `source_label` key (sl-1 rows) if present,
    else the legacy `source` key. `default` is returned for a missing/unparseable legacy source (None: raise)."""
    if row.get("source_label") is not None:
        return parse_source(row["source_label"], strict=True)
    try:
        return parse_legacy_source(row.get("source"))
    except (ValueError, TypeError):
        if default is not None:
            return default
        raise


def stamp_source_label(row: dict, kind: Source | str, detail: str | None = None, *,
                       enabled: bool | None = None) -> dict:
    """Add `source_label` (strict `source_label(kind, detail)`) and `source_label_version` to a NEW row when the
    switch is on (see canonical_source_labels); a no-op otherwise, so default rows stay byte-identical. The
    legacy `source` key is never touched; if it parses, its kind must agree with `kind` (catches mislabels)."""
    if not canonical_source_labels(enabled):
        return row
    lab = source_label(kind, detail)
    if "source" in row:
        try:
            old = parse_legacy_source(row["source"]).kind
        except (ValueError, TypeError):
            old = None
        if old is not None and old is not Source(kind):
            raise ValueError(f"source_label {lab!r} contradicts the row's legacy source {row['source']!r}")
    row["source_label"] = lab
    row["source_label_version"] = SOURCE_LABEL_VERSION
    return row

# ------------------------------------------------------------------------------------------------ code / weights
def repo_root() -> Path:
    """The rrp source checkout (or $RRP_HOME; the cwd when rrp is an installed package): rrp.contracts.paths."""
    from .paths import rrp_home
    return rrp_home()


def installed_revision(dist: str = "rrp") -> str | None:
    """Commit of an rrp installed from git (pip/uv record it in direct_url.json); None otherwise."""
    try:
        from importlib.metadata import distribution
        d = json.loads(distribution(dist).read_text("direct_url.json") or "{}")
    except Exception:  # noqa: BLE001 - not installed / no direct_url
        return None
    return (d.get("vcs_info") or {}).get("commit_id")


@functools.lru_cache(maxsize=4)
def _git_info(root: str) -> tuple[str | None, bool | None]:
    env_sha = os.environ.get("RRP_GIT_SHA")
    try:
        sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip() or None
        if sha:
            st = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=20).stdout
            return sha, bool(st.strip())
    except Exception:  # noqa: BLE001 - no git binary
        pass
    rev = Path(root) / ".rrp_revision"          # synced copies (peer) have no .git; a sync may drop this file
    if rev.exists():
        try:
            d = json.loads(rev.read_text())
            return d.get("git_sha"), d.get("dirty")
        except Exception:  # noqa: BLE001
            pass
    return env_sha, None


class CodeProvenance(Strict):
    git_sha: str | None = None           # None: unknown (no .git, e.g. a synced peer copy without .rrp_revision)
    dirty: bool | None = None


def code_provenance(root: Path | None = None) -> CodeProvenance:
    """git sha/dirty of `root` (default: the rrp checkout). An INSTALLED rrp (no checkout, no RRP_HOME) records the
    commit it was installed from, never the git state of whatever directory the consumer runs in."""
    if root is None and not os.environ.get("RRP_HOME"):
        from .paths import is_checkout
        if not is_checkout():
            rev = installed_revision()
            return CodeProvenance(git_sha=rev or os.environ.get("RRP_GIT_SHA"), dirty=False if rev else None)
    sha, dirty = _git_info(str(root or repo_root()))
    return CodeProvenance(git_sha=sha, dirty=dirty)


def weights_digest(state_dict: dict) -> str:
    """sha256 over parameter/buffer names and raw bytes (dtype-exact), sorted by name (12 hex chars).
    Identical to the arm bundle fingerprint (rrp.control.latent_realizer.weights_digest, D-038)."""
    import torch
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k]
        if not isinstance(t, torch.Tensor):
            continue
        t = t.detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:12]


def is_state_dict(v) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(v, dict) and len(v) > 0 and all(isinstance(x, torch.Tensor) for x in v.values())


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ the record
class Provenance(Strict):
    schema_version: Literal["provenance-1"] = PROVENANCE_SCHEMA
    source: str                                            # canonical SourceLabel string
    physics: PhysicsProvenance | None = None
    featurizer_version: str | None = None
    code: CodeProvenance = Field(default_factory=CodeProvenance)
    weights: dict[str, str] = Field(default_factory=dict)  # component -> weights_digest; {} for data / legacy
    bundle_fingerprint: str | None = None                   # digest over `weights` (None when there are no weights)
    versions: dict[str, str] = Field(default_factory=dict)  # e.g. latent_space_version, realizer_compat_version
    flags: dict[str, Any] = Field(default_factory=dict)     # zero_prev_action, realizer_drop_qd, latent lv_min, ...
    legacy: bool = False                                    # reconstructed for a run written before provenance-1
    created_at: float | None = None
    notes: str = ""

    @field_validator("source")
    @classmethod
    def _src(cls, v):
        return str(parse_source(v))

    @property
    def source_label(self) -> SourceLabel:
        return parse_source(self.source)

    @property
    def fingerprinted(self) -> bool:
        return bool(self.weights) and not self.legacy

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, s: str | dict) -> "Provenance":
        return cls.model_validate(json.loads(s) if isinstance(s, str) else s)


def bundle_fingerprint(weights: dict[str, str]) -> str | None:
    if not weights:
        return None
    return hashlib.sha256(json.dumps(weights, sort_keys=True).encode()).hexdigest()[:12]


def make_provenance(source: str | Source | SourceLabel, *, model=None, contact_version: str | None = None,
                    physics: PhysicsProvenance | dict | None = None, featurizer_version: str | None = None,
                    weights: dict | None = None, versions: dict | None = None, flags: dict | None = None,
                    notes: str = "") -> Provenance:
    """New-run provenance. Physics comes from `model` (an MjModel) or an already extracted `physics` record.
    `weights` may map names to state_dicts (fingerprinted here) or to digest strings."""
    if physics is None and model is not None:
        physics = physics_provenance(model, contact_version)
    w = {}
    for k, v in (weights or {}).items():
        w[k] = v if isinstance(v, str) else weights_digest(v)
    return Provenance(source=str(parse_source(source, strict=True)),
                      physics=physics,
                      featurizer_version=featurizer_version, code=code_provenance(), weights=w,
                      bundle_fingerprint=bundle_fingerprint(w), versions={k: str(v) for k, v in (versions or {}).items()},
                      flags=dict(flags or {}), created_at=time.time(), notes=notes)


def legacy_provenance(source: str | None = None, **kw) -> Provenance:
    """Record for a run written before provenance-1 (never claims fields that were not recorded)."""
    src = source or "unknown"
    try:
        src = str(parse_source(src))
    except ValueError:
        kw["notes"] = (kw.get("notes", "") + f" unparsed legacy source {src!r}").strip()
        src = "unknown"
    return Provenance(source=src, legacy=True, **kw)


def read_provenance(meta: dict | None) -> Provenance:
    """Provenance of a manifest / meta.json / checkpoint dict: the stored record if present, else a legacy one
    built from whatever the old format kept (source, featurizer, physics_dt)."""
    meta = meta or {}
    for key in ("provenance", "_provenance"):
        if isinstance(meta.get(key), dict):
            return Provenance.from_json(meta[key])
    return legacy_provenance(meta.get("source"), featurizer_version=meta.get("featurizer"),
                             notes="legacy: no provenance record (unfingerprinted)")


TRAINING_FLAG_KEYS = ("zero_prev_action", "realizer_drop_qd", "realizer_anchor", "realizer_qd_dropout", "probe_lv_min",
                      "semantic_weight", "packet_semantic_weight", "qd_dropout", "cf_mix", "init")
# NOTE (D-144 addendum, decision (b), unit R2c -- re-confirms R2's original finding): `probe_lv_min` / `semantic_weight`
# / `packet_semantic_weight` are kept here, byte-identical to pre-R2, although unit R2 (docs/relations.md 10) retires
# them as LatentConfig's own primary surface (see nets/semantic_latent.py `LatentConfig.factors` / the ONE legacy
# mapping table `LATENT_LEGACY_KEYS`). This function reads ANY config dict generically by key presence -- it is the
# provenance path every `save_checkpoint` call goes through (`nets/checkpoint.py`), arm and legged alike, and legged
# checkpoints/configs are not migrated by any merged unit yet (fanout unit R4, not merged: `legged_latent_train.py`
# still writes these flat keys). Dropping them would silently stop recording them for every still-legacy-style (arm
# OR legged) config or checkpoint. R2c confirmed concretely (not just by inspection) that even the last arm-only
# occurrences cannot be retired in isolation: `dags/arm_lineage.yaml` / `dags/templates/dual_lineage.yaml` still
# render a flat `latent.semantic_weight` block coupled to the `Flags.probe_lv_min` mechanism (`core/runconfig.py`'s
# `FLAG_NAMES` note), proven byte-identical to on-disk `configs/ladder/**` files by `test_dag.py::
# test_arm_dag_reproduces_legacy_configs` (general infra, not owned by this row). Only `binding_cf_weight` (renamed
# `cf_mix`) is retired outright: dead here otherwise (grep confirms no config ever set it), so nothing depends on
# the old name.


def training_flags(config: dict | None) -> dict:
    """Meaning-changing training flags from a run config (top level and the nested 'latent' block).
    zero_prev_action is always present in the result: None means the config did not state it (legacy)."""
    config = config if isinstance(config, dict) else {}
    out: dict = {}
    for src in (config.get("latent") if isinstance(config.get("latent"), dict) else {}, config):
        for k in TRAINING_FLAG_KEYS:
            if k in src:
                out[k] = src[k]
    out.setdefault("zero_prev_action", None)
    return out


# ------------------------------------------------------------------------------------------- zero_prev_action
class MissingFlagError(ValueError):
    pass


def resolve_zero_prev_action(cfg: dict, *, where: str, new_run: bool) -> bool:
    """Bug B-1 (D-044/D-045): zero_prev_action must be explicit. Missing in a NEW run -> MissingFlagError.
    Missing when reading/resuming an existing run -> loud warning and the legacy default False (contaminated
    input: the teacher's previous command is visible in node column 28)."""
    if "zero_prev_action" in cfg:
        v = cfg["zero_prev_action"]
        if not isinstance(v, bool):
            raise ValueError(f"{where}: zero_prev_action must be a bool, got {v!r}")
        return v
    if new_run:
        raise MissingFlagError(f"{where}: config has no explicit 'zero_prev_action' (bug B-1). Set it to true "
                               "(deployment-consistent) or false (legacy, contaminated) explicitly.")
    msg = (f"LEGACY {where}: 'zero_prev_action' missing -> assuming False (B-1 contaminated input). "
           "Set it explicitly in the config.")
    warnings.warn(msg, stacklevel=2)
    print(f"[provenance] WARNING {msg}", file=sys.stderr, flush=True)
    return False
