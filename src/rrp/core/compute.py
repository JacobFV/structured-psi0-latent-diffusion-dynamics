"""Compute options of a run (docs/architecture.md 14.7; unit prec of the D-147 compute-aware acceleration).

ONE module for every trainer: the `Compute` block of `RunConfig` (`compute:`), the process-wide ambient value the pipeline
sets from the rendered RunConfig (`configure`, called by `harness.pipelines.base.apply_run_context`), and `setup`, which a
trainer calls once to get its `Cx` (autocast context, TF32 switch, compile wrapper with the CUDA-graph option, fp32
islands, provenance stamp). It lives in `core` because the numerics islands sit in `policies` (probe / flow losses) and
the trainers in `harness`; it imports torch lazily so the CLI bootstrap stays stdlib + pydantic.

Everything is an explicit, recorded option and today's behaviour is the default:
- `RunConfig.compute is None` (the default) is serialised as ABSENT, so every existing `config_hash` / `config.json` /
  golden is unchanged; a non-default block hashes into the config, so runs with different compute settings never adopt
  each other's outputs (and the pipeline manifest records the block).
- `precision: None` means "each trainer's historical behaviour" (`LEGACY`: fp32, except the pointer and Psi0 trainers,
  which have always run bf16 autocast on CUDA, and the behaviour trainer, which has always enabled TF32 matmul).
  `precision: bf16` is autocast for matmul / conv / attention PLUS the fp32 islands below; `precision: fp32` turns the
  historical autocast off.
- fp32 islands (`f32`): a function decorated with it (likelihood / probe variance / loss reductions) runs with autocast
  disabled on float32 copies of its tensor arguments, but ONLY under an explicit `precision: bf16`. Under the legacy
  default they are the identity, so legacy numerics (including the pointer / Psi0 bf16 runs) are bit-for-bit unchanged.
  Optimizer state is always fp32 (autocast never touches parameters). The adapt.py likelihood ratios are forced to fp32
  by their own code and are not routed through here.
- `compile` / `cuda_graphs`: `compile: off | reduce-overhead | max-autotune` wraps the callables a trainer hands to
  `Cx.compile`. `reduce-overhead` IS CUDA graphs (inductor mode) and so needs `cuda_graphs: true`; `max-autotune` uses the
  graph-free mode unless `cuda_graphs: true`. A compile failure on the first call falls back to eager and is RECORDED in
  the stamp (`compiled[name].status`), never silent.
- `seeds_per_job`, `eval_backend`: declared here so every compute choice hashes together; the units that implement them
  (seeds, eval) read them from `current()`.
"""
from __future__ import annotations

import contextlib
import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA = "compute-1"

# trainer name -> historical behaviour. Anything absent: fp32, TF32 untouched.
LEGACY: dict[str, dict] = {
    "behavior": dict(precision="fp32", tf32=True),          # device_setup() always set matmul.allow_tf32
    "pointer": dict(precision="bf16"),                      # harness.train.pointer.train.amp (CUDA autocast)
    "psi0": dict(precision="bf16"),                         # policies.psi0.train (CUDA autocast)
}


class Compute(BaseModel):
    """`RunConfig.compute`. Defaults reproduce today's behaviour exactly (see the module docstring)."""
    model_config = ConfigDict(extra="forbid")
    precision: Literal["fp32", "bf16"] | None = None     # None: the trainer's historical precision (LEGACY)
    tf32: bool | None = None                              # None: leave torch's flag (LEGACY[...]['tf32'] where set)
    compile: Literal["off", "reduce-overhead", "max-autotune"] = "off"
    cuda_graphs: bool = False
    seeds_per_job: int = Field(1, ge=1)                   # seeds unit
    eval_backend: Literal["cpu", "warp"] = "cpu"          # eval unit

    @model_validator(mode="after")
    def _check(self):
        if self.cuda_graphs and self.compile == "off":
            raise ValueError("compute.cuda_graphs needs compute.compile != off")
        if self.compile == "reduce-overhead" and not self.cuda_graphs:
            raise ValueError("compute.compile 'reduce-overhead' is the CUDA-graph mode: set compute.cuda_graphs: true "
                             "(or use compile: max-autotune)")
        return self

    @property
    def is_default(self) -> bool:
        return self == Compute()

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------------------------------- ambient value
_AMBIENT = Compute()
_ACTIVE: "Cx | None" = None


def configure(c: Compute | None) -> Compute:
    """Set the process-wide compute block (the pipeline does, from the rendered RunConfig); returns the previous one."""
    global _AMBIENT
    prev, _AMBIENT = _AMBIENT, (c if c is not None else Compute())
    return prev


def current() -> Compute:
    return _AMBIENT


# ------------------------------------------------------------------------------------------------------ fp32 islands
def _to_f32(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.float() if x.is_floating_point() and x.dtype != torch.float32 else x
    if isinstance(x, dict):
        return {k: _to_f32(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_f32(v) for v in x)
    return x


def upcast(*xs):
    """float32 copies of `xs` at the boundary between a net's (possibly bf16) outputs and a loss reduction: under an explicit
    `precision: bf16` (islands on), else the arguments unchanged. One tensor in -> one out; several in -> a tuple."""
    out = tuple(_to_f32(x) for x in xs) if _island_on() else xs
    return out[0] if len(out) == 1 else out


def _island_on() -> bool:
    return _ACTIVE is not None and _ACTIVE.island


def f32(fn):
    """fp32 island: under an explicit `precision: bf16`, run `fn` with autocast disabled on float32 copies of its tensor
    arguments (dicts / tuples included). The identity otherwise (legacy numerics are untouched)."""
    @functools.wraps(fn)
    def wrapped(*a, **k):
        if not _island_on():
            return fn(*a, **k)
        import torch
        with torch.autocast(device_type=_ACTIVE.device_type, enabled=False):
            return fn(*_to_f32(a), **_to_f32(k))
    return wrapped


# ----------------------------------------------------------------------------------------------------------- trainer
@dataclass
class Cx:
    """What a trainer gets from `setup`. `name` is the trainer id (a key of LEGACY, or a dotted name under one)."""
    name: str
    requested: Compute
    device_type: str                 # "cuda" | "cpu"
    precision: str                   # effective: fp32 | bf16
    precision_source: str            # "explicit" | "legacy"
    island: bool                     # fp32 islands active (explicit bf16)
    tf32: bool | None                # effective flag set on torch (None: untouched)
    compiled: dict = field(default_factory=dict)

    # ---- autocast
    def autocast(self):
        """bf16 autocast for matmul / conv / attention when the effective precision is bf16 (else a no-op context)."""
        if self.precision != "bf16":
            return contextlib.nullcontext()
        import torch
        return torch.autocast(self.device_type, dtype=torch.bfloat16)

    def fp32(self):
        """An fp32 region inside an autocast region (explicit bf16 only; else a no-op)."""
        if not self.island:
            return contextlib.nullcontext()
        import torch
        return torch.autocast(self.device_type, enabled=False)

    # ---- compile
    def _mode(self) -> str | None:
        c = self.requested
        if c.compile == "off" or self.device_type != "cuda":
            return None
        if c.compile == "reduce-overhead":
            return "reduce-overhead"
        return "max-autotune" if c.cuda_graphs else "max-autotune-no-cudagraphs"

    def compile(self, target, name: str, methods: tuple[str, ...] | None = None):
        """Compile `target` per the run's compute block and return it. A callable is wrapped; an nn.Module gets `methods`
        (default: forward) replaced by compiled ones on the instance (state_dict keys unchanged). `off` (and CPU) returns `target` untouched."""
        mode = self._mode()
        if mode is None:
            self.compiled[name] = dict(status="off")
            return target
        import torch
        rec = self.compiled[name] = dict(status="pending", mode=mode)
        if isinstance(target, torch.nn.Module):
            for m in methods or ("forward",):       # instance attributes: state_dict keys and class code are untouched
                eager = getattr(target, m)
                setattr(target, m, _Guarded(torch.compile(eager, mode=mode), eager, rec))
            return target
        return _Guarded(torch.compile(target, mode=mode), target, rec)

    # ---- provenance
    def stamp(self) -> dict:
        return dict(schema=SCHEMA, trainer=self.name, requested=self.requested.to_dict(),
                    effective=dict(precision=self.precision, precision_source=self.precision_source, fp32_islands=self.island,
                                   tf32=self.tf32, device_type=self.device_type),
                    compiled=self.compiled, torch=_torch_version())

    def write_stamp(self, where) -> dict:
        """The stamp as JSON: `<where>/compute.json` for a directory, or `where` itself when it ends in `.json`. It holds
        the requested block and what was effective (precision source, islands, TF32, per-callable compile status /
        fallback reason); every trainer writes it next to its result."""
        s = self.stamp()
        p = Path(where)
        p = p if p.suffix == ".json" else p / "compute.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(s, indent=1, default=str))
        return s


def _torch_version() -> str | None:
    try:
        import torch
        return torch.__version__
    except Exception:  # noqa: BLE001
        return None


class _Guarded:
    """A compiled callable that falls back to its eager original if the FIRST call raises (recorded in `rec`)."""

    def __init__(self, compiled, eager, rec: dict):
        self.compiled, self.eager, self.rec = compiled, eager, rec

    def __call__(self, *a, **k):
        if self.rec["status"] == "pending":
            try:
                out = self.compiled(*a, **k)
                self.rec["status"] = "compiled"
                return out
            except Exception as e:  # noqa: BLE001 - recorded, then eager for the rest of the run
                self.rec.update(status="fallback_eager", reason=f"{type(e).__name__}: {str(e)[:300]}")
                self.compiled = self.eager
        return self.compiled(*a, **k)


def resolve(name: str, dev, compute: Compute | None = None) -> Cx:
    """The `Cx` for trainer `name` on device `dev` under `compute` (default: the ambient block). No torch side effects."""
    c = compute or _AMBIENT
    base = name.split(".")[0]
    leg = LEGACY.get(base, {})
    dt = "cuda" if str(dev).startswith("cuda") else "cpu"
    if c.precision is not None:
        prec, src = c.precision, "explicit"
    else:
        prec, src = leg.get("precision", "fp32"), "legacy"
        if dt != "cuda":
            prec = "fp32"                  # the legacy autocast was CUDA-only
    tf32 = c.tf32 if c.tf32 is not None else leg.get("tf32")
    return Cx(name=name, requested=c, device_type=dt, precision=prec, precision_source=src,
              island=(src == "explicit" and prec == "bf16"), tf32=tf32)


def setup(name: str, dev, compute: Compute | None = None) -> Cx:
    """Resolve `name`'s compute settings, apply the TF32 switch to torch and make this the ACTIVE `Cx` (what the fp32
    islands consult). Call once at trainer entry, after the device is chosen."""
    global _ACTIVE
    cx = resolve(name, dev, compute)
    if cx.tf32 is not None and cx.device_type == "cuda":
        import torch
        torch.backends.cuda.matmul.allow_tf32 = bool(cx.tf32)
        if (compute or _AMBIENT).tf32 is not None:         # an explicit request: convolutions and the precision hint follow
            torch.backends.cudnn.allow_tf32 = bool(cx.tf32)
            torch.set_float32_matmul_precision("high" if cx.tf32 else "highest")
    _ACTIVE = cx
    return cx


def active() -> "Cx | None":
    return _ACTIVE
