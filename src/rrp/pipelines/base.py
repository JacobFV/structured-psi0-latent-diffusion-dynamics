"""Pipeline(family): a registry of stages, each a function RunConfig -> outputs + a manifest with Provenance (W5).

Stage functions CALL the existing training/evaluation code (in-process library functions, or the existing script /
module entry points as subprocesses where the logic lives in a script's main, e.g. scripts/ladder.py). They never
reimplement numerics. `Pipeline(family).run(rc)`:
1. checks family/stage and that every input path exists;
2. calls the stage with a StageContext (native config, resolved inputs, out dir, subprocess helpers);
3. writes <out>/pipeline_manifest.json through the single manifest writer (rrp.data.manifest.write_manifest) with a
   Provenance (source label, flags, code revision) plus the RunConfig, its hash, the native config, input and output
   digests and the stage metrics. The DAG runner treats "rc 0 AND manifest with the same config_hash" as done.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rrp.contracts.paths import rrp_home
from rrp.contracts.runconfig import BUILTIN_FAMILIES, PIPELINE_STAGES, RunConfig, RunIndex, ensure_family

MANIFEST = "pipeline_manifest.json"


class StageError(RuntimeError):
    pass


class GateFailed(StageError):
    """The stage's output failed its W6 gate (rrp.evaluation.gates); <out>/gate_report.json holds the report. The job
    exits GATE_EXIT and run-dag marks the node failed with the gate's reason, without retries (D-112)."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__("gate failed: " + "; ".join(report.get("failed", [])))


GATE_EXIT = 86


def apply_gate(ctx: "StageContext", report: dict) -> dict:
    """Write <out>/gate_report.json; raise GateFailed on verdict 'fail' unless options.gate == 'report' (report only)."""
    from rrp.evaluation.gates import write_report
    write_report(report, ctx.out)
    ctx.log(f"gate {report['gate']}: {report['verdict']}" + (f" ({'; '.join(report['failed'])})" if report["failed"] else ""))
    if report["verdict"] == "fail" and ctx.opts.get("gate", "enforce") != "report":
        raise GateFailed(report)
    return dict(verdict=report["verdict"], failed=report["failed"], not_evaluated=report["not_evaluated"])


@dataclass
class StageContext:
    rc: RunConfig
    index: RunIndex
    root: Path
    python: str = sys.executable
    log: Callable[[str], None] = field(default=lambda m: print(f"[pipeline] {m}", flush=True))

    @property
    def native(self) -> dict:
        return self.rc.to_native(self.index)

    @property
    def out(self) -> Path:
        return self.root / self.rc.out

    @property
    def opts(self) -> dict:
        return dict(self.rc.options)

    def inp(self, key: str, required: bool = True):
        paths = self.rc.input_paths(self.index)
        if key not in paths:
            if required:
                raise StageError(f"{self.rc.family}/{self.rc.stage}: input {key!r} missing")
            return None
        return paths[key]

    def env(self, **extra) -> dict:
        e = dict(os.environ)
        src = str(self.root / "src")
        e["PYTHONPATH"] = src + (":" + e["PYTHONPATH"] if e.get("PYTHONPATH") and src not in e["PYTHONPATH"] else "")
        e.update({k: str(v) for k, v in extra.items()})
        return e

    def run(self, argv: list[str], *, env: dict | None = None, log_to: Path | None = None) -> None:
        """Run [python, *argv] (argv[0] may be '-m') in the repo root; non-zero exit -> StageError."""
        cmd = [self.python, *argv]
        self.log("run " + " ".join(cmd))
        fh = open(log_to, "a") if log_to else None
        try:
            r = subprocess.run(cmd, cwd=self.root, env=env or self.env(), stdout=fh, stderr=subprocess.STDOUT if fh else None)
        finally:
            if fh:
                fh.close()
        if r.returncode != 0:
            raise StageError(f"exit {r.returncode}: {' '.join(cmd)}" + (f" (log {log_to})" if log_to else ""))

    def run_parallel(self, jobs: list[tuple[list[str], dict | None, Path | None]], workers: int) -> None:
        """Bounded parallel subprocesses; every job runs, then any failure raises (listing all failures)."""
        errs = []

        def one(j):
            try:
                self.run(j[0], env=j[1], log_to=j[2])
            except StageError as e:
                errs.append(str(e))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            list(ex.map(one, jobs))
        if errs:
            raise StageError(f"{len(errs)} of {len(jobs)} jobs failed: " + "; ".join(errs[:5]))


@dataclass
class StageSpec:
    family: str
    stage: str
    fn: Callable[[StageContext], dict]
    source: str          # Source kind of what the stage produces (for the manifest's Provenance)
    doc: str = ""


_REGISTRY: dict[tuple[str, str], StageSpec] = {}


def register(family: str, stage: str, *, source: str):
    """Decorator: register `fn(ctx) -> {"outputs", "metrics", ...}` as `family`'s `stage`. An extension family must be
    registered first (rrp.contracts.runconfig.register_family); stages are the fixed PIPELINE_STAGES."""
    if stage not in PIPELINE_STAGES:
        raise ValueError(f"unknown stage {stage}")
    if family not in BUILTIN_FAMILIES:
        ensure_family(family)

    def deco(fn):
        _REGISTRY[(family, stage)] = StageSpec(family, stage, fn, source, (fn.__doc__ or "").strip().split("\n")[0])
        return fn
    return deco


def _load_families():
    from rrp.pipelines import arm, dual, legged  # noqa: F401  (registration side effects)


def unregister_stages(family: str) -> None:
    """Drop every stage of an extension family (tests)."""
    if family in BUILTIN_FAMILIES:
        raise ValueError(f"{family} is built in")
    for k in [k for k in _REGISTRY if k[0] == family]:
        del _REGISTRY[k]


class Pipeline:
    def __init__(self, family: str):
        _load_families()
        if family not in BUILTIN_FAMILIES:
            ensure_family(family)                    # extension families: registered or loaded from entry points
        self.family = family

    def stages(self) -> list[str]:
        return [s for s in PIPELINE_STAGES if (self.family, s) in _REGISTRY]

    def spec(self, stage: str) -> StageSpec:
        try:
            return _REGISTRY[(self.family, stage)]
        except KeyError:
            raise StageError(f"{self.family}: stage {stage!r} is not implemented") from None

    def run(self, rc: RunConfig, *, index: RunIndex | None = None, root: Path | None = None,
            check_inputs: bool = True) -> dict:
        if rc.family != self.family:
            raise StageError(f"config family {rc.family} != pipeline {self.family}")
        spec = self.spec(rc.stage)
        root = Path(root or rrp_home()).resolve()     # rrp_home(): the checkout, $RRP_HOME or the cwd (installed)
        index = index or RunIndex.load(root=root)
        ctx = StageContext(rc=rc, index=index, root=root)
        missing = [p for p in _flat(rc.input_paths(index).values()) if check_inputs and not (root / p).exists()]
        if missing:
            raise StageError(f"missing inputs ({len(missing)}): {missing[:5]}")
        out = ctx.out
        out.mkdir(parents=True, exist_ok=True)
        (out / MANIFEST).unlink(missing_ok=True)        # a stale manifest must never mark a rerun done
        (out / "gate_report.json").unlink(missing_ok=True)   # nor a stale gate report fail it
        t0 = time.time()
        cwd = os.getcwd()
        gc = rc.options.get("grasp_contact")            # D-126: arm grasp contact version for the whole stage (opt-in)
        gc_old = os.environ.get("RRP_GRASP_CONTACT")
        if gc is not None:
            from rrp.physics.grasp_contact import resolve
            if gc_old is not None and resolve(gc_old) != resolve(gc):
                raise StageError(f"options.grasp_contact {gc!r} contradicts $RRP_GRASP_CONTACT={gc_old!r}")
            os.environ["RRP_GRASP_CONTACT"] = resolve(gc)   # in-process code and every subprocess (ctx.env) see it
        kf = rc.options.get("kinfeat")                  # D-137 ablation: $RRP_KINFEAT for the whole stage (opt-in)
        kf_old = os.environ.get("RRP_KINFEAT")
        if kf is not None:
            if kf != "v1":
                raise StageError(f"options.kinfeat {kf!r}: only v1")
            if kf_old not in (None, "", kf):
                raise StageError(f"options.kinfeat {kf!r} contradicts $RRP_KINFEAT={kf_old!r}")
            os.environ["RRP_KINFEAT"] = kf
        os.chdir(root)                                   # existing code resolves artifacts/... relative to the repo
        try:
            res = spec.fn(ctx) or {}
        finally:
            os.chdir(cwd)
            if kf is not None:
                if kf_old is None:
                    os.environ.pop("RRP_KINFEAT", None)
                else:
                    os.environ["RRP_KINFEAT"] = kf_old
            if gc is not None:
                if gc_old is None:
                    os.environ.pop("RRP_GRASP_CONTACT", None)
                else:
                    os.environ["RRP_GRASP_CONTACT"] = gc_old
        return write_stage_manifest(ctx, spec, res, started=t0)


def _flat(vals):
    for v in vals:
        if isinstance(v, list):
            yield from v
        else:
            yield v


def _digest(p: Path) -> str | None:
    from rrp.contracts.provenance import file_digest
    return file_digest(p) if p.is_file() else None


def write_stage_manifest(ctx: StageContext, spec: StageSpec, res: dict, *, started: float) -> dict:
    from rrp.contracts.provenance import make_provenance, source_label
    from rrp.data.manifest import write_manifest
    rc = ctx.rc
    outputs = {k: str(v) for k, v in (res.get("outputs") or {}).items()}
    missing = [v for v in outputs.values() if not (ctx.root / v).exists()]
    if missing:
        raise StageError(f"stage finished but outputs are missing: {missing[:5]}")
    src = res.get("source") or spec.source
    if ":" not in src and src in ("learned", "bc"):
        src = source_label(src, res.get("source_detail") or rc.run_id)
    inputs = {}
    for k, v in rc.input_paths(ctx.index).items():
        vs = v if isinstance(v, list) else [v]
        inputs[k] = [dict(path=p, digest=_digest(ctx.root / p)) for p in vs]
    prov = make_provenance(src, flags=rc.flag_dict(), versions=res.get("versions") or {},
                           notes=f"pipeline {rc.family}/{rc.stage} {rc.run_id}")
    extra = dict(schema="pipeline-manifest-1", run_id=rc.run_id, config_hash=rc.config_hash(),
                 runconfig=rc.model_dump(mode="json"), native_config=_safe_native(ctx),
                 inputs=inputs, outputs={k: dict(path=v, digest=_digest(ctx.root / v)) for k, v in outputs.items()},
                 metrics=res.get("metrics") or {}, started=started, finished=time.time())
    body = write_manifest(ctx.out, rc.run_id, [], extra, provenance=prov, filename=MANIFEST)
    ctx.log(f"manifest {ctx.out / MANIFEST} ({rc.config_hash()})")
    return body


def _safe_native(ctx: StageContext):
    try:
        return ctx.native
    except Exception as e:  # stages without a native config (evaluations) still get a manifest
        return {"_error": str(e)}


def read_stage_manifest(out_dir: Path) -> dict | None:
    p = Path(out_dir) / MANIFEST
    return json.loads(p.read_text()) if p.exists() else None
