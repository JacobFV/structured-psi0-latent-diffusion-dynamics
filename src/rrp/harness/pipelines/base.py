"""Pipeline(family): a registry of stages, each a function RunConfig -> outputs + a manifest with Provenance (W5).

Stage functions CALL the existing training/evaluation code (in-process library functions, or `rrp <group> <tool>`
subprocesses, e.g. `rrp suite ladder`). They never
reimplement numerics. `Pipeline(family).run(rc)`:
1. checks family/stage and that every input path exists;
2. calls the stage with a StageContext (native config, resolved inputs, out dir, subprocess helpers);
3. writes <out>/pipeline_manifest.json through the single manifest writer (rrp.harness.data.manifest.write_manifest) with a
   Provenance (source label, flags, code revision) plus the RunConfig, its hash, the native config, input and output
   digests and the stage metrics. The DAG runner treats "rc 0 AND manifest with the same config_hash" as done.

docs/architecture.md 14.2 (unit F3): the stage registry is OPEN (`register_stage`; a family registers in its own
pipeline module and the stage names are derived); the RunConfig is the ONE channel to child processes: `apply_run_context`
runs at stage entry in the parent and, through `StageContext.run` -> `child_main`, in every subprocess a stage spawns;
a grandchild (a process a child spawns) finds the same context at `$RRP_RUN_CONTEXT` (the path of the rendered
`run_context.json`; `rrp.cli.main` applies it on entry);
`PIPELINE_VERSION` + the factor / catalog versions (`stage_versions`) are the pins a completed node is adopted under.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rrp.core.paths import rrp_home
from rrp.core.runconfig import (BUILTIN_FAMILIES, PIPELINE_STAGES, RunConfig, RunIndex, declare_stage, ensure_family,
                                families, register_family, undeclare_stage)

MANIFEST = "pipeline_manifest.json"
CONTEXT_FILE = "run_context.json"
CONTEXT_ENV = "RRP_RUN_CONTEXT"      # path of the rendered context: exported to every child, read by `apply_run_context()`
# Bumped when a stage changes what its outputs MEAN (not for refactors): a completed node is adopted only under the
# same PIPELINE_VERSION (docs/architecture.md 14.2). Manifests written before it existed carry no pin and are never adopted.
PIPELINE_VERSION = 1


class StageError(RuntimeError):
    pass


class GateFailed(StageError):
    """The stage's output failed its W6 gate (rrp.harness.eval.gates); <out>/gate_report.json holds the report. The job
    exits GATE_EXIT and run-dag marks the node failed with the gate's reason, without retries (D-112)."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__("gate failed: " + "; ".join(report.get("failed", [])))


GATE_EXIT = 86


def apply_gate(ctx: "StageContext", report: dict) -> dict:
    """Write <out>/gate_report.json; raise GateFailed on verdict 'fail' unless options.gate == 'report' (report only)."""
    from rrp.harness.eval.gates import write_report
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
        """The child environment: resources (PYTHONPATH, threads, devices) and `$RRP_RUN_CONTEXT`, the path of the rendered
        RunConfig that every descendant applies. Run semantics travel in the RunConfig."""
        e = dict(os.environ)
        e[CONTEXT_ENV] = str(self.context_file())
        src = str(self.root / "src")
        e["PYTHONPATH"] = src + (":" + e["PYTHONPATH"] if e.get("PYTHONPATH") and src not in e["PYTHONPATH"] else "")
        e.update({k: str(v) for k, v in extra.items()})
        return e

    def context_file(self) -> Path:
        """<out>/run_context.json: this stage's rendered RunConfig, the input of every child's `apply_run_context`."""
        p = self.out / CONTEXT_FILE
        self.out.mkdir(parents=True, exist_ok=True)
        text = self.rc.to_json()
        if not p.exists() or p.read_text() != text:
            tmp = p.with_suffix(".tmp")
            tmp.write_text(text)
            tmp.replace(p)
        return p

    def run(self, argv: list[str], *, env: dict | None = None, log_to: Path | None = None) -> None:
        """Run [python, *argv] (argv[0] may be '-m') in the repo root; non-zero exit -> StageError. The child starts
        through `child_main`, which applies this stage's RunConfig (`apply_run_context`) before the target runs."""
        cmd = [self.python, "-c", _CHILD_BOOT, "--context", str(self.context_file()), "--", *argv]
        self.log("run " + " ".join([self.python, *argv]))
        fh = open(log_to, "a") if log_to else None
        try:
            r = subprocess.run(cmd, cwd=self.root, env=env or self.env(), stdout=fh, stderr=subprocess.STDOUT if fh else None)
        finally:
            if fh:
                fh.close()
        if r.returncode != 0:
            raise StageError(f"exit {r.returncode}: {' '.join([self.python, *argv])}" + (f" (log {log_to})" if log_to else ""))

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


def register_stage(family: str, stage: str, fn: Callable[[StageContext], dict] | None = None, *, source: str = "learned",
                   flags=None, doc: str = "") -> Callable:
    """Register `fn(ctx) -> {"outputs", "metrics", ...}` as `family`'s `stage` (a decorator when `fn` is omitted).
    OPEN registry: an unknown family is registered on the spot (no flags) and a new stage name joins PIPELINE_STAGES.
    `flags` states the meaning-changing flags that apply (names from runconfig.FLAG_NAMES, or {flag: where}); None keeps
    a core pair's table and gives a new pair none. Families call this from their own pipeline module
    (`harness/pipelines/<family>.py`)."""
    def deco(fn):
        if family not in families():
            register_family(family)
        declare_stage(family, stage, flags)
        _REGISTRY[(family, stage)] = StageSpec(family, stage, fn, source, doc or (fn.__doc__ or "").strip().split("\n")[0])
        return fn
    return deco if fn is None else deco(fn)


def _load_families():
    """Import every pipeline module (registration side effects): a new family adds one module, nothing else."""
    import importlib
    import pkgutil
    import rrp.harness.pipelines as pkg
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name != "base":
            importlib.import_module(f"{pkg.__name__}.{m.name}")


def unregister_stage(family: str, stage: str) -> None:
    """Drop one registered stage and its declaration when it was not part of the core table (tests)."""
    _REGISTRY.pop((family, stage), None)
    undeclare_stage(family, stage)


def unregister_stages(family: str) -> None:
    """Drop every stage of an extension family (tests)."""
    if family in BUILTIN_FAMILIES:
        raise ValueError(f"{family} is built in")
    for k in [k for k in _REGISTRY if k[0] == family]:
        unregister_stage(*k)


# ------------------------------------------------------------------------------------ run context (14.2)
_CHILD_BOOT = "import sys; from rrp.harness.pipelines.base import child_main; sys.exit(child_main(sys.argv[1:]))"


def _factor_items(rc: RunConfig) -> list[list]:
    """The factor lists a rendered config carries: `params.latent.factors` (the probe), `latent.encoder_factors` and
    `latent.realizer_factors` (the per-site lists of a representation), `params.policy.factors` (relation policies) and
    `params.factors` (flow / bc heads)."""
    p = rc.params or {}
    out = []
    for path in (("latent", "factors"), ("latent", "encoder_factors"), ("latent", "realizer_factors"),
                 ("policy", "factors"), ("factors",)):
        cur = p
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if isinstance(cur, list) and cur:
            out.append(cur)
    return out


def _is_feat_item(it) -> bool:
    from rrp.policies.features import kinfeat
    name = it if isinstance(it, str) else (it.get("name") if isinstance(it, dict) else None)
    return name == kinfeat.FACTOR_NAME


def _feat_on(it) -> bool:
    return not (isinstance(it, dict) and it.get("control") == "off")


@functools.lru_cache(maxsize=256)
def _resolve_cached(blob: str) -> tuple:
    from rrp.policies.relations.base import resolve
    return resolve(json.loads(blob))


def _resolved_factors(rc: RunConfig) -> tuple[tuple, bool | None]:
    """(catalog FactorSpecs, feat.base_axes) of a rendered config. `feat.base_axes` is a featurizer option, not a catalog
    factor: it may be stated as a factor item (`{name: feat.base_axes}`) or as `options.kinfeat: v1`; both must agree."""
    from rrp.policies.features import kinfeat
    kf = rc.options.get("kinfeat")
    if kf is not None and kf != "v1":
        raise StageError(f"options.kinfeat {kf!r}: only v1")
    axes = kinfeat.legacy_bool(kf) if kf is not None else None
    specs = []
    for items in _factor_items(rc):
        feat = [it for it in items if _is_feat_item(it)]
        for it in feat:
            v = _feat_on(it)
            if axes is not None and axes != v:
                raise StageError(f"{rc.run_id}: {kinfeat.FACTOR_NAME} stated twice with different values "
                                 "(options.kinfeat and params factors)")
            axes = v
        rest = [it for it in items if not _is_feat_item(it)]
        if rest:
            specs.extend(_resolve_cached(json.dumps(rest, sort_keys=True, default=str)))
    return tuple(specs), axes


@dataclass
class RunContext:
    """What a stage's RunConfig configures process-wide; `restore()` undoes it (the parent stage does, a child exits)."""
    base_axes: bool | None
    specs: tuple
    _prev: bool | None = None
    applied: bool = True

    def restore(self) -> None:
        from rrp.policies.features import kinfeat
        if self.applied:
            kinfeat.set_base_axes(self._prev)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore()


def inherited_run_config() -> RunConfig | None:
    """The stage RunConfig a parent rendered for this process ($RRP_RUN_CONTEXT is only its path), or None."""
    path = os.environ.get(CONTEXT_ENV)
    return RunConfig.model_validate_json(Path(path).read_text()) if path else None


def apply_run_context(rc: RunConfig | None = None) -> RunContext:
    """Resolve the factor list of `rc` and configure the process from it: `feat.base_axes` (kinfeat's ambient value),
    and, with `options.deploy`, the deploy guard (no ground-truth factor source may be active). Called at stage entry
    in the parent (`Pipeline.run`) and in every child a stage spawns (`child_main`): the RunConfig is the one channel,
    `$RRP_KINFEAT` no longer exists. Without `rc` (a grandchild, `rrp.cli.main`) the context is the rendered config
    at `$RRP_RUN_CONTEXT`; with no variable set there is no context and nothing is applied."""
    from rrp.policies.features import kinfeat
    if rc is None:
        rc = inherited_run_config()
        if rc is None:
            return RunContext(base_axes=None, specs=(), applied=False)
    specs, axes = _resolved_factors(rc)
    if rc.options.get("deploy"):
        from rrp.policies.relations.base import assert_deployable
        assert_deployable(specs)
    return RunContext(base_axes=axes, specs=specs, _prev=kinfeat.set_base_axes(axes))


def child_main(argv: list[str]) -> int:
    """`python -c <boot> --context <run_context.json> -- [-m module | script.py] args...`: apply the stage's RunConfig,
    then run the target exactly as `python <argv>` would."""
    import runpy
    import argparse
    ap = argparse.ArgumentParser(prog="rrp stage child")
    ap.add_argument("--context", required=True)
    ap.add_argument("target", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    target = a.target[1:] if a.target[:1] == ["--"] else a.target
    if not target:
        raise SystemExit("child_main: no target")
    _load_families()                       # the config names a family: its module (or plugin) must be registered first
    from rrp.core.runconfig import load_family_plugins
    load_family_plugins()
    os.environ[CONTEXT_ENV] = str(Path(a.context).resolve())      # this child's own children apply the same context
    apply_run_context()
    if target[0] == "-m":
        sys.argv = [target[1], *target[2:]]
        runpy.run_module(target[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = list(target)
        runpy.run_path(target[0], run_name="__main__")
    return 0


def stage_versions(rc: RunConfig) -> dict:
    """The versions a stage's outputs are pinned to, derived from the rendered config alone: `pipeline`
    (PIPELINE_VERSION), `factors` (the checkpoint's combined string: relation structure hash [+ kinfeat version]; ""
    without factors) and `catalog` (hash of the registry entries the factor specs use)."""
    from rrp.policies.features import kinfeat
    specs, axes = _resolved_factors(rc)
    from rrp.policies.relations.base import stamp_versions
    cat = ""
    if specs:
        from dataclasses import asdict
        from rrp.policies.relations.base import get_factor
        defs = [asdict(get_factor(n)) for n in sorted({s.name for s in specs})]
        cat = "cat-" + hashlib.sha256(json.dumps(defs, sort_keys=True, default=repr).encode()).hexdigest()[:12]
    factors = stamp_versions({}, specs or None, [kinfeat.VERSION] if axes else []).get("factors", "")   # the checkpoint's string
    return dict(pipeline=str(PIPELINE_VERSION), factors=factors, catalog=cat)


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
            from rrp.bodies.grasp_contact import resolve
            if gc_old is not None and resolve(gc_old) != resolve(gc):
                raise StageError(f"options.grasp_contact {gc!r} contradicts $RRP_GRASP_CONTACT={gc_old!r}")
            os.environ["RRP_GRASP_CONTACT"] = resolve(gc)   # in-process code and every subprocess (ctx.env) see it
        os.chdir(root)                                   # existing code resolves artifacts/... relative to the repo
        rctx = None
        try:
            rctx = apply_run_context(rc)                 # factors -> featurizer options; children get the same via ctx.run
            res = spec.fn(ctx) or {}
        finally:
            os.chdir(cwd)
            if rctx is not None:
                rctx.restore()
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
    from rrp.core.provenance import file_digest
    return file_digest(p) if p.is_file() else None


def write_stage_manifest(ctx: StageContext, spec: StageSpec, res: dict, *, started: float) -> dict:
    from rrp.core.provenance import make_provenance, source_label
    from rrp.harness.data.manifest import write_manifest
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
    pins = stage_versions(rc)
    versions = dict(res.get("versions") or {})
    if pins["factors"]:
        versions.setdefault("factors", pins["factors"])           # a stage's measured value wins
    prov = make_provenance(src, flags=rc.flag_dict(), versions=versions,
                           notes=f"pipeline {rc.family}/{rc.stage} {rc.run_id}")
    extra = dict(schema="pipeline-manifest-1", run_id=rc.run_id, config_hash=rc.config_hash(), pins=pins,
                 runconfig=rc.model_dump(mode="json"), native_config=_safe_native(ctx),
                 inputs=inputs, outputs={k: dict(path=v, digest=_digest(ctx.root / v)) for k, v in outputs.items()},
                 metrics=res.get("metrics") or {}, started=started, finished=time.time())
    specs, _ = _resolved_factors(rc)
    if specs:
        from rrp.policies.relations.base import provenance as factor_provenance
        extra["factors"] = factor_provenance(specs)
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


def stage_main(argv=None) -> int:
    """`rrp stage run (--config FILE | --config-b64 B64) [--root R] [--no-check-inputs]` | `rrp stage list`."""
    import argparse
    import base64
    import json
    ap = argparse.ArgumentParser(prog="rrp stage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one stage of a RunConfig and write its manifest")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--config")
    g.add_argument("--config-b64")
    r.add_argument("--root", default=".")
    r.add_argument("--no-check-inputs", action="store_true")
    sub.add_parser("list", help="list the registered stages per family")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        from rrp.core.runconfig import families, load_family_plugins
        load_family_plugins()
        for fam in families():
            print(fam, " ".join(Pipeline(fam).stages()))
        return 0
    from rrp.core.runconfig import RunConfig
    d = json.loads(base64.b64decode(a.config_b64)) if a.config_b64 else json.loads(Path(a.config).read_text())
    rc = RunConfig.model_validate(d)
    try:
        body = Pipeline(rc.family).run(rc, root=Path(a.root), check_inputs=not a.no_check_inputs)
    except GateFailed as e:
        print(f"[pipeline] {e}", flush=True)
        return GATE_EXIT
    print(json.dumps(dict(run_id=body["run_id"], config_hash=body["config_hash"], metrics=body["metrics"]), default=str)[:4000])
    return 0
