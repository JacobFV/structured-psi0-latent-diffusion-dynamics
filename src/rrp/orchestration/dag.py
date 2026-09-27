"""`rrp run-dag dags/<lineage>.yaml`: DAG orchestration of pipeline stages (W5; replaces the chain scripts' need()/node()
functions and their .done/.failed markers).

DAG file (YAML; see dags/arm_lineage.yaml):
  name, family, track, lineage (template), label_prefix (template)
  matrix: {variant: [...], seed: [...]}        # the whole DAG is instantiated per point; node ids get "@<variant>.s<seed>"
  axis_vars: {variant: {sem: {...}}, seed: {...}}   # variables per axis value
  vars: {name: template}                        # evaluated in order after the axis vars (may use them)
  lists: {name: [...]}                          # referenced as "$name" anywhere in a node config
  defaults: {placement, retries, resources: {cpu, mem, gpu, gpu_mem, max_seconds}, max_parallel, admission_timeout_s,
             max_parallel_gpu, max_cpu}      # caps over this DAG's running nodes (declared resources)
  base: RunConfig fields shared by every node (flags!, params, options)
  nodes: {name: {stage, tag, deps, resources, placement, retries, only: {axis: [values]},
                 per: {axis: {value: <config overlay>}}, config: <RunConfig overlay>, scope: point|global}}
Inputs reference upstream nodes as "@node" or "@node:file" (the planner adds the dependency) and existing runs by id.
scope: global (W8): the node is planned ONCE, outside the matrix (id without "@..." suffix; lineage `global_lineage`,
default the DAG name; templates see only `vars` that render without axis values, and the base is rendered the same
way, so axis-dependent flags belong in the point nodes' own config). Point nodes may reference a global node
("@collect"); a global node may reference only global nodes. Use it for shared inputs (data collection, a BC
positive control, the teacher reference) that must not be duplicated per matrix point.

Execution: every node is ONE leased job through the existing broker: `rrp ops run --detach` on the host or
scripts/peer_run.sh --detach on the peer (RRP_PEER_REPO), running `python -m rrp.pipelines run --config-b64 ...`.
A node is completed iff its job exit code is 0 AND <out>/pipeline_manifest.json carries the node's config_hash.
Retries are bounded (node `retries`, default 0 as D-061); broker refusals for capacity are waited for (bounded by
admission_timeout_s) and are not attempts. State lives in ONE JSON ledger per DAG (atomic writes, lock file); a rerun
resumes idempotently (completed nodes skipped, running leases re-adopted by lease id). The runner never stops a lease.
"""
from __future__ import annotations

import base64
import copy
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rrp.contracts.runconfig import (FLAG_NAMES, FLAG_SPEC, ensure_family, SCHEMA_VERSION, RunConfig, RunIndex, overlay, render)

LEDGER_SCHEMA = "dag-ledger-1"
TERMINAL = ("completed", "failed", "blocked")
REF = re.compile(r"^@([A-Za-z0-9_\-]+)(?::(.*))?$")


class DagError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ spec + planning
@dataclass
class Resources:
    cpu: float
    mem: str
    gpu: bool = False
    gpu_mem: str | None = None
    max_seconds: int = 3600
    disk: str | None = None

    def ops_args(self) -> list[str]:
        a = ["--cpu", str(self.cpu), "--mem", self.mem, "--max-seconds", str(int(self.max_seconds))]
        if self.gpu:
            a += ["--gpu"] + (["--gpu-mem", self.gpu_mem] if self.gpu_mem else [])
        if self.disk:
            a += ["--disk", self.disk]
        return a


@dataclass
class PlannedNode:
    id: str
    name: str
    point: dict
    rc: RunConfig
    deps: list[str]
    resources: Resources
    placement: str
    retries: int
    label: str

    def summary(self) -> dict:
        return dict(id=self.id, stage=self.rc.stage, tag=self.rc.tag, placement=self.placement, deps=self.deps,
                    resources=self.resources.__dict__, retries=self.retries, label=self.label, run_id=self.rc.run_id,
                    out=self.rc.out, config_hash=self.rc.config_hash())


@dataclass
class Plan:
    name: str
    nodes: dict[str, PlannedNode]
    order: list[str]
    defaults: dict
    track: str
    source: str = ""

    def select(self, only: str | None = None, points: list[dict] | None = None) -> "Plan":
        keep = set()
        for nid, n in self.nodes.items():
            if only and not re.search(only, nid):
                continue
            if points and not any(all(str(n.point.get(k)) == str(v) for k, v in p.items()) for p in points):
                continue
            keep.add(nid)
        # a selected node needs its dependencies (they are resumed from the ledger or run first)
        stack = list(keep)
        while stack:
            for d in self.nodes[stack.pop()].deps:
                if d not in keep:
                    keep.add(d)
                    stack.append(d)
        return Plan(self.name, {k: v for k, v in self.nodes.items() if k in keep},
                    [k for k in self.order if k in keep], self.defaults, self.track, self.source)


def load_dag(path: Path | str, _seen: tuple = ()) -> dict:
    """Load a DAG file. `extends: <parent.yaml>` (path relative to this file) deep-merges this file over the parent
    (runconfig.overlay: dicts merge, lists and scalars replace, None deletes), e.g. a new data/expert set that changes
    only names, matrix and input vars of an existing DAG (dags/arm_lineage_v2.yaml)."""
    from rrp.orchestration.yamlmini import load
    p = Path(path)
    d = load(p.read_text())
    if not isinstance(d, dict):
        raise DagError(f"{p}: not a DAG file")
    parent = d.pop("extends", None)
    if parent:
        pp = (p.parent / parent).resolve()
        if pp in _seen:
            raise DagError(f"{p}: circular extends ({pp})")
        d = overlay(load_dag(pp, _seen + (p.resolve(),)), d)
    if "nodes" not in d:
        raise DagError(f"{p}: not a DAG file (no nodes)")
    return d


def _subst_lists(obj, lists: dict):
    if isinstance(obj, str) and obj.startswith("$") and obj[1:] in lists:
        return copy.deepcopy(lists[obj[1:]])
    if isinstance(obj, list):
        return [_subst_lists(x, lists) for x in obj]
    if isinstance(obj, dict):
        return {k: _subst_lists(v, lists) for k, v in obj.items()}
    return obj


def _points(matrix: dict) -> list[dict]:
    pts = [{}]
    for axis, values in (matrix or {}).items():
        pts = [dict(p, **{axis: v}) for p in pts for v in values]
    return pts


def _suffix(point: dict) -> str:
    if not point:
        return ""
    parts = []
    for k, v in point.items():
        parts.append(f"s{v}" if k == "seed" else str(v))
    return "@" + ".".join(parts)


def _env_for(spec: dict, point: dict) -> dict:
    env = dict(point)
    for axis, table in (spec.get("axis_vars") or {}).items():
        if axis in point:
            env.update((table or {}).get(point[axis]) or {})
    for k, v in (spec.get("vars") or {}).items():
        env[k] = render(v, env)
    return env


def _global_env(spec: dict) -> dict:
    """Template variables for scope-global nodes: the `vars` that render without any axis value."""
    env = {}
    for k, v in (spec.get("vars") or {}).items():
        try:
            env[k] = render(v, env)
        except Exception:      # needs an axis value: not available to global nodes
            continue
    return env


def _is_global(n: dict) -> bool:
    sc = n.get("scope", "point")
    if sc not in ("point", "global"):
        raise DagError(f"scope {sc!r} (point|global)")
    return sc == "global"


def plan_dag(spec: dict, *, source: str = "") -> Plan:
    name = spec["name"]
    family = spec["family"]
    track = spec.get("track", "pipeline")
    defaults = spec.get("defaults") or {}
    lists = spec.get("lists") or {}
    dres = defaults.get("resources") or {}
    nodes: dict[str, PlannedNode] = {}
    gspec = {k: v for k, v in spec["nodes"].items() if _is_global(v)}
    grids: dict[str, str] = {}              # global node -> run id (visible to every point)
    if gspec:
        genv = _global_env(spec)
        gl = render(spec.get("global_lineage", name), genv)
        _plan_point(spec, {}, genv, "", gl, gspec, {}, nodes, grids, family, track, defaults, lists, dres)
    for point in _points(spec.get("matrix") or {}):
        env = _env_for(spec, point)
        sfx = _suffix(point)
        lineage = render(spec.get("lineage", name), env)
        pspec = {k: v for k, v in spec["nodes"].items() if not _is_global(v)}
        _plan_point(spec, point, env, sfx, lineage, pspec, grids, nodes, {}, family, track, defaults, lists, dres)
    for nid, n in nodes.items():
        for d in n.deps:
            if d not in nodes:
                raise DagError(f"{nid}: unknown dependency {d}")
    order = _toposort(nodes)
    if len({n.rc.run_id for n in nodes.values()}) != len(nodes):
        raise DagError("two nodes derive the same output directory (give them distinct tags)")
    return Plan(name, nodes, order, defaults, track, source)


def _plan_point(spec, point, env, sfx, lineage, node_specs, grids, nodes, out_rids, family, track, defaults, lists,
                dres):
    """Plan `node_specs` at one matrix point (point={} and sfx="" for the global nodes). `grids`: run ids of the
    global nodes (referenced without suffix); `out_rids` receives this pass's run ids."""
    name = spec["name"]
    if True:
        # pass 1: run ids of every node at this point (for @refs)
        rids = {}
        raw = {}
        for nname, n in node_specs.items():
            only = n.get("only") or {}                 # {axis: [values]}: the node exists only at these points
            if any(str(point.get(ax)) not in {str(v) for v in vals} for ax, vals in only.items()):
                continue
            n = copy.deepcopy(n)
            for ax, table in (n.pop("per", None) or {}).items():   # {axis: {value: config overlay}}
                ov = (table or {}).get(point.get(ax))
                if ov:
                    n["config"] = overlay(n.get("config") or {}, ov)
            n.pop("only", None)
            n.pop("scope", None)
            n = _subst_lists(render(n, env), lists)
            raw[nname] = n
        for nname, n in raw.items():
            rc = _build_rc(spec, n, nname, point, lineage, track, family, env, lists, resolve_refs=None)
            rids[nname] = rc.run_id
        out_rids.update(rids)
        allr = {**grids, **rids}                 # a point node shadows a global node of the same name

        def dep_id(d, me):
            if d in rids:
                return d + sfx
            if d in grids:
                return d
            raise DagError(f"{me}: reference to unknown node @{d}")
        for nname, n in raw.items():
            refs: set[str] = set()
            rc = _build_rc(spec, n, nname, point, lineage, track, family, env, lists,
                           resolve_refs=lambda r: _resolve_ref(r, allr, refs, nname))
            deps = sorted({dep_id(d, nname) for d in (n.get("deps") or [])} | {dep_id(d, nname) for d in refs})
            res = Resources(**{**dres, **(n.get("resources") or {})})
            placement = n.get("placement", defaults.get("placement", "host"))
            if placement == "auto":
                placement = "peer" if res.gpu else defaults.get("cpu_placement", "host")
            if placement not in ("host", "peer"):
                raise DagError(f"{nname}: placement {placement!r}")
            lp = spec.get("global_label_prefix", name) if not point else spec.get("label_prefix", name)
            label = re.sub(r"[^A-Za-z0-9_]", "_", render(lp, env) + "_" + nname)[:60]
            nodes[nname + sfx] = PlannedNode(nname + sfx, nname, point, rc, deps, res, placement,
                                             int(n.get("retries", defaults.get("retries", 0))), label)


def _resolve_ref(r: str, rids: dict, refs: set, me: str) -> str:
    m = REF.match(r)
    if not m:
        return r
    node, file = m.group(1), m.group(2)
    if node not in rids:
        raise DagError(f"{me}: reference to unknown node @{node}")
    refs.add(node)
    return rids[node] + (f":{file}" if file else "")


def _flatten_input(v):
    """A list may mix refs and {runs, files} products; it becomes one ordered list of refs."""
    if isinstance(v, list) and any(isinstance(x, dict) for x in v):
        out = []
        for x in v:
            out += [f"{r}:{f}" for r in x["runs"] for f in x["files"]] if isinstance(x, dict) else [x]
        return out
    return v


def _map_refs(v, fn):
    if isinstance(v, str):
        return fn(v)
    if isinstance(v, list):
        return [_map_refs(x, fn) for x in v]
    if isinstance(v, dict):
        return {k: (_map_refs(x, fn) if k == "runs" else x) for k, x in v.items()}
    return v


def _build_rc(spec, n, nname, point, lineage, track, family, env, lists, resolve_refs) -> RunConfig:
    stage = n["stage"]
    base = _subst_lists(render(copy.deepcopy(spec.get("base") or {}), env), lists)
    cfg = overlay(base, n.get("config") or {})
    flags_in = cfg.pop("flags", {}) or {}
    ensure_family(family)                      # extension families (entry points) before the FLAG_SPEC lookup
    applicable = FLAG_SPEC[(family, stage)]
    flags = {}
    for f in FLAG_NAMES:
        if f in applicable:
            if f not in flags_in:
                raise DagError(f"{nname}: flag {f!r} applies to {family}/{stage} and is not stated (no silent default)")
            flags[f] = flags_in[f]
        else:
            flags[f] = None
    inputs = {k: _flatten_input(v) for k, v in (cfg.pop("inputs", {}) or {}).items()}
    if resolve_refs is not None:
        inputs = {k: _map_refs(v, resolve_refs) for k, v in inputs.items()}
    else:
        inputs = {k: _map_refs(v, lambda r: "runs/_placeholder/x" if r.startswith("@") else r) for k, v in inputs.items()}
    d = dict(schema_version=SCHEMA_VERSION, family=family, stage=stage, variant=point.get("variant", cfg.pop("variant", "na")),
             seed=int(point.get("seed", cfg.pop("seed", 0))), lineage=lineage, track=track,
             tag=n.get("tag"), inputs=inputs, flags=flags, params=cfg.pop("params", {}) or {},
             options=cfg.pop("options", {}) or {}, note=n.get("note", ""))
    cfg.pop("variant", None)
    cfg.pop("seed", None)
    if cfg:
        raise DagError(f"{nname}: unknown config keys {sorted(cfg)}")
    try:
        return RunConfig.model_validate(d)
    except Exception as e:
        raise DagError(f"{nname}{_suffix(point)}: {e}") from e


def _toposort(nodes: dict[str, PlannedNode]) -> list[str]:
    order, state = [], {}

    def visit(k, stack):
        if state.get(k) == 2:
            return
        if state.get(k) == 1:
            raise DagError(f"cycle: {' -> '.join(stack + [k])}")
        state[k] = 1
        for d in nodes[k].deps:
            visit(d, stack + [k])
        state[k] = 2
        order.append(k)
    for k in nodes:
        visit(k, [])
    return order


# ---------------------------------------------------------------------------------------------------- ledger
class Ledger:
    """One JSON file per DAG: node -> state, config_hash, run_id, attempts [{lease_id, log, placement, rc, ...}]."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = None
        self.data = json.loads(self.path.read_text()) if self.path.exists() else \
            dict(schema=LEDGER_SCHEMA, created=time.time(), nodes={})
        if self.data.get("schema") != LEDGER_SCHEMA:
            raise DagError(f"{self.path}: not a {LEDGER_SCHEMA} ledger")

    def lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = open(self.path.with_suffix(".lock"), "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DagError(f"another run-dag holds {self.path}.lock") from None

    def node(self, nid: str) -> dict:
        return self.data["nodes"].setdefault(nid, dict(state="planned", attempts=[]))

    def set(self, nid: str, **kw):
        self.node(nid).update(kw, updated=time.time())
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str))
        tmp.replace(self.path)


# ---------------------------------------------------------------------------------------------------- runners
class AdmissionRefused(RuntimeError):
    """The broker refused the lease for capacity (not a job failure; the node is retried later, bounded)."""


def _parse_ops_json(text: str) -> dict:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and '"lease_id"' in line:
            return json.loads(line)
    return {}


_CAPACITY = re.compile(r"aggregate limit|gpu owners|CapacityError|capacity|insufficient|AdmissionStopped|admission stopped", re.I)


class OpsRunner:
    """Launch through the existing broker. host: `python -m rrp.cli ops run --detach`; peer: scripts/peer_run.sh
    --detach with RRP_PEER_REPO (the synced code dir; its artifacts/ is the shared peer store)."""

    def __init__(self, root: Path, host_python: str = sys.executable, peer: str = "gb10-direct",
                 peer_repo: str | None = None, peer_python: str = "PY"):
        self.root = Path(root)
        self.host_python = host_python
        self.peer = peer
        self.peer_repo = peer_repo or os.environ.get("RRP_PEER_REPO")
        self.peer_python = peer_python

    def command(self, node: PlannedNode, python: str) -> list[str]:
        b64 = base64.b64encode(json.dumps(node.rc.model_dump(mode="json")).encode()).decode()
        return [python, "-m", "rrp.pipelines", "run", "--config-b64", b64]

    def launch_argv(self, node: PlannedNode) -> tuple[list[str], dict]:
        ops = ["--label", node.label, *node.resources.ops_args(), "--detach", "--"]
        if node.placement == "host":
            return ([self.host_python, "-m", "rrp.cli", "ops", "run", *ops, *self.command(node, self.host_python)],
                    dict(os.environ, PYTHONPATH=str(self.root / "src")))
        if not self.peer_repo or not self.peer_repo.startswith("/dev/shm/rrp-brandonin/wt/"):
            raise DagError("peer placement needs RRP_PEER_REPO=/dev/shm/rrp-brandonin/wt/<track> (never repo or wt/ladder)")
        return ([str(self.root / "scripts/peer_run.sh"), *ops, *self.command(node, self.peer_python)],
                dict(os.environ, RRP_PEER_REPO=self.peer_repo))

    def launch(self, node: PlannedNode) -> dict:
        argv, env = self.launch_argv(node)
        r = subprocess.run(argv, cwd=self.root, env=env, capture_output=True, text=True)
        info = _parse_ops_json(r.stderr + "\n" + r.stdout)
        if not info.get("lease_id"):
            if _CAPACITY.search(r.stderr + r.stdout):
                raise AdmissionRefused((r.stderr or r.stdout)[-300:])
            raise RuntimeError(f"launch failed (exit {r.returncode}): {(r.stderr or r.stdout)[-500:]}")
        return dict(lease_id=info["lease_id"], log=info.get("log"), unit=info.get("unit"), placement=node.placement)

    def _sh(self, placement: str, script: str) -> str:
        if placement == "host":
            return subprocess.run(["bash", "-c", script], capture_output=True, text=True).stdout
        return subprocess.run(["ssh", self.peer, script], capture_output=True, text=True).stdout

    def poll(self, handle: dict) -> int | None:
        """Exit code of a launched job, None while it runs; -1 if the unit is gone without an exit code."""
        rc_file = str(Path(handle["log"]).with_suffix(".rc"))
        unit = handle.get("unit") or f"rrp-job-{handle['lease_id']}.service"
        out = self._sh(handle["placement"], f"(cat {shlex.quote(rc_file)} && echo) 2>/dev/null || echo NORC; "
                                            f"systemctl --user is-active {shlex.quote(unit)} 2>/dev/null || true")
        lines = out.split()
        if lines and lines[0] != "NORC":
            try:
                return int(lines[0])
            except ValueError:
                return -1
        state = lines[-1] if lines else ""
        if state in ("active", "activating", "reloading"):
            return None
        handle["_gone"] = handle.get("_gone", 0) + 1
        if handle["_gone"] < 3:                          # grace: the rc file is written just before the unit ends
            return None
        res = self._sh(handle["placement"], f"systemctl --user show -p Result --value {shlex.quote(unit)} 2>/dev/null").strip()
        handle["unit_result"] = res or "unknown"         # e.g. oom-kill (recorded in the ledger attempt)
        return -1

    def manifest(self, node: PlannedNode) -> dict | None:
        if node.placement == "host":
            p = self.root / node.rc.out / "pipeline_manifest.json"
            return json.loads(p.read_text()) if p.exists() else None
        txt = self._sh("peer", f"cat {shlex.quote(self.peer_repo + '/' + node.rc.out + '/pipeline_manifest.json')} 2>/dev/null")
        return json.loads(txt) if txt.strip() else None


# ---------------------------------------------------------------------------------------------------- executor
@dataclass
class Executor:
    plan: Plan
    ledger: Ledger
    runner: OpsRunner
    max_parallel: int = 4
    max_parallel_gpu: int | None = None     # cap on running GPU nodes (a share of a shared GPU; W8)
    max_cpu: float | None = None            # cap on the summed declared CPU of running nodes
    poll_s: float = 30.0
    admission_timeout_s: float = 10800.0
    log: callable = field(default=lambda m: print(f"{time.strftime('%F %T')} [run-dag] {m}", flush=True))
    sleep: callable = time.sleep

    def _check_ledger(self):
        for nid in self.plan.order:
            n, e = self.plan.nodes[nid], self.ledger.node(nid)
            h = n.rc.config_hash()
            if e.get("config_hash") and e["config_hash"] != h and e["state"] != "planned":
                raise DagError(f"{nid}: the ledger ran config {e['config_hash']} but the DAG now gives {h}; "
                               f"use --reset {nid} (outputs go to the same dir {n.rc.out})")
            e.setdefault("config_hash", h)
            e.update(run_id=n.rc.run_id, out=n.rc.out, stage=n.rc.stage, placement=n.placement)
        self.ledger.save()

    def run(self) -> dict:
        self._check_ledger()
        waiting_since: dict[str, float] = {}
        while True:
            running = [k for k in self.plan.order if self.ledger.node(k)["state"] == "running"]
            for nid in running:
                self._poll(nid)
            self._block_dependants()
            running = [k for k in self.plan.order if self.ledger.node(k)["state"] == "running"]
            ready = [k for k in self.plan.order if self.ledger.node(k)["state"] == "planned"
                     and all(self.ledger.node(d)["state"] == "completed" for d in self.plan.nodes[k].deps)]
            for nid in ready:
                if len(running) >= self.max_parallel:
                    break
                if not self._fits(nid, running):
                    continue
                if self._adopt_existing(nid):
                    continue
                try:
                    h = self.runner.launch(self.plan.nodes[nid])
                except AdmissionRefused as e:
                    if nid not in waiting_since:
                        self.log(f"{nid}: broker refused admission ({str(e).strip().splitlines()[-1][:160]}); waiting "
                                 f"(bounded {self.admission_timeout_s:.0f} s, not an attempt)")
                    t0 = waiting_since.setdefault(nid, time.time())
                    if time.time() - t0 > self.admission_timeout_s:
                        self._fail(nid, f"admission refused for {self.admission_timeout_s:.0f} s: {e}", final=True)
                    continue
                except Exception as e:  # launch error (not a job failure of a started lease)
                    self._fail(nid, f"launch error: {e}", final=True)
                    continue
                waiting_since.pop(nid, None)
                e = self.ledger.node(nid)
                e["attempts"].append(dict(h, started=time.time()))
                self.ledger.set(nid, state="running")
                self.log(f"{nid}: launched lease {h['lease_id']} on {h['placement']}")
                running.append(nid)
            states = [self.ledger.node(k)["state"] for k in self.plan.order]
            if all(s in TERMINAL for s in states):
                break
            if not running and not ready and not waiting_since:
                break          # nothing can progress (should not happen: blocked handles failed deps)
            self.sleep(self.poll_s)
        summary = {s: sum(1 for k in self.plan.order if self.ledger.node(k)["state"] == s)
                   for s in ("completed", "failed", "blocked", "planned", "running")}
        self.log(f"done: {summary}")
        return summary

    def _fits(self, nid: str, running: list[str]) -> bool:
        """Per-DAG resource caps (declared resources of the running nodes; the broker still enforces the host/peer
        limits). A node larger than max_cpu alone may still run when nothing else is running."""
        r = self.plan.nodes[nid].resources
        if r.gpu and self.max_parallel_gpu is not None and \
                sum(self.plan.nodes[k].resources.gpu for k in running) >= self.max_parallel_gpu:
            return False
        if self.max_cpu is not None and running and \
                sum(self.plan.nodes[k].resources.cpu for k in running) + r.cpu > self.max_cpu:
            return False
        return True

    def _adopt_existing(self, nid: str) -> bool:
        """Outputs already complete (manifest with this config_hash, e.g. run by hand): mark completed, no lease."""
        m = self.runner.manifest(self.plan.nodes[nid])
        if m and m.get("config_hash") == self.plan.nodes[nid].rc.config_hash():
            self.ledger.set(nid, state="completed", adopted=True, metrics=m.get("metrics"))
            self.log(f"{nid}: adopted existing outputs")
            return True
        return False

    def _poll(self, nid: str):
        e = self.ledger.node(nid)
        att = e["attempts"][-1]
        rc = self.runner.poll(att)
        if rc is None:
            return
        att.update(rc=rc, finished=time.time())
        node = self.plan.nodes[nid]
        if rc == 0:
            m = self.runner.manifest(node)
            if m and m.get("config_hash") == node.rc.config_hash():
                self.ledger.set(nid, state="completed", metrics=m.get("metrics"))
                self.log(f"{nid}: completed (lease {att['lease_id']})")
                return
            reason = "exit 0 but no manifest with the node's config_hash"
        else:
            reason = f"exit code {rc}" + (f" unit {att['unit_result']}" if att.get("unit_result") else "") + \
                f" (log {att.get('log')})"
        attempts = len(e["attempts"])
        if attempts <= node.retries:
            self.ledger.set(nid, state="planned", last_error=reason)
            self.log(f"{nid}: {reason}; retry {attempts}/{node.retries}")
        else:
            self._fail(nid, reason, final=True)

    def _fail(self, nid, reason, final):
        self.ledger.set(nid, state="failed", last_error=reason)
        self.log(f"{nid}: FAILED {reason}")

    def _block_dependants(self):
        changed = True
        while changed:
            changed = False
            for k in self.plan.order:
                e = self.ledger.node(k)
                if e["state"] == "planned" and any(self.ledger.node(d)["state"] in ("failed", "blocked")
                                                   for d in self.plan.nodes[k].deps):
                    self.ledger.set(k, state="blocked", last_error="dependency failed")
                    changed = True


def default_ledger_path(root: Path, plan: Plan) -> Path:
    return Path(root) / "artifacts/runs" / plan.track / "_dags" / plan.name / "ledger.json"


def format_plan(plan: Plan, ledger: Ledger | None = None, runner: OpsRunner | None = None) -> str:
    lines = [f"DAG {plan.name}: {len(plan.nodes)} nodes (source {plan.source})"]
    for nid in plan.order:
        n = plan.nodes[nid]
        st = ledger.node(nid)["state"] if ledger and nid in ledger.data["nodes"] else "planned"
        r = n.resources
        res = f"cpu={r.cpu} mem={r.mem}" + (f" gpu={r.gpu_mem or 'yes'}" if r.gpu else "") + f" max={r.max_seconds}s"
        lines.append(f"- {nid:34s} [{st:9s}] {n.rc.stage}{'-' + n.rc.tag if n.rc.tag else ''} @{n.placement} {res}"
                     f" retries={n.retries} label={n.label}")
        lines.append(f"    out={n.rc.out} hash={n.rc.config_hash()}")
        if n.deps:
            lines.append(f"    deps={','.join(n.deps)}")
        ins = n.rc.input_paths(RunIndex.load(root=runner.root) if runner else RunIndex())
        for k, v in ins.items():
            vv = v if isinstance(v, str) else f"[{len(v)} paths: {v[0]} ... {v[-1]}]"
            lines.append(f"    in.{k}={vv}")
        if runner:
            try:
                argv, _ = runner.launch_argv(n)
                lines.append("    cmd=" + " ".join(shlex.quote(a) for a in argv[:-1]) + " <config-b64>")
            except DagError as e:
                lines.append(f"    cmd=<{e}>")
    return "\n".join(lines)
