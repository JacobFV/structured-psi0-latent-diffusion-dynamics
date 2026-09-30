"""Training-batch data mix (D-144, docs/relations.md 5.3): each batch = the scheduler's shares of composed relgen
samples + the main (task) data, fraction-exact per batch and deterministic under the seed. `allocate` is the exact
integer split; `mixed_batches` (unit R10) streams batches from the main iterator and the relgen shards;
`relation_batches` (unit R1, docs/architecture.md 14.2) is the trainer hook around it: it resolves the run's factors,
loads their shards, runs the `Scheduler` (decisions at every interval, `steer.jsonl` read, `schedule.jsonl` appended)
and stacks the labels of each batch for `batch.extra["relation_labels"]`."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Sequence

import numpy as np

from rrp.harness.data.manifest import write_manifest
from rrp.harness.data.relgen import Label, Sample

MANIFEST_SCHEMA = "relgen-shard-1"


class RelgenError(ValueError):
    """A factor names a label / transform relgen does not have, or a run's shards / schedule are unusable."""


# ------------------------------------------------------------------------------------------------ shard IO
def write_shard(factor: str, version: str, samples: Sequence[Sample], out_root: Path, *, shard_id: str) -> dict:
    """Write one shard (`<factor>.value`/`.valid` arrays per label, per sample, in one `.npz`; rows of an earlier write
    of the same `shard_id` are replaced, so a rerun is idempotent) plus its manifest
    entry (`rrp.harness.data.manifest.write_manifest`, the one manifest writer): one row per sample with its full
    provenance record (active set, label versions, transforms, env, task, seed) and the array keys that hold it."""
    d = Path(out_root) / factor / version
    d.mkdir(parents=True, exist_ok=True)
    npz_path = d / f"{shard_id}.npz"
    arrays: dict[str, np.ndarray] = {}
    rows = []
    for i, s in enumerate(samples):
        keys = []
        for lname, lab in s["labels"].items():
            vk, ok = f"{i}.{lname}.value", f"{i}.{lname}.valid"
            arrays[vk] = np.asarray(lab.value)
            arrays[ok] = np.asarray(lab.valid)
            keys.append(lname)
        rows.append({"shard": npz_path.name, "row": i, "status": "ok", "label_keys": keys, **s["provenance"]})
    np.savez_compressed(npz_path, **arrays)
    existing = read_shard_manifest(d) or {}
    kept = [r for r in existing.get("episodes") or [] if r.get("shard") != npz_path.name]   # same shard id: replaced
    all_rows = kept + rows
    return write_manifest(d, factor, all_rows, {"schema": MANIFEST_SCHEMA, "factor": factor, "version": version},
                          filename="manifest.json")


def read_shard_manifest(shard_dir: Path) -> dict | None:
    from rrp.harness.data.manifest import read_manifest
    p = Path(shard_dir) / "manifest.json"
    return read_manifest(p) if p.exists() else None


def load_shard_rows(factor: str, version: str, out_root: Path | str) -> list[Sample]:
    """Read one factor/version's shard rows back into `Sample`s (labels rehydrated as `Label`s) -- what
    `harness.data.mix.mixed_batches` pools from (unit R10)."""
    d = Path(out_root) / factor / version
    man = read_shard_manifest(d)
    if not man:
        return []
    by_shard: dict[str, np.lib.npyio.NpzFile] = {}
    out = []
    for row in man["episodes"]:
        shard = row["shard"]
        if shard not in by_shard:
            by_shard[shard] = np.load(d / shard)
        z = by_shard[shard]
        i = row["row"]
        labels = {}
        for lname in row.get("label_keys", []):
            labels[lname] = Label(value=z[f"{i}.{lname}.value"], valid=z[f"{i}.{lname}.valid"],
                                  prov=next((r["prov"] for r in row.get("labels", []) if r["label"] == lname), "gt"),
                                  version=next((r["version"] for r in row.get("labels", []) if r["label"] == lname), ""))
        prov = {k: v for k, v in row.items() if k not in ("shard", "row", "status", "label_keys")}
        out.append({"inputs": {}, "labels": labels, "provenance": prov})
    return out




def allocate(n: int, shares: dict[str, float]) -> dict[str, int]:
    """Largest-remainder split of n rows over shares (sum <= 1; the remainder goes to key "main")."""
    s = {k: max(0.0, float(v)) for k, v in shares.items()}
    tot = sum(s.values())
    if tot > 1 + 1e-9:
        raise ValueError(f"shares sum to {tot} > 1")
    s["main"] = s.get("main", 0.0) + max(0.0, 1.0 - tot)
    raw = {k: n * v for k, v in s.items()}
    out = {k: int(np.floor(v)) for k, v in raw.items()}
    rest = n - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[:rest]:
        out[k] += 1
    return out


def mask_missing_labels(samples, label_names):
    """Every sample gets every name in `label_names`: a label it already carries passes through; one it does not
    carry becomes an all-invalid placeholder shaped like that label on another sample (or, when no sample carries
    it, like any other label present, or scalar) -- so a probe's loss over a mixed batch simply ignores rows that do
    not carry it, rather than erroring on a missing key (docs/relations.md 5.3: "labels a sample does not carry are
    masked")."""
    from rrp.harness.data.relgen import Label
    ref: dict = {}
    for s in samples:
        for name, lab in s.get("labels", {}).items():
            ref.setdefault(name, lab)
    default = next(iter(ref.values())).valid.shape if ref else (1,)
    out = []
    for s in samples:
        labels = dict(s.get("labels", {}))
        for name in label_names:
            if name not in labels:
                r = ref.get(name)
                labels[name] = (Label(value=np.zeros_like(r.value), valid=np.zeros_like(r.valid), prov="none", version="")
                                if r is not None else
                                Label(value=np.zeros(default + (1,)), valid=np.zeros(default, dtype=bool),
                                      prov="none", version=""))
        out.append({**s, "labels": labels})
    return out


def mixed_batches(main, scheduler, shards, batch_size: int, rng, *, label_names=None, before=None,
                  start_step: int = 0):
    """Yield training batches mixing `main` with scheduler-chosen relgen rows (unit R10).

    `main`: an iterable of main (task) batches, each an iterable of samples (dicts with a "labels" key; plain task
    rows carry none). `scheduler`: a `relgen.curriculum.Scheduler` (`.sample(step, n)` drives the split; its shares
    ARE `allocate`'s split, so the count per key is exact). `shards`: `{factor_name: [Sample, ...]}` (e.g.
    `pipelines.relations.load_shard_rows` per factor) -- the relgen row pools `sample` draws from, keyed by the
    active set's first (sorted) factor name (composed multi-factor active sets are unit R11's `compose`; until then
    each factor's own pool stands in for its active set). `rng`: a `numpy.random.Generator` -- same seed => the same
    draw sequence (row indices), so two runs over the same shards are byte-identical.

    Each yielded batch is `{"step", "main": [...], "relgen": [...], "counts": {key: n}}`; `sum(counts.values()) ==
    batch_size` exactly (the same integer split `allocate` computes from the scheduler's current shares) and every
    row in "main" + "relgen" is masked (`mask_missing_labels`) over the union of labels present anywhere in the
    batch (plus `label_names`, when given), so downstream code can stack them uniformly. `rng` may also be a callable
    `step -> Generator` (resume-exact draws). `before(step)` runs ahead of each batch's draw (the decision hook of
    `relation_batches`); `active` lists `(sorted factor names, count)` per relgen share, the batch's composition."""
    step = start_step
    for main_batch in main:
        main_rows = list(main_batch)
        if before is not None:
            before(step)
        active = scheduler.sample(step, batch_size)
        draw = rng(step) if callable(rng) else rng
        counts = {"main": batch_size}
        rel_rows = []
        for factor_set, count in active:
            if count <= 0:
                continue
            key = sorted(factor_set)[0]
            counts[key] = count
            counts["main"] -= count
            pool = shards.get(key) or []
            if not pool:
                continue
            idx = draw.integers(0, len(pool), size=count)
            rel_rows.extend(pool[int(i)] for i in idx)
        n_main = max(0, counts["main"])
        batch_main = [main_rows[i % len(main_rows)] for i in range(n_main)] if main_rows else []
        names = set(label_names or [])
        for s in batch_main + rel_rows:
            names |= set(s.get("labels", {}))
        # mask the combined batch in one call so a main row missing a label borrows its shape from a relgen row
        # that carries it, not only from another main row (mask_missing_labels' reference shape is batch-wide)
        masked = mask_missing_labels(batch_main + rel_rows, names)
        yield {"step": step, "main": masked[:len(batch_main)], "relgen": masked[len(batch_main):], "counts": counts,
               "active": [(sorted(fs), c) for fs, c in active if c > 0]}
        step += 1


# ------------------------------------------------------------------------------------------------ trainer hook (R1)
def stack_labels(rows, family: str) -> dict:
    """`{token set: {label: [B, T.., d], label + ".valid": [B, T..]}}` for the labels of `rows` (every row carries every
    label: `mask_missing_labels`), padded with invalid positions to the longest token axis, as torch tensors, in the
    form `batch.extra["relation_labels"]` / `relation_token_sets(..., labels=)` take. The set of a label is the one
    `FAMILIES[family].labels` attaches it to; a label the family does not attach is an error."""
    import torch
    from rrp.policies.relations.base import FAMILIES, FactorError
    if family not in FAMILIES:
        raise FactorError(f"unknown net family {family!r}; families: {sorted(FAMILIES)}")
    attach = {n: st for st, names in FAMILIES[family].labels.items() for n in names}
    out: dict = {}
    for name in sorted({n for r in rows for n in r["labels"]}):
        if name not in attach:
            raise FactorError(f"label {name!r} is not one family {family!r} attaches ({sorted(attach)})")
        labs = [r["labels"][name] for r in rows]
        nd = labs[0].valid.ndim
        T = max(l.valid.shape[0] for l in labs)
        d = labs[0].value.shape[nd:]
        val = np.zeros((len(labs),) + (T,) * nd + d, dtype=np.float32)
        ok = np.zeros((len(labs),) + (T,) * nd, dtype=bool)
        for i, l in enumerate(labs):
            sl = (i,) + tuple(slice(0, n) for n in l.valid.shape)
            val[sl], ok[sl] = l.value, l.valid
        out.setdefault(attach[name], {}).update({name: torch.from_numpy(val), name + ".valid": torch.from_numpy(ok)})
    return out


def _run_specs(cfg) -> tuple:
    """The catalog FactorSpecs a RunConfig states (`params.latent.factors`, `params.policy.factors`, `params.factors`; the
    featurizer switch `feat.base_axes` is not a catalog factor). Same reading as `pipelines.base._resolved_factors`, which
    this layer cannot import."""
    from rrp.policies.features import kinfeat
    from rrp.policies.relations.base import resolve
    specs = []
    for path in (("latent", "factors"), ("policy", "factors"), ("factors",)):
        cur = cfg.params or {}
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
        items = [it for it in (cur if isinstance(cur, list) else [])
                 if (it if isinstance(it, str) else (it.get("name") if isinstance(it, dict) else None)) != kinfeat.FACTOR_NAME]
        if items:
            specs.extend(resolve(items))
    return tuple(specs)


def _curriculum(cur: dict, specs) -> tuple:
    from rrp.harness.data.relgen.curriculum import SchedulerConfig
    cur = dict(cur)
    roots = cur.pop("shards", None)
    roots = [roots] if isinstance(roots, str) else list(roots or [])
    if not roots:
        raise ValueError("params.curriculum.shards: the relations_data output dir(s) (`<run>/relgen`) is required")
    names = cur.pop("factors", None)
    live = {s.name: s for s in specs if s.control != "off"}
    if names is None:
        names = [n for n, s in live.items() if (s.mix or 0) > 0]
    bad = [n for n in names if n not in live]
    if bad:
        raise ValueError(f"curriculum.factors {bad} are not active factors of the run ({sorted(live)})")
    if not names:
        raise ValueError("nothing to schedule: give params.curriculum.factors or a factor spec with mix > 0")
    unknown = set(cur) - {f.name for f in fields(SchedulerConfig)}
    if unknown:
        raise ValueError(f"unknown curriculum keys {sorted(unknown)}")
    mixes = [live[n].mix for n in names if live[n].mix]
    if mixes:
        cur.setdefault("share_max", max(mixes))
    for k in ("promote", "parts"):
        if k in cur:
            cur[k] = tuple(tuple(x) if isinstance(x, list) else x for x in cur[k])
    return SchedulerConfig(factors=tuple(names), **cur), roots


class RelationBatches:
    """Iterator of training batches, `mixed_batches` over the run's relgen shards driven by the `Scheduler`. Each batch
    is `{"step", "main", "relgen", "counts", "active", "labels"}` with `labels` = `stack_labels(main + relgen)` (torch).
    At every decision interval (and the first step) it reads the lines appended to `<out_dir>/steer.jsonl` since the
    last decision (an unparseable line is logged as a REJECTED steer, not applied), decides, and appends one record to
    `<out_dir>/schedule.jsonl`: the `ScheduleState` plus the `steers` applied and the `metrics` observed since the last
    decision, which is everything `Scheduler.replay_records` needs to reproduce the run's composition exactly.
    `observe(step, {factor: {"competence": .., ...}})` feeds the scheduler; `observe_estimates(step, metrics)` takes
    `estimates_loss`' metrics dict directly. `start_step > 0` resumes: the scheduler is rebuilt from the records
    before that step (later ones are dropped and re-decided)."""

    def __init__(self, cfg, out_dir, main, *, batch_size: int | None = None, start_step: int = 0):
        from rrp.harness.data.relgen.curriculum import Scheduler
        from rrp.policies.relations.base import get_factor
        specs = _run_specs(cfg)
        cur = (cfg.params or {}).get("curriculum")
        if not isinstance(cur, dict):
            raise ValueError("params.curriculum (a dict) is required by relation_batches")
        self.family, self.batch_size = cfg.family, batch_size or (cfg.params or {}).get("batch_size")
        if not self.batch_size:
            raise ValueError("relation_batches needs batch_size (argument or params.batch_size)")
        self.cfg, roots = _curriculum(cur, specs)
        pools = {}
        for n in self.cfg.factors:
            v = get_factor(n).version
            pools[n] = [r for root in roots for r in load_shard_rows(n, v, root)]
            if not pools[n]:
                raise RelgenError(f"no shard rows for {n} v{v} under {roots}: run the relations_data stage first")
        names = sorted({k for rows in pools.values() for r in rows for k in r["labels"]})
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self._sched_path, self._steer_path = self.out / "schedule.jsonl", self.out / "steer.jsonl"
        records = [json.loads(l) for l in self._sched_path.read_text().splitlines() if l.strip()] \
            if self._sched_path.exists() else []
        if records and start_step == 0:
            raise RelgenError(f"{self._sched_path} already has decisions: pass start_step to resume this run")
        past = [r for r in records if r["step"] < start_step]
        if len(past) != len(records):
            self._sched_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in past))
        seed = int(cfg.seed)
        self.scheduler = Scheduler.replay_records(self.cfg, seed, past)[0] if past else Scheduler(self.cfg, seed)
        self._lines = past[-1]["steer_line"] if past else 0
        self._pending: list = []
        self.step = start_step
        self._it = mixed_batches(main, self.scheduler, pools, int(self.batch_size),
                                 lambda step: np.random.default_rng([seed, int(step), 0x52454C]),
                                 label_names=names, before=self._decide, start_step=start_step)

    def _read_steers(self):
        from rrp.harness.data.relgen.curriculum import SteerOp, parse_steer
        if not self._steer_path.exists():
            return []
        lines = self._steer_path.read_text().splitlines(keepends=True)
        lines = lines[:len(lines) - (0 if not lines or lines[-1].endswith("\n") else 1)]    # a half-written tail waits
        ops = []
        for k, line in enumerate(lines[self._lines:], self._lines + 1):
            if not line.strip():
                continue
            try:
                ops.append(parse_steer(line))
            except (ValueError, TypeError) as e:
                ops.append(SteerOp(op="invalid", reason=f"steer.jsonl line {k} unparseable: {e}"))
        self._lines = len(lines)
        return ops

    def _decide(self, step: int) -> None:
        sc = self.scheduler
        if sc.history and step % sc.cfg.interval != 0:
            return                                                    # Scheduler.replay's rule: decide on interval steps
        ops = [sc.steer(op, step) for op in self._read_steers()]
        sc.decide(step)
        sc.export(self._sched_path, extra=dict(steers=[json.loads(o.to_json()) for o in ops], metrics=self._pending,
                                               steer_line=self._lines))
        self._pending = []

    def observe(self, step: int, metrics: dict) -> None:
        metrics = json.loads(json.dumps(metrics, default=float))       # exactly what schedule.jsonl will hold
        self.scheduler.observe(step, metrics)
        self._pending.append([int(step), metrics])

    def observe_estimates(self, step: int, metrics: dict) -> None:
        """`estimates_loss` metrics -> per-factor competence: `<factor>_acc` = (hits, count) pairs, count > 0."""
        m = {k[:-4]: {"competence": h / n} for k, (h, n) in metrics.items() if k.endswith("_acc") and n > 0
             and k[:-4] in self.cfg.factors}
        if m:
            self.observe(step, m)

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        b = next(self._it)
        b["labels"] = stack_labels(b["main"] + b["relgen"], self.family)
        self.step = b["step"] + 1
        return b


def relation_batches(cfg, out_dir, main, *, batch_size: int | None = None, start_step: int = 0) -> RelationBatches:
    """The one-line trainer hook (docs/architecture.md 14.2): `batches = relation_batches(rc, out_dir, main_batches)`.
    `cfg` is the stage's `RunConfig` (`params.factors` / `policy.factors` / `latent.factors`, `params.curriculum`);
    `main` iterates the trainer's own batches (each an iterable of task rows); see `RelationBatches`."""
    return RelationBatches(cfg, out_dir, main, batch_size=batch_size, start_step=start_step)
