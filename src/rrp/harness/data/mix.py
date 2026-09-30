"""Training-batch data mix (D-144, docs/relations.md 5.3): each batch = the scheduler's shares of composed relgen
samples + the main (task) data, fraction-exact per batch and deterministic under the seed. `allocate` is the exact
integer split; `mixed_batches` (unit R10) streams batches from the main iterator and the relgen shards;
`relation_batches` (unit R1, docs/architecture.md 14.2) is the trainer hook around it: it loads the run's factor shards,
runs the `Scheduler` (decisions at every interval, `steer.jsonl` read, `schedule.jsonl` appended) and hands the trainer
the shard rows of each step; `collate_rows` turns those rows into a forwardable `Batch` whose
`extra["relation_labels"]` carries their privileged labels. Shard rows carry the family collate input (the featurizer
output on the snapshot observation), so a trainer forwards them like any pack row -- but they have no action target,
so only the factors' own losses (`estimates_loss`) ever touch them."""
from __future__ import annotations

import json
import dataclasses
import itertools
from dataclasses import fields
from pathlib import Path
from typing import Sequence

import numpy as np

from rrp.harness.data.manifest import write_manifest
from rrp.harness.data.relgen import Label, Sample

MANIFEST_SCHEMA = "relgen-shard-2"


class RelgenError(ValueError):
    """A factor names a label / transform relgen does not have, or a run's shards / schedule are unusable."""


# ------------------------------------------------------------------------------------------------ shard IO
_PI_ARRAYS = ("act_node_feats", "act_node_morph_index", "relations", "pointers", "q0")


def _pi_arrays(i: int, pi) -> dict:
    """One `PolicyInput`'s arrays under `<row>.pi.*` (per-bank dicts under `<row>.pi.<field>.<bank>`)."""
    from rrp.policies.nets.batch import BANKS
    out = {f"{i}.pi.{k}": np.asarray(getattr(pi, k)) for k in _PI_ARRAYS}
    for field in ("tokens", "token_kind", "pointer_text"):
        out.update({f"{i}.pi.{field}.{b}": np.asarray(getattr(pi, field)[b]) for b in BANKS})
    return out


def _read_pi(z, i: int, meta: dict):
    from rrp.policies.features.featurizer import PolicyInput
    from rrp.policies.nets.batch import BANKS
    by_bank = {f: {b: z[f"{i}.pi.{f}.{b}"] for b in BANKS} for f in ("tokens", "token_kind", "pointer_text")}
    return PolicyInput(**by_bank, **{k: z[f"{i}.pi.{k}"] for k in _PI_ARRAYS}, meta=dict(meta))


def write_shard(factor: str, version: str, samples: Sequence[Sample], out_root: Path, *, shard_id: str) -> dict:
    """Write one shard (`<factor>.value`/`.valid` arrays per label, per sample, in one `.npz`; rows of an earlier write
    of the same `shard_id` are replaced, so a rerun is idempotent) plus its manifest
    entry (`rrp.harness.data.manifest.write_manifest`, the one manifest writer): one row per sample with its full
    provenance record (active set, label versions, transforms, env, task, seed) and the array keys that hold it.
    Every sample must carry `inputs["policy_input"]` (the featurizer's `PolicyInput` on the snapshot observation, stored
    next to its labels so the row can be forwarded) and `inputs["tokens"]["ctx"]` (the entity id of each of its
    concatenated bank tokens, what the labels are indexed by); a sample without them is a `RelgenError`."""
    d = Path(out_root) / factor / version
    d.mkdir(parents=True, exist_ok=True)
    npz_path = d / f"{shard_id}.npz"
    arrays: dict[str, np.ndarray] = {}
    rows = []
    for i, s in enumerate(samples):
        pi = s.get("inputs", {}).get("policy_input")
        ids = s.get("inputs", {}).get("tokens", {}).get("ctx")
        if pi is None or ids is None:
            raise RelgenError(f"{factor}: sample {i} has no inputs['policy_input'] / inputs['tokens']['ctx']: a shard row "
                              "must be forwardable (the featurizer output on the snapshot observation)")
        arrays.update(_pi_arrays(i, pi))
        keys = []
        for lname, lab in s["labels"].items():
            vk, ok = f"{i}.{lname}.value", f"{i}.{lname}.valid"
            arrays[vk] = np.asarray(lab.value)
            arrays[ok] = np.asarray(lab.valid)
            keys.append(lname)
        rows.append({"shard": npz_path.name, "row": i, "status": "ok", "label_keys": keys, "ctx_ids": list(ids),
                     "pi_meta": dict(pi.meta), **s["provenance"]})
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


_ROW_KEYS = ("shard", "row", "status", "label_keys", "ctx_ids", "pi_meta")


def load_shard_rows(factor: str, version: str, out_root: Path | str) -> list[Sample]:
    """Read one factor/version's shard rows back into `Sample`s: labels rehydrated as `Label`s, `inputs` =
    `{"policy_input": PolicyInput, "tokens": {"ctx": [entity id | None, ...]}}` (what `collate_rows` batches) -- what
    `harness.data.mix.mixed_batches` pools from (unit R10). A shard written before rows were forwardable is refused."""
    d = Path(out_root) / factor / version
    man = read_shard_manifest(d)
    if not man:
        return []
    if man.get("schema") != MANIFEST_SCHEMA:
        raise RelgenError(f"{d}: shard schema {man.get('schema')!r} != {MANIFEST_SCHEMA!r} (rows without a collate "
                          "input cannot be forwarded): rerun the relations_data stage")
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
        prov = {k: v for k, v in row.items() if k not in _ROW_KEYS}
        out.append({"inputs": {"policy_input": _read_pi(z, i, row["pi_meta"]), "tokens": {"ctx": list(row["ctx_ids"])}},
                    "labels": labels, "provenance": prov})
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
def collate_rows(rows, family: str):
    """Shard rows -> one forwardable `nets.batch.Batch` (`collate_inputs` over the rows' `PolicyInput`s) whose
    `extra["relation_labels"] = {"ctx": {label: [B, C.., d], label + ".valid": [B, C..]}}` holds the rows' privileged
    labels in the form `relation_token_sets(..., labels=)` takes. A label is indexed by its row's concatenated bank
    tokens (`inputs["tokens"]["ctx"]`, unpadded); the collate pads every bank to the batch maximum, so each label is
    scattered into the collated `ctx` layout (`bank_offset`), and every position a row does not label is invalid. The
    set of a label is the one `FAMILIES[family].labels` attaches it to (only `ctx` is served); a row carrying a label
    the family does not attach is an error, and a placeholder label (`prov == "none"`, `mask_missing_labels`) is absent."""
    import torch
    from rrp.policies.nets.batch import BANKS, collate_inputs
    from rrp.policies.relations.base import FAMILIES, FactorError
    if family not in FAMILIES:
        raise FactorError(f"unknown net family {family!r}; families: {sorted(FAMILIES)}")
    if not rows:
        raise RelgenError("collate_rows: no rows")
    attach = {n: st for st, names in FAMILIES[family].labels.items() for n in names}
    pis = [r["inputs"]["policy_input"] for r in rows]
    batch = collate_inputs(pis)
    C = batch.ctx_mask.shape[1]
    pos = [np.concatenate([batch.bank_offset[b] + np.arange(len(pi.tokens[b])) for b in BANKS]) for pi in pis]
    for r, p in zip(rows, pos):
        if len(r["inputs"]["tokens"]["ctx"]) != len(p):
            raise RelgenError(f"a row's ctx entity ids ({len(r['inputs']['tokens']['ctx'])}) do not match its bank "
                              f"tokens ({len(p)}): the shard was written with another featurizer")
    out: dict = {}
    for name in sorted({n for r in rows for n, l in r["labels"].items() if l.prov != "none"}):
        if attach.get(name) != "ctx":
            raise FactorError(f"label {name!r} is not one family {family!r} attaches to ctx ({sorted(attach)})")
        have = [(i, r["labels"][name]) for i, r in enumerate(rows) if name in r["labels"] and r["labels"][name].prov != "none"]
        nd, d = have[0][1].valid.ndim, have[0][1].value.shape[have[0][1].valid.ndim:]
        val = np.zeros((len(rows),) + (C,) * nd + d, dtype=np.float32)
        ok = np.zeros((len(rows),) + (C,) * nd, dtype=bool)
        for i, lab in have:
            p = pos[i]
            if lab.valid.shape != (len(p),) * nd:
                raise RelgenError(f"label {name!r} of row {i} has shape {lab.valid.shape}, its tokens are {len(p)}")
            ix = (i, p) if nd == 1 else (i, p[:, None], p[None, :])
            val[ix], ok[ix] = lab.value, lab.valid
        out.setdefault("ctx", {}).update({name: torch.from_numpy(val), name + ".valid": torch.from_numpy(ok)})
    return dataclasses.replace(batch, extra={**batch.extra, "relation_labels": out})


def _curriculum(cur: dict, specs, relgen_dir) -> tuple:
    from rrp.harness.data.relgen.curriculum import SchedulerConfig
    cur = dict(cur)
    roots = cur.pop("shards", None) or relgen_dir
    roots = [roots] if isinstance(roots, str) else list(roots or [])
    if not roots:
        raise ValueError("relgen shards: give the trainer `inputs.relgen` (`@relgen:relgen`) or params.curriculum.shards "
                         "(the relations_data output dir(s), `<run>/relgen`)")
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
    """Iterator of the trainer's shard rows, `mixed_batches` over the run's relgen shards driven by the `Scheduler`. Each
    step is `{"step", "main", "relgen", "counts", "active"}`: `counts` the exact split of `batch_size` over the main data
    and each scheduled factor (`counts["main"]` is how many rows the trainer draws from its own data), `relgen` the shard
    rows (`collate_rows` makes them a `Batch`; `main` holds what the optional `main` iterable supplied, none for a
    trainer that draws its own). At every decision interval (and the first step) it reads the lines appended to
    `<out_dir>/steer.jsonl` since the last decision (an unparseable line is logged as a REJECTED steer, not applied),
    decides, and appends one record to `<out_dir>/schedule.jsonl`: the `ScheduleState` plus the `steers` applied and
    the `metrics` observed since the last decision, which is everything `Scheduler.replay_records` needs to reproduce
    the run's composition exactly. `observe(step, {factor: {"competence": .., ...}})` feeds the scheduler;
    `observe_estimates(step, metrics)` takes `estimates_loss`' metrics dict directly, `loss(rc, specs, step)` runs
    both. `start_step > 0` resumes: the scheduler is rebuilt from the records before that step (later ones are
    dropped and re-decided). Shard rows only enter `estimates_loss`: they have no action target."""

    def __init__(self, cfg: dict, out_dir, specs, *, batch_size: int | None = None, family: str = "arm", main=None,
                 start_step: int = 0):
        from rrp.harness.data.relgen.curriculum import Scheduler
        from rrp.policies.relations.base import get_factor
        cur = cfg.get("curriculum")
        if not isinstance(cur, dict):
            raise ValueError("params.curriculum (a dict) is required by relation_batches")
        self.family, self.batch_size, self.specs = family, batch_size or cfg.get("batch_size"), tuple(specs)
        if not self.batch_size:
            raise ValueError("relation_batches needs batch_size (argument or params.batch_size)")
        self.cfg, roots = _curriculum(cur, specs, cfg.get("relgen"))
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
        seed = int(cfg["seed"])
        self.scheduler = Scheduler.replay_records(self.cfg, seed, past)[0] if past else Scheduler(self.cfg, seed)
        self._lines = past[-1]["steer_line"] if past else 0
        self._pending: list = []
        self._active: list = []
        self.step = start_step
        self._it = mixed_batches(itertools.repeat(()) if main is None else main, self.scheduler, pools,
                                 int(self.batch_size), lambda step: np.random.default_rng([seed, int(step), 0x52454C]),
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

    def draw(self, dev):
        """`(step record, shard Batch on dev | None)`: the next step's composition and its collated shard rows (None when
        the scheduler gave this step no relgen share)."""
        b = next(self)
        return b, (collate_rows(b["relgen"], self.family).to(dev) if b["relgen"] else None)

    def loss(self, rc, step: int):
        """The factor loss of a forwarded shard batch (`rc` = that forward's `RelCtx`): `estimates_loss` over the run's
        factor specs, its metrics fed to the scheduler at `step`. Returns `(loss, logs)`."""
        from rrp.policies.relations.base import estimates_loss
        el, logs, metrics = estimates_loss(rc, self.specs)
        silent = [f for f in self._active if f"probe_{f}" not in logs]
        if silent:
            raise RelgenError(f"step {step}: scheduled factor(s) {silent} added no term to the estimate loss: the net wrote "
                              "no estimate for them (a `given` source has no readout head: give the factor "
                              "`source: probe`) or the shard rows carry no label for them")
        self.observe_estimates(step, metrics)
        return el, logs

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        b = next(self._it)
        if b["counts"]["main"] < 1:
            raise RelgenError(f"step {b['step']}: the curriculum shares {b['active']} leave no main rows of "
                              f"{self.batch_size}")
        self.step = b["step"] + 1
        self._active = sorted({n for fs, c in b["active"] if c > 0 for n in fs})
        return b


def relation_batches(cfg: dict, out_dir, specs, *, batch_size: int | None = None, family: str = "arm", main=None,
                     start_step: int = 0) -> RelationBatches | None:
    """The trainer hook (docs/architecture.md 14.2): `batches = relation_batches(cfg, out_dir, specs, family=)`. `cfg` is
    the trainer's native config (`curriculum` = params.curriculum, `relgen` = the `relations_data` output dir input,
    `seed`, `batch_size`); `specs` the net's resolved factor specs; each `b, shard = batches.draw(dev)` gives the step's
    composition and the forwardable shard `Batch` (see `RelationBatches`). A config with neither `curriculum` nor
    `relgen` is a run without relation shards: None. `inputs.relgen` without `params.curriculum` (nothing says what
    share of the batch the shards get), and a prefetching loader (it draws its rows ahead of the scheduler), are errors."""
    if cfg.get("curriculum") is None:
        if cfg.get("relgen"):
            raise ValueError("inputs.relgen is set but params.curriculum is not: the shards' share of the batch is "
                             "params.curriculum (docs/relations.md 5.5)")
        return None
    if cfg.get("prefetch"):
        raise ValueError("params.curriculum with prefetch: the prefetching loader draws rows ahead of the scheduler; "
                         "drop prefetch for a relgen run")
    return RelationBatches(cfg, out_dir, specs, batch_size=batch_size, family=family, main=main, start_step=start_step)
