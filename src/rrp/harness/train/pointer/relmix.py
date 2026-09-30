"""The pointer trainers' batch source when a factor has `mix > 0`: `rrp.harness.data.mix.relation_batches` (docs/architecture.md
14.2) over main = demo ticks, relgen = the run's shard rows, driven by the `Scheduler` (`schedule.jsonl` / `steer.jsonl` next
to the checkpoint, `<out stem>.relgen/`).

Main rows carry the teacher's `ui.drag_to` label (`Demos.drag_labels`); relgen rows must carry the pointer family's public
inputs (`row["inputs"]`: the keys of `Demos.batch`'s public batch), because the net cannot run on a label alone. Shards written
before the relgen `PolicyInput` rows have `inputs == {}`, and relgen-shard-2 shards carry the arm family's `policy_input` + entity
ids instead of the pointer batch: both are refused by name (`FactorError`), never mixed as labels without inputs.
Only the factor loss reads the relgen rows (they have no demo chunk); the action losses read the main rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rrp.harness.data.mix import load_shard_rows, relation_batches
from rrp.harness.data.relgen import Label
from rrp.harness.data.relgen.ui import DRAG_TO_VERSION
from rrp.policies.relations.base import FactorError, get_factor


def needs_drag(specs) -> bool:
    return any(s.control != "off" and get_factor(s.name).label == "drag_to" for s in specs)


def drag_label_dict(data, specs, ix):
    """The teacher's `drag_to` label of demo ticks `ix` in the `inject_labels` layout, or None when no active spec reads it."""
    if not needs_drag(specs):
        return None
    y, v = data.drag_labels(ix)
    return {"drag_to": y, "drag_to.valid": v}


class RelStream:
    """`fit`'s batch source (`stream.next() -> main sample indices`); keeps the batch's relgen rows for `supervise`."""

    def __init__(self, a, data, specs, dev):
        raw = getattr(a, "curriculum", None)
        if not raw:
            raise FactorError("a factor with mix > 0 needs --curriculum (JSON: shards = the relations_data output dir(s), "
                              "interval, ... : `relgen.curriculum.SchedulerConfig`)")
        cur = json.loads(raw) if isinstance(raw, str) else dict(raw)
        self.data, self.dev, self.specs = data, dev, specs
        self.ref = {k: v for k, v in data.batch(data.train_idx[:1])[0].items()}
        self.keys = tuple(self.ref)
        roots = [cur["shards"]] if isinstance(cur.get("shards"), str) else list(cur.get("shards") or [])
        self._check_inputs(specs, roots, self.keys)
        cfg = dict(seed=a.seed, batch_size=a.batch, curriculum=cur)
        self.out = Path(a.out).with_suffix(".relgen")
        self.rb = relation_batches(cfg, self.out, specs, batch_size=a.batch, family="pointer", main=self._main(a.batch))
        self.cur = None

    @staticmethod
    def _check_inputs(specs, roots, keys):
        for s in specs:
            if s.control == "off" or not (s.mix or 0) > 0:
                continue
            for root in roots:
                rows = load_shard_rows(s.name, get_factor(s.name).version, root)
                if rows and not all(k in rows[0]["inputs"] for k in keys):
                    raise FactorError(
                        f"{s.name}: the shards under {root} carry no pointer public inputs (row['inputs'] keys "
                        f"{sorted(rows[0]['inputs'])}, needed {list(keys)}); the pointer trainers mix relgen rows only when they "
                        "hold the pointer public batch (relgen rows are the arm family's PolicyInput)")

    def _main(self, n):
        """Main 'batches': n demo ticks per step, each a row {ix, labels}; `mixed_batches` takes the first n_main of them."""
        while True:
            ix = self.data.sample(n)
            y = drag_label_dict(self.data, self.specs, ix)
            yc, vc = (y["drag_to"].cpu().numpy(), y["drag_to.valid"].cpu().numpy()) if y else (None, None)
            yield [{"ix": int(i), "inputs": {}, "provenance": {},
                    "labels": {} if y is None else {"drag_to": Label(value=yc[k], valid=vc[k], prov="gt",
                                                                     version=DRAG_TO_VERSION)}}
                   for k, i in enumerate(ix.tolist())]

    def next(self):
        b = next(self.rb)
        self.cur = b
        rows = b["main"]
        ix = torch.tensor([r["ix"] for r in rows], dtype=torch.long, device=self.dev)
        self.n_main = len(rows)
        return ix

    def observe(self, metrics: dict) -> None:
        self.rb.observe_estimates(self.rb.step - 1, metrics)

    def relgen_batch(self):
        """(public batch, labels) of this step's relgen rows, or None when the scheduler gave them no share."""
        rows = self.cur["relgen"]
        if not rows:
            return None
        missing = sorted({k for r in rows for k in self.keys if k not in r["inputs"]})
        if missing:
            raise FactorError(f"relgen rows lack the pointer public inputs {missing} (shards without the relgen PolicyInput rows)")
        b = {k: torch.as_tensor(np.stack([np.asarray(r["inputs"][k]) for r in rows]), device=self.dev,
                                dtype=self.ref[k].dtype) for k in self.keys}
        names = sorted({n for r in rows for n in r["labels"]})
        lab = {}
        for n in names:                     # `mixed_batches` masked every row over the batch's label union: stackable
            lab[n] = torch.as_tensor(np.stack([np.asarray(r["labels"][n].value) for r in rows]), device=self.dev, dtype=torch.float32)
            lab[n + ".valid"] = torch.as_tensor(np.stack([np.asarray(r["labels"][n].valid) for r in rows]), device=self.dev)
        return b, lab
