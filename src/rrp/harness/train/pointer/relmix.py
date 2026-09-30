"""The pointer trainers' batch source when a factor has `mix > 0`: `rrp.harness.data.mix.relation_batches` (docs/architecture.md
14.2) over main = demo ticks, relgen = the run's shard rows, driven by the `Scheduler` (`schedule.jsonl` / `steer.jsonl` next
to the checkpoint, `<out stem>.relgen/`).

Main rows carry the teacher's `ui.drag_to` label (`Demos.drag_labels`); relgen rows must carry the pointer family's public
inputs (`row["inputs"]`: the keys of `Demos.batch`'s public batch), because the net cannot run on a label alone. Shards written
before the relgen `PolicyInput` rows have `inputs == {}`: they are refused by name, never mixed as labels without inputs.
Only the factor loss reads the relgen rows (they have no demo chunk); the action losses read the main rows."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
        cfg = SimpleNamespace(family="pointer", seed=a.seed, params=dict(
            factors=[s.to_dict() for s in specs], curriculum=cur, batch_size=a.batch))
        roots = [cur["shards"]] if isinstance(cur["shards"], str) else list(cur["shards"])
        self._check_inputs(specs, roots)
        self.out = Path(a.out).with_suffix(".relgen")
        self.rb = relation_batches(cfg, self.out, self._main(a.batch), batch_size=a.batch)
        self.cur = None
        self.keys = tuple(data.batch(data.train_idx[:1])[0])
        self.ref = {k: v for k, v in data.batch(data.train_idx[:1])[0].items()}

    @staticmethod
    def _check_inputs(specs, roots):
        for s in specs:
            if s.control == "off" or not (s.mix or 0) > 0:
                continue
            for root in roots:
                rows = load_shard_rows(s.name, get_factor(s.name).version, root)
                if rows and not rows[0]["inputs"]:
                    raise FactorError(
                        f"{s.name}: the shards under {root} carry no policy inputs (row['inputs'] == {{}}); the pointer trainers "
                        "mix relgen rows only when they hold the pointer public batch (the relgen PolicyInput rows)")

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
        lab = self.cur["labels"].get("ctx", {})
        return b, {k: v[self.n_main:] for k, v in lab.items()}
