"""Several training runs in ONE process (`compute.seeds_per_job`, docs/architecture.md 14.4 / research/tracks/compute.md).

A trainer exposes its loop as a generator (`_flow_steps`, `_representation_steps`): it yields once after setup and once
after every optimizer step and returns its result dict. `drain` runs one to completion (the single-run trainer, bitwise the
pre-generator loop); `run_members` runs n of them in the same process. Every member keeps its OWN model, optimizer,
scheduler, python/torch generators, data stream and output directory; nothing is shared but the process (CUDA context, the
imported code, the page-cached packed arrays).

mode="lockstep" (default): members step round-robin on one thread; the process-global RNG streams (torch CPU/CUDA, python
`random`, numpy) are swapped in and out around each member's step, so a member draws exactly the numbers it would draw
alone -- per-seed results are BITWISE the independent runs' (CPU; on GPU up to kernel nondeterminism).
mode="threads": after a serial setup (same swap) every member runs on its own thread and CUDA stream, so kernels of
different members overlap. Torch releases the GIL inside ops, but the global RNG cannot be swapped between threads:
the outcome is exact only while no member draws from a global stream inside its loop (flow policies draw from their own
`torch.Generator`; dropout 0). `info["rng_exact"]` records whether the global state stayed untouched; if it did not, the
members are statistically (not bitwise) equivalent to independent runs.
"""
from __future__ import annotations

import random
import threading
from dataclasses import dataclass


@dataclass
class Outcome:
    ok: bool
    value: object = None          # the member's result dict, or the exception that stopped it

    def unwrap(self):
        if not self.ok:
            raise self.value
        return self.value


def drain(gen):
    """Run a stepping generator to its return value."""
    try:
        while True:
            next(gen)
    except StopIteration as s:
        return s.value


def _get_rng() -> dict:
    import numpy as np
    import torch
    st = dict(py=random.getstate(), np=np.random.get_state(), cpu=torch.get_rng_state())
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng(st: dict) -> None:
    import numpy as np
    import torch
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    torch.set_rng_state(st["cpu"])
    if "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


def _same(a: dict, b: dict) -> bool:
    import numpy as np
    import torch
    return (a["py"] == b["py"] and torch.equal(a["cpu"], b["cpu"]) and all(torch.equal(x, y) for x, y in zip(a.get("cuda", []), b.get("cuda", [])))
            and all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(a["np"], b["np"])))


def run_members(gens: list, mode: str = "lockstep") -> tuple[list[Outcome], dict]:
    """Run stepping generators together; a member that raises stops alone (its Outcome carries the exception, the others
    continue). Returns (outcomes in member order, info)."""
    if mode not in ("lockstep", "threads"):
        raise ValueError(f"group mode {mode!r} (lockstep|threads)")
    n = len(gens)
    out: list[Outcome | None] = [None] * n
    rng: list[dict | None] = [None] * n
    info = dict(mode=mode, members=n, rng_exact=True)

    def advance(i: int) -> bool:
        """One step of member i with its own global RNG streams; False when it has finished or failed."""
        if rng[i] is not None:
            _set_rng(rng[i])
        try:
            next(gens[i])
        except StopIteration as s:
            out[i] = Outcome(True, s.value)
            return False
        except Exception as e:                      # this member only; KeyboardInterrupt / SystemExit propagate
            out[i] = Outcome(False, e)
            return False
        rng[i] = _get_rng()
        return True

    alive = [i for i in range(n) if advance(i)]                      # serial setup: the init draws of each member
    if mode == "lockstep":
        while alive:
            alive = [i for i in alive if advance(i)]
        return out, info
    import torch
    before = _get_rng()

    def drive(i: int) -> None:
        try:
            if torch.cuda.is_available():
                with torch.cuda.stream(torch.cuda.Stream()):
                    out[i] = Outcome(True, drain(gens[i]))
            else:
                out[i] = Outcome(True, drain(gens[i]))
        except Exception as e:
            out[i] = Outcome(False, e)

    ts = [threading.Thread(target=drive, args=(i,), name=f"member{i}") for i in alive]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    info["rng_exact"] = _same(before, _get_rng())
    return out, info
