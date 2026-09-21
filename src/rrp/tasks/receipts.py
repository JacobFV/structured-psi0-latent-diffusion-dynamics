"""Scoped event outputs with explicit provenance.

A consumer names event_id + attempt + output_name (+ optional version). The store NEVER
resolves "the latest value of that type". Separate failure codes: missing, wrong_type,
foreign_same_type, stale, invalid.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

from rrp.contracts.errors import ProvenanceError


@dataclass
class Receipt:
    event_id: str
    attempt: int
    output_name: str
    type: str
    version: int
    value: dict
    participants: list = field(default_factory=list)
    valid: bool = True
    invalid_reason: str | None = None
    created_at: float = field(default_factory=time.time)
    source_observation_id: str | None = None
    covariance_diag: list | None = None
    graph_version: int | None = None

    def key(self):
        return (self.event_id, self.attempt, self.output_name)


class ReceiptStore:
    def __init__(self):
        self._by_key: dict[tuple, list[Receipt]] = {}
        self.log: list[dict] = []

    def put(self, r: Receipt) -> Receipt:
        hist = self._by_key.setdefault(r.key(), [])
        if hist and r.version <= hist[-1].version:
            raise ProvenanceError(f"receipt version must increase for {r.key()}", code="version_not_monotonic")
        hist.append(r)
        self.log.append(dict(op="put", **{k: v for k, v in asdict(r).items() if k != "value"}))
        return r

    def invalidate(self, event_id: str, attempt: int, output_name: str, reason: str):
        for r in self._by_key.get((event_id, attempt, output_name), []):
            if r.valid:
                r.valid = False
                r.invalid_reason = reason
        self.log.append(dict(op="invalidate", event_id=event_id, attempt=attempt, output_name=output_name,
                             reason=reason))

    def latest(self, event_id, attempt, output_name) -> Receipt | None:
        h = self._by_key.get((event_id, attempt, output_name))
        return h[-1] if h else None

    def require(self, *, event_id: str, attempt: int, output_name: str, type: str,
                version: int | None = None) -> Receipt:
        hist = self._by_key.get((event_id, attempt, output_name))
        if not hist:
            foreign = [r for rs in self._by_key.values() for r in rs if r.type == type]
            if foreign:
                raise ProvenanceError(
                    f"no receipt from {event_id}#{attempt}.{output_name}; a same-type output exists from "
                    f"{foreign[-1].event_id}#{foreign[-1].attempt} but foreign provenance is not accepted",
                    code="foreign_same_type")
            raise ProvenanceError(f"missing receipt {event_id}#{attempt}.{output_name}", code="missing")
        cand = hist[-1] if version is None else next((r for r in hist if r.version == version), None)
        if cand is None:
            raise ProvenanceError(f"no version {version} of {event_id}#{attempt}.{output_name}", code="missing")
        if cand.type != type:
            raise ProvenanceError(f"{event_id}.{output_name} has type {cand.type}, expected {type}",
                                  code="wrong_type")
        if not cand.valid:
            code = "stale" if cand is not hist[-1] or cand.invalid_reason in ("superseded", "expired", "stale") \
                else "invalid"
            raise ProvenanceError(f"receipt {event_id}#{attempt}.{output_name} v{cand.version} is not valid: "
                                  f"{cand.invalid_reason}", code=code)
        return cand  # an explicitly requested older version that is still valid is accepted

    def all(self) -> list[Receipt]:
        return [r for rs in self._by_key.values() for r in rs]
