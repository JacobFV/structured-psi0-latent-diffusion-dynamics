"""Dataset manifests and lineage-disjoint split checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def lineage_key(lineage: list[str]) -> str:
    return "|".join(sorted(lineage))


def lineage_atoms(lineages) -> set[str]:
    """Every lineage element of every item; used to detect near-duplicate leakage."""
    out = set()
    for l in lineages:
        if isinstance(l, str):
            out.add(l)
        else:
            out.update(l)
    return out


def assert_disjoint_lineages(train, test, *, family_level: bool = False, shared_ok: set[str] | None = None):
    """Raise if train/test share a body/module lineage. With family_level, any shared family
    prefix (e.g. 'procedural_arm_family/v1') also counts unless whitelisted in shared_ok."""
    tr, te = lineage_atoms(train), lineage_atoms(test)
    shared = (tr & te) - (shared_ok or set())
    if shared:
        raise ValueError(f"lineage leakage across split: {sorted(shared)}")
    if family_level:
        fam = lambda s: {x.split("/")[0] for x in s}
        f_shared = (fam(tr) & fam(te)) - (shared_ok or set())
        if f_shared:
            raise ValueError(f"family-level leakage across split: {sorted(f_shared)}")


def write_manifest(out_dir: Path, name: str, episodes: list[dict], extra: dict | None = None) -> dict:
    body = dict(name=name, n_episodes=len(episodes),
                status_counts={s: sum(1 for e in episodes if e["status"] == s) for s in
                               sorted({e["status"] for e in episodes})},
                episodes=episodes, **(extra or {}))
    txt = json.dumps(body, indent=1, sort_keys=True, default=str)
    body["manifest_hash"] = hashlib.sha256(txt.encode()).hexdigest()[:16]
    (out_dir / "manifest.json").write_text(json.dumps(body, indent=1, sort_keys=True, default=str))
    return body
