"""Dataset manifests and lineage-disjoint split checks."""
from __future__ import annotations

from rrp.core.provenance import json_digest
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


# ---------------------------------------------------------------------------------------------- provenance (W3)
def dataset_provenance(episodes: list[dict], *, source: str, featurizer_version: str | None = None,
                       flags: dict | None = None, notes: str = ""):
    """Dataset-level Provenance from per-episode metas. Every episode meta written since provenance-1 carries
    `physics` (rrp.core.provenance.physics_provenance); exactly one distinct physics record is expected.
    Mixed or missing physics (e.g. resumed legacy episodes) is recorded in `notes`, never guessed."""
    from rrp.core.provenance import make_provenance
    phys = [e["physics"] for e in episodes if isinstance(e.get("physics"), dict)]
    uniq = {json.dumps(p, sort_keys=True) for p in phys}
    physics = phys[0] if len(uniq) == 1 else None
    ok = [e for e in episodes if e.get("status") != "generation_error"]
    missing = len(ok) - len(phys)
    if len(uniq) > 1:
        notes = (notes + f" MIXED physics: {len(uniq)} distinct records across episodes").strip()
    if missing > 0:
        notes = (notes + f" {missing} episode(s) without a physics record (legacy/resumed)").strip()
    return make_provenance(source, physics=physics, featurizer_version=featurizer_version, flags=flags, notes=notes)


def write_manifest(out_dir: Path, name: str, episodes: list[dict], extra: dict | None = None, *,
                   provenance=None, filename: str = "manifest.json") -> dict:
    """THE dataset manifest writer (arm, dual and legged collection). `provenance` (a Provenance or its dict)
    is stored under "provenance" and covered by manifest_hash."""
    extra = dict(extra or {})
    if provenance is not None:
        extra["provenance"] = provenance if isinstance(provenance, dict) else provenance.to_dict()
    body = dict(name=name, n_episodes=len(episodes),
                status_counts={s: sum(1 for e in episodes if e.get("status") == s) for s in
                               sorted({e.get("status") for e in episodes}, key=str)},
                episodes=episodes, **extra)
    body["manifest_hash"] = json_digest(body, indent=1, default=str)
    tmp = Path(out_dir) / (filename + ".tmp")
    tmp.write_text(json.dumps(body, indent=1, sort_keys=True, default=str))
    tmp.replace(Path(out_dir) / filename)
    return body


def read_manifest(path: Path) -> dict:
    """Read any manifest format: hashed write_manifest (arm/dual), the old unhashed legged summary
    (body/n/success/fell/episodes), a packed meta.json or a legged shard .json. Adds `provenance` (a Provenance;
    legacy=True when the file predates provenance-1) and `hash_ok` (None when the file carries no hash)."""
    from rrp.core.provenance import read_provenance
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    body = json.loads(path.read_text())
    out = dict(body)
    h = body.get("manifest_hash")
    if h is not None:
        b = {k: v for k, v in body.items() if k != "manifest_hash"}
        out["hash_ok"] = json_digest(b, indent=1, default=str) == h
    else:
        out["hash_ok"] = None
    if "n_episodes" not in out and isinstance(body.get("episodes"), list):
        out["n_episodes"] = len(body["episodes"])
    src_meta = dict(body)
    eps0 = (body.get("episodes") or [None])[0]
    if "featurizer" not in src_meta and isinstance(eps0, dict) and eps0.get("featurizer"):
        src_meta["featurizer"] = eps0["featurizer"]
    if "H" in body and "stride" in body and "provenance" not in body:     # packed meta.json: source = dataset path
        dp = body.get("dataset_provenance")
        src_meta["source"] = (dp or {}).get("source") or "unknown"
        if dp:
            src_meta["featurizer"] = dp.get("featurizer_version")
    if "source" not in src_meta:
        eps = body.get("episodes") or []
        if eps and isinstance(eps[0], dict) and eps[0].get("source"):
            src_meta["source"] = eps[0]["source"]
        elif "success" in body and "fell" in body:          # old legged manifest: scripted WaypointTeacher data
            src_meta["source"] = "scripted_teacher"
    out["provenance"] = read_provenance(src_meta)
    return out
