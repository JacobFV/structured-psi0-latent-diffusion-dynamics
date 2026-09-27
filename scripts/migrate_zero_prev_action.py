"""Make bug B-1's `zero_prev_action` explicit in every run config whose trainer reads it (W3, D-044/D-045).

Configs that omit the key were trained with the legacy default False (the teacher's previous command is visible in
node column 28). This script writes `"zero_prev_action": false` into them, so their meaning is unchanged, and new
runs can require the key (rrp.contracts.provenance.resolve_zero_prev_action raises for a new run without it).

In scope (the trainers that read the flag: rrp.learning.latent_train rep/flow/refit, rrp.learning.behavior with a
packed_dir, and SFT through the source checkpoint's config):
  configs/latent/*.json except pack-*.json (packing does not read it)
  configs/ladder/**/*.json
  configs/model/*.json that have "packed_dir"
Never touched: configs/ladder/armseed2/** (a running experiment reads them); they are only listed.
Legged/dual-arm tracker configs have no previous-action input and are out of scope.

The key is inserted textually right after the opening brace, so the rest of each file keeps its formatting; the
result is checked to parse to exactly the old dict plus the new key.

usage: python scripts/migrate_zero_prev_action.py [--apply] [--root .]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROTECTED = ("configs/ladder/armseed2/",)


def in_scope(rel: str, cfg: dict) -> bool:
    if not isinstance(cfg, dict):
        return False
    if rel.startswith("configs/latent/"):
        return not Path(rel).name.startswith("pack-")
    if rel.startswith("configs/ladder/"):
        return True
    if rel.startswith("configs/model/"):
        return "packed_dir" in cfg
    return False


def insert_key(text: str) -> str:
    m = re.match(r"\s*\{(\s*)", text)
    if not m:
        raise ValueError("not a JSON object")
    ws = m.group(1)
    sep = ws if ws else " "
    return text[:m.end(0) - len(ws)] + ws + '"zero_prev_action": false,' + sep + text[m.end(0):]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--apply", action="store_true", help="write the files (default: dry run)")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    migrated, protected, already = [], [], 0
    for p in sorted((root / "configs").rglob("*.json")):
        rel = p.relative_to(root).as_posix()
        text = p.read_text()
        try:
            cfg = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not in_scope(rel, cfg):
            continue
        if "zero_prev_action" in cfg:
            already += 1
            continue
        if rel.startswith(PROTECTED):
            protected.append(rel)
            continue
        new = insert_key(text)
        if json.loads(new) != dict(cfg, zero_prev_action=False):
            raise RuntimeError(f"{rel}: insertion changed the content")
        migrated.append(rel)
        if a.apply:
            p.write_text(new)
    armseed2 = sorted(q.relative_to(root).as_posix() for q in (root / "configs/ladder/armseed2").rglob("*.json")) \
        if (root / "configs/ladder/armseed2").exists() else []
    print(json.dumps(dict(applied=a.apply, migrated=len(migrated), already_explicit=already,
                          protected_missing=protected, files=migrated,
                          armseed2_untouched=len(armseed2)), indent=1))


if __name__ == "__main__":
    main()
