#!/usr/bin/env python3
"""Verify handoff checksums and JSON syntax; no network or workload execution."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


def verify_manifest(root: Path) -> list[str]:
    root = root.resolve()
    issues: list[str] = []
    try:
        lines = (root/'SHA256SUMS').read_text().splitlines()
    except OSError as exc:
        return [f'manifest unavailable: {exc}']
    seen: set[str] = set()
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split('  ', 1)
        if len(fields) != 2 or not re.fullmatch(r'[0-9a-f]{64}', fields[0]):
            issues.append(f'malformed checksum record on line {index}')
            continue
        digest, name = fields
        relative = PurePosixPath(name)
        if (relative.is_absolute() or '..' in relative.parts or not relative.parts
                or '\\' in name or name in seen):
            issues.append(f'unsafe or duplicate manifest path: {name}')
            continue
        seen.add(name)
        path = root.joinpath(*relative.parts)
        if any(p.is_symlink() for p in [path, *path.parents] if p == root or p.is_relative_to(root)):
            issues.append(f'symlink not accepted in package: {name}')
            continue
        if not path.resolve().is_relative_to(root):
            issues.append(f'unsafe resolved path: {name}')
            continue
        try:
            h = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024*1024), b''):
                    h.update(block)
            if h.hexdigest() != digest:
                issues.append(f'checksum mismatch: {name}')
        except OSError as exc:
            issues.append(f'unreadable package file {name}: {type(exc).__name__}')
    if not seen:
        issues.append('manifest contains no files')
    return issues


def no_duplicate_keys(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f'duplicate JSON key {key!r}')
        data[key] = value
    return data


def verify_json(root: Path) -> list[str]:
    issues = []
    for path in sorted(root.rglob('*.json')):
        if path.is_symlink():
            issues.append(f'JSON symlink not accepted: {path.relative_to(root)}')
            continue
        try:
            json.loads(path.read_text(), object_pairs_hook=no_duplicate_keys,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'nonfinite JSON: {value}')))
        except (OSError, ValueError) as exc:
            issues.append(f'invalid JSON {path.relative_to(root)}: {exc}')
    return issues


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    root=args.root.resolve()
    issues=verify_manifest(root)+verify_json(root)
    if issues:
        print('\n'.join(issues), file=sys.stderr)
        return 1
    print('PASS: package checksums and JSON syntax; this does not validate an implemented robot system')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
