"""Release gate: 'complete' requires every requirement verified with existing evidence."""
from __future__ import annotations

from pathlib import Path

OK_STATES = {"verified", "completed", "failed_hypothesis"}   # a negative result can still be complete research


def validate_release(doc: dict, root: Path | None = None) -> dict:
    status = doc.get("status") or doc.get("overall_status")
    problems = []
    for r in doc.get("requirements", []):
        paths = r.get("evidence_paths") or r.get("artifact_paths") or []
        if r.get("status") in OK_STATES:
            if not paths:
                problems.append(f"{r['id']}: {r['status']} without evidence paths")
            elif root is not None:
                missing = [p for p in paths if not (root / p).exists()]
                if missing:
                    problems.append(f"{r['id']}: missing evidence {missing}")
        elif status == "complete":
            problems.append(f"{r['id']}: status {r.get('status')} is not complete")
    if status == "complete" and problems:
        raise ValueError("release cannot be complete: " + "; ".join(problems))
    return dict(status=status, problems=problems)
