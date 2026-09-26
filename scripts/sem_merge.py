"""Merge sharded semantic-edit / arm-edit rows and write one summary.
Usage: sem_merge.py --kind semantic|arm --out DIR/summary.json --meta '{"source": ...}' rows1.jsonl rows2.jsonl ..."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rrp.evaluation import latent_semantic_edits as se  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["semantic", "arm"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--meta", default="{}")
ap.add_argument("rows", nargs="+")
a = ap.parse_args()
rows = [json.loads(l) for p in a.rows for l in open(p) if l.strip()]
summ = se.summarize_semantic(rows) if a.kind == "semantic" else se.summarize_arm(rows)
Path(a.out).write_text(json.dumps(dict(meta=dict(json.loads(a.meta), shards=a.rows, n_rows=len(rows)), summary=summ),
                                  indent=1))
print(json.dumps(summ, indent=1))
