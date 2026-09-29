"""Summarize legged ladder rows: success / stages / Wilson 95% per file; writes <file>.summary.json next to each.
Thin wrapper over rrp.evaluation.legged_summaries.ladder_summary_main (moved there, D-126; same CLI, stdout and output files)."""
import sys
from pathlib import Path

# this checkout's library first (the pre-move script needed no PYTHONPATH and ran its own code)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rrp.harness.eval.legged_summaries import ladder_summary_main  # noqa: E402

if __name__ == "__main__":
    ladder_summary_main(sys.argv[1:])
