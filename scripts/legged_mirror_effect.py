"""Mirror edits: paired lateral / yaw effect toward the mirrored goal side (mirror_effects.json in DIR).
Thin wrapper over rrp.evaluation.legged_summaries.mirror_effect_main (moved there, D-126; same CLI, stdout and output files)."""
import sys
from pathlib import Path

# this checkout's library first (the pre-move script needed no PYTHONPATH and ran its own code)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rrp.harness.eval.legged_summaries import mirror_effect_main  # noqa: E402

if __name__ == "__main__":
    mirror_effect_main(sys.argv[1:])
