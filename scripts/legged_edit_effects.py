"""Paired edit effects vs the unedited control on the same seeds (effects.json in DIR).
Thin wrapper over rrp.evaluation.legged_summaries.edit_effects_main (moved there, D-126; same CLI, stdout and output files)."""
import sys
from pathlib import Path

# this checkout's library first (the pre-move script needed no PYTHONPATH and ran its own code)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rrp.evaluation.legged_summaries import edit_effects_main  # noqa: E402

if __name__ == "__main__":
    edit_effects_main(sys.argv[1:])
