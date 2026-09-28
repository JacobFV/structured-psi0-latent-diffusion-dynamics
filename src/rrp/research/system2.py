"""System II (psi0 Qwen3-VL System-II backbone) as a PUBLIC task-binding provider for system i (legged track).

Role in the corrected architecture: system II reads a rendered image from a declared public scene camera plus the
operator's instruction text, and outputs TASK STRUCTURE: which detected marker is bound to the first `walk_to`
event and which to the second (the task graph's entity descriptors). That binding enters system i only through the
public task view (`LeggedSession._bind_entities` -> detector slots -> body-frame waypoint estimates + active event).
System II never sees simulator state, privileged labels or future outcomes, and its output is not a packet.

Two readouts of the frozen VLM, reported separately:
  zero_shot  : constrained answer scoring — logit of the first token of " orange" vs " cyan" after the question.
  probe      : a logistic readout trained on frozen pooled hidden states (train templates / seeds), evaluated on
               held-out templates and seeds. It is a trained head on system-II features, labelled as such.
Controls: blank image (image content removed), shuffled image (another scene).

Instruction families:
  color    : order stated by color ("walk to the cyan marker, then the orange one") — text alone suffices.
  distance : order stated by spatial relation ("first the marker nearer to you") — needs the image.
"""
from __future__ import annotations

# D-126 #31: the code moved to the library (same objects re-exported here; behaviour unchanged):
#   instructions / public task binding / declared camera -> rrp.evaluation.system2 (harness, System2Target, ...)
#   frozen VLM wrapper `System2`                          -> rrp.models.system2_vlm
from rrp.evaluation.system2 import (HELD_OUT_TEMPLATES, SYSTEM2_CAMERA, TEMPLATES, make_instruction,  # noqa: F401
                                    public_task, render_public, scenario_with_truth)
from rrp.models.system2_vlm import COLORS, QUESTION, System2  # noqa: F401
