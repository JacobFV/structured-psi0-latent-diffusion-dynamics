"""Re-export: the W12 contact metrics live in rrp.data.contact_metrics (so the dual collector, which sits below the
evaluation layer, can record them for the dataset gate; same pattern as rrp.evaluation.motion_quality)."""
from rrp.data.contact_metrics import *  # noqa: F401,F403
from rrp.data.contact_metrics import (CF_VERSION, TASK_CONTACT_SPECS, _held_segments, _intersect, _trim,  # noqa: F401
                                      arm_contact_motion, contact_metrics_enabled, contact_sequence, dual_contact_motion,
                                      event_times, legged_contact_motion, receipt_latency, relative_drift, settle_latency,
                                      stance_drift, swap_hands)
