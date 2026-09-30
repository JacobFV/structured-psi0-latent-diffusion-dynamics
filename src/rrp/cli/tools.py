"""Tool commands: the argparse entry points of library modules, exposed as `rrp <group> <name> ARGS...` (D-140 S6b;
they were `python -m <module>` entry points). Each tool keeps its own argument parser: `rrp.cli.main` dispatches
`rrp <group> <name> ...` here BEFORE the top-level parser, so every argument reaches the tool unchanged and in order.
Standard library only at import time; the tool module is imported when its command runs.
"""
from __future__ import annotations

import importlib

# (group, name) -> ("module:function", help). The function takes argv (list[str]) and returns an exit code or None.
TOOLS: dict[tuple[str, str], tuple[str, str]] = {
    # data collection
    ("data", "collect-dual"): ("rrp.harness.data.collect_dual:main", "dual-arm teacher data collection"),
    ("data", "legged-collect"): ("rrp.harness.data.legged_collect:main", "legged waypoint teacher data (10 Hz)"),
    ("data", "legged-latent-collect"): ("rrp.harness.data.legged_latent_collect:main",
                                        "tick-level legged teacher data for the latent path (50 Hz)"),
    ("data", "psi0-features"): ("rrp.policies.psi0.data:cache_features", "Psi0 frozen-VLM feature cache of a SIMPLE task (peer)"),
    ("data", "vlm-features"): ("rrp.harness.data.vlm_features:main", "rendered-image VLM feature cache (peer)"),
    # training
    ("train", "legged-latent"): ("rrp.harness.train.legged_latent_train:main", "legged Stage A / flow / probe {rep,flow,probe}"),
    ("train", "legged-bc"): ("rrp.harness.train.legged_bc:main", "legged BC positive control"),
    ("train", "legged-dagger"): ("rrp.harness.train.legged_dagger:main", "legged DAgger {collect,gate,refit}"),
    ("train", "tracker-cpu"): ("rrp.harness.train.tracker_training:main", "legged tracker PPO on CPU MuJoCo"),
    ("train", "tracker-warp"): ("rrp.harness.train.warp_tracker_ppo:main", "legged / humanoid tracker PPO on MuJoCo Warp (GPU)"),
    ("train", "joint-adapt"): ("rrp.harness.train.joint_adapt:main", "joint system-i / system-0 adaptation (arm)"),
    ("train", "packet-ood"): ("rrp.harness.train.packet_ood_fit:main", "fit the packet OOD detector"),
    ("train", "vlm"): ("rrp.harness.train.vlm_train:main", "VLM policy training / evaluation"),
    ("train", "pointer"): ("rrp.harness.train.pointer:main",
                           "ComputerWorld pointer {split,collect,rep,flow,bc,probe,edit} (research/tracks/pointer.md)"),
    ("train", "synthetic"): ("rrp.harness.train.synthetic:main", "synthetic flow-SDE GRPO sanity check (not robot)"),
    # evaluation suites, audits and validators
    ("suite", "ladder"): ("rrp.harness.eval.ladder_cli:main", "arm closed-loop ladder (R0 teacher / R1 oracle / R2 generated)"),
    ("suite", "legged"): ("rrp.harness.eval.legged_latent_eval:main", "legged routes, edits, deploy options, videos"),
    ("suite", "legged-summary"): ("rrp.harness.eval.legged_summaries:ladder_summary_main", "legged ladder summaries"),
    ("suite", "legged-edit-effects"): ("rrp.harness.eval.legged_summaries:edit_effects_main", "legged edit effects"),
    ("suite", "legged-mirror-effect"): ("rrp.harness.eval.legged_summaries:mirror_effect_main", "legged mirror effect"),
    ("suite", "robustness"): ("rrp.harness.eval.robustness:main", "physics robustness sweeps"),
    ("suite", "target"): ("rrp.harness.eval.target_eval:main", "sealed target-body evaluation"),
    ("suite", "sealed-log"): ("rrp.core.sealed:main", "sealed run-once logs: list cells, record an infrastructure failure"),
    ("suite", "system2"): ("rrp.harness.eval.system2:main", "system II (VLM) harness"),
    ("suite", "teacher-quality"): ("rrp.harness.eval.teacher_quality:main", "arm teacher motion-quality gates"),
    ("suite", "dual-teacher-quality"): ("rrp.harness.eval.dual_teacher_quality:main", "dual teacher quality"),
    ("suite", "dual-validate"): ("rrp.policies.teachers.dual_validate:main", "dual teacher validation over arm pairs"),
    ("suite", "composition"): ("rrp.policies.teachers.functional_composition:main", "dual functional composition"),
    ("suite", "tracker-validation"): ("rrp.harness.eval.tracker_validation:main", "legged tracker validation (gates)"),
    ("suite", "privileged-audit"): ("rrp.harness.eval.privileged_audit:main", "privileged-information audit"),
    ("suite", "checkpoint-audit"): ("rrp.harness.eval.checkpoint_audit:main", "load-test checkpoints with provenance"),
    ("suite", "legged-catalog"): ("rrp.harness.eval.legged_catalog:main", "legged body catalogue status"),
    ("suite", "humanoid-steps"): ("rrp.harness.eval.humanoid_eval:steps_main", "C-MuJoCo h_steps evaluation (expert / blind tracker)"),
    ("suite", "humanoid-gap"): ("rrp.harness.eval.humanoid_eval:gap_main", "C-MuJoCo h_gap_sidestep evaluation (expert / blind tracker)"),
    ("suite", "humanoid-gap-smoke"): ("rrp.harness.eval.humanoid_eval:gap_smoke_main", "WarpGapEnv smoke with a tracker actor (peer GPU)"),
    ("suite", "contact-waypoint"): ("rrp.harness.eval.humanoid_eval:waypoint_main", "WaypointTeacher driving a tracker on waypoint_contact"),
    ("eval", "humanoid-transfer"): ("rrp.harness.eval.humanoid_eval:transfer_main",
                                    "humanoid transfer matrix: sealed bodies x methods x demo budgets x seeds, Level 1 apart from Level 2"),
    ("suite", "grasp-rig"): ("rrp.harness.eval.grasp_rig:main", "grasp contact rig [VERSION [GRIPPER [FRICTION]]]"),
    ("suite", "relations-curriculum"): ("rrp.cli.curriculum:suite_main",
                                        "curriculum schedule / competence-by-depth / interfering-pairs report (R11)"),
    # pipelines, visualization, ops internals
    ("stage", "run"): ("rrp.harness.pipelines.base:stage_main", "run one pipeline stage of a RunConfig (leased job)"),
    ("stage", "list"): ("rrp.harness.pipelines.base:stage_main", "list the registered stages per family"),
    ("viz", "export"): ("rrp.viz.export:main", "room data exporter"),
    ("viz", "api"): ("rrp.viz.api:main", "room API helper (read-only)"),
    ("viz", "record"): ("rrp.viz.record:main", "replay recorder (peer)"),
    ("viz", "specs"): ("rrp.viz.record_specs:main", "write the replay-recorder specs viz/specs/*.yaml"),
    ("video", "arm"): ("rrp.harness.eval.video_arm:main", "labelled arm episode videos (teacher / learned / latent)"),
    ("video", "dual"): ("rrp.harness.eval.video_dual:main", "labelled dual-arm episode videos (with per-slot packet probe)"),
    ("video", "legged"): ("rrp.harness.eval.video_legged:main", "labelled legged / humanoid teacher+tracker clips"),
    ("factors", "list"): ("rrp.policies.relations.base:cli", "relation-factor registry: list [GLOB]"),
    ("factors", "show"): ("rrp.policies.relations.base:cli", "relation-factor registry: show NAME"),
    ("factors", "presets"): ("rrp.policies.relations.base:cli", "relation-factor presets"),
    ("factors", "coverage"): ("rrp.harness.data.relgen:coverage_main", "factor x family x env coverage JSON (labels, parts, resolves)"),
    ("ops", "child"): ("rrp.ops.child:main", "internal: the leased workload runner inside rrp-job-<lease>.service"),
}
GROUP_HELP = {"factors": "relation-factor registry (docs/relations.md)", "suite": "evaluation suites, audits and validators", "stage": "pipeline stages (run-dag jobs)",
              "viz": "visualization room: export / api / record / specs",
              "video": "labelled demo-video renderers (peer; artifacts/video/INDEX.md)"}


def dispatch(argv: list[str]):
    """Run the tool named by argv[:2]; returns (True, exit code) or (False, None) when argv is not a tool command."""
    key = tuple(argv[:2])
    if len(key) < 2 or key not in TOOLS:
        return False, None
    mod, fn = TOOLS[key][0].split(":")
    args = argv[2:]
    if key[0] in ("stage", "factors"):          # one parser for all commands of the group
        args = argv[1:]
    import sys
    sys.argv[0] = f"rrp {key[0]}" if key[0] in ("stage", "factors") else f"rrp {key[0]} {key[1]}"   # argparse prog / usage lines
    return True, getattr(importlib.import_module(mod), fn)(args)


def register(sub) -> None:
    """Help entries: tools join an existing group's subcommands (data, train, ops) or a new group (suite, stage, viz)."""
    import argparse
    groups = {}
    for g, p in sub.choices.items():
        acts = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)]
        if acts:
            groups[g] = acts[0]
    for (g, name), (_, hlp) in TOOLS.items():
        if g in sub.choices and g not in groups:       # a leaf command of that name (`rrp eval`): dispatch reaches the tool anyway
            continue
        if g not in groups:
            groups[g] = sub.add_parser(g, help=GROUP_HELP.get(g, g)).add_subparsers(dest=f"{g}_cmd", required=True)
        groups[g].add_parser(name, help=hlp, add_help=False)
