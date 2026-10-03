"""ComputerWorld pointer policies: split, teacher-demo collection, training and packet diagnostics (track pointer;
research/tracks/pointer.md "pointer policy"). `rrp train pointer <cmd> ...`:

    split    write the seed lists of a split file (--split; held-out variants are declared there: v1 lists them, v2
             (research/splits/cworld_pointer_v2.json, procedural strings) holds out a string hash bucket)
    collect  scripted-teacher demos (DART pointer noise on move ticks for a fraction of episodes) -> .npz (peer store);
             widget tables come from the live featurizer (`public_features` + `env_widget_table`), incl. z-layer, parent,
             focus rank and the ui-rel-v1 edges
    (rep / flow / bc take --factors <relations.resolve items>, stamped into the checkpoint and checked on load;
     --split-seed fixes the train / validation split independently of --seed)
    rep      E (encoder) + R (learned system 0) + P (packet probe) on demo chunks; --variant semfix | nosem
    flow     system i (rectified flow) on the frozen representation's posterior means (--target latent), or on the
             engineered encoding (--target eng, for the SCRIPTED engineered system 0)
    bc       BC baseline with the same public inputs and demos
    probe    post-hoc packet probes (and the metadata-only control) on frozen packets: UI probes for semfix vs nosem
    edit     causal packet edits: probe-guided retargeting of a received packet, realized by system 0 in closed loop
    video    labelled demo video of episodes of any pointer policy (peer: rendering)

Sources: demos are `scripted_teacher` (privileged labels: teacher target widget, destination, phase); every trained
model reads only rrp.policies.pointer.public_features. Weights and datasets stay in the peer store (never committed).

Modules: split (seed lists, no-leak guard), collect, data (`Demos`), losses, train (`fit` + rep / flow / bc / probe),
diagnostics (causal edits), video. The entry point stays `rrp.harness.train.pointer:main`.
"""
from __future__ import annotations

import argparse

from rrp.harness.train.pointer.split import SPLIT_PATH, SPLIT_PATH_V2, TASKS

# The package's public names live in submodules that import torch (`data`, `train`, ...); the pipeline modules only need
# the torch-free `split` (paths, seed lists), so they resolve lazily (PEP 562) and `import rrp.harness.train.pointer`
# stays light. `from rrp.harness.train.pointer import Demos` still works.
_LAZY = {
    "cmd_collect": "collect", "TABLE_KEYS": "data", "Demos": "data", "pointer_geometry": "data",
    "cmd_edit": "diagnostics", "action_loss": "losses", "action_metrics": "losses", "probe_loss": "losses",
    "probe_metrics": "losses", "check_no_leak": "split", "cmd_split": "split", "excluded_seeds": "split",
    "heldout_goal": "split", "load_split": "split", "cmd_bc": "train", "cmd_flow": "train", "cmd_probe": "train",
    "cmd_rep": "train", "eng_targets": "train", "fit": "train", "frozen_mu": "train", "save_checkpoint": "train",
    "FrameHook": "video", "cmd_video": "video",
}


def __getattr__(name):
    import importlib
    if name in _LAZY:
        return getattr(importlib.import_module(f"rrp.harness.train.pointer.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main(argv=None):
    from rrp.harness.train.pointer import collect, diagnostics, split, train, video
    ap = argparse.ArgumentParser(prog="rrp train pointer", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, steps=20000, lr=3e-4):
        p.add_argument("--data", nargs="+", required=True, help="collected .npz files (one per task)")
        p.add_argument("--out", required=True)
        p.add_argument("--steps", type=int, default=steps)
        p.add_argument("--batch", type=int, default=512)
        p.add_argument("--lr", type=float, default=lr)
        p.add_argument("--seed", type=int, default=0, help="init / sampling / noise seed")
        p.add_argument("--split-seed", type=int, default=0, help="train / validation episode split seed (independent "
                       "of --seed: a seed sweep keeps one held-out set)")
        p.add_argument("--device", default=None)
        p.add_argument("--log-every", type=int, default=1000)
        p.add_argument("--split", default=SPLIT_PATH_V2)

    p = sub.add_parser("split")
    p.add_argument("--split", default=SPLIT_PATH_V2)
    p.add_argument("--n-dev", type=int, default=50)
    p.add_argument("--n-sealed", type=int, default=100)
    p.add_argument("--n-heldout", type=int, default=50)
    p.set_defaults(fn=split.cmd_split)
    p = sub.add_parser("collect")
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--start", type=int, default=0, help="first training seed tried")
    p.add_argument("--dart-frac", type=float, default=0.5)
    p.add_argument("--dart-px", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default=SPLIT_PATH_V2)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=collect.cmd_collect)
    p = sub.add_parser("rep")
    common(p)
    p.add_argument("--variant", required=True, choices=("semfix", "nosem"))
    p.add_argument("--dz", type=int, default=16)
    p.add_argument("--w-sem", type=float, default=1.0)
    p.add_argument("--lv-min", type=float, default=-4.0, help="semfix: bounded probe NLL (as the arm's semfix)")
    p.add_argument("--beta", type=float, default=3e-3)
    p.add_argument("--w-xy", type=float, default=5.0)
    p.add_argument("--factors", nargs="*", default=None, metavar="SPEC",
                   help="relation factors of the net's public context (relations.resolve items: names, globs, "
                        "preset:ui, or JSON specs); default preset:none")
    p.add_argument("--curriculum", default=None, metavar="JSON",
                   help="relgen mix (a factor with mix > 0): JSON of `relgen.curriculum.SchedulerConfig` keys plus shards = the "
                        "relations_data output dir(s); batches then come from `relation_batches`")
    p.set_defaults(fn=train.cmd_rep)
    p = sub.add_parser("flow")
    common(p, steps=30000)
    p.add_argument("--target", choices=("latent", "eng"), default="latent")
    p.add_argument("--representation", help="rep checkpoint (target latent)")
    p.add_argument("--eng-version", default="cw_pointer_eng.v1", choices=("cw_pointer_eng.v1", "cw_pointer_eng.v2"),
                   help="engineered packet encoding of --target eng (v2: 7-bit key code)")
    p.add_argument("--key-head", choices=("free", "copy"), default="free",
                   help="copy: instruction-copy mixture key head (architecture 14.6; needs --eng-version v2)")
    p.add_argument("--w-sem", type=float, default=0.5)
    p.add_argument("--factors", nargs="*", default=None, metavar="SPEC",
                   help="relation factors of the net's public context (relations.resolve items: names, globs, "
                        "preset:ui, or JSON specs); default preset:none")
    p.add_argument("--curriculum", default=None, metavar="JSON",
                   help="relgen mix (a factor with mix > 0): JSON of `relgen.curriculum.SchedulerConfig` keys plus shards = the "
                        "relations_data output dir(s); batches then come from `relation_batches`")
    p.set_defaults(fn=train.cmd_flow)
    p = sub.add_parser("bc")
    common(p, steps=50000)
    p.add_argument("--w-xy", type=float, default=5.0)
    p.add_argument("--key-head", choices=("free", "copy"), default="free",
                   help="copy: instruction-copy mixture key head (architecture 14.6)")
    p.add_argument("--factors", nargs="*", default=None, metavar="SPEC",
                   help="relation factors of the net's public context (relations.resolve items: names, globs, "
                        "preset:ui, or JSON specs); default preset:none")
    p.add_argument("--curriculum", default=None, metavar="JSON",
                   help="relgen mix (a factor with mix > 0): JSON of `relgen.curriculum.SchedulerConfig` keys plus shards = the "
                        "relations_data output dir(s); batches then come from `relation_batches`")
    p.set_defaults(fn=train.cmd_bc)
    p = sub.add_parser("probe")
    common(p, steps=8000, lr=1e-3)
    p.add_argument("--representation", required=True)
    p.add_argument("--flow", help="probe packets generated by this flow instead of E means")
    p.set_defaults(fn=train.cmd_probe)
    p = sub.add_parser("edit")
    common(p, steps=0)
    p.add_argument("--representation", required=True)
    p.add_argument("--flow", required=True)
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--probe-steps", type=int, default=6000)
    p.add_argument("--edit-steps", type=int, default=60)
    p.add_argument("--edit-lr", type=float, default=0.05)
    p.add_argument("--anchor", type=float, default=0.1)
    p.set_defaults(fn=diagnostics.cmd_edit)
    p = sub.add_parser("video")
    p.add_argument("--policy", required=True, help="NAME or NAME=JSON kwargs (as rrp eval)")
    p.add_argument("--episodes", nargs="+", required=True, help="task@seed ...")
    p.add_argument("--caption", default="")
    p.add_argument("--env-kw", action="append", default=[], help="k=v env kwargs as rrp eval (e.g. strings=procedural)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=video.cmd_video)
    a = ap.parse_args(argv)
    return a.fn(a)
