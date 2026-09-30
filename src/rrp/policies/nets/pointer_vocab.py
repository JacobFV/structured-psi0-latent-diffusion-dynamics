"""Sizes and vocabularies the pointer nets and the public featurization share (torch-free).

Lives beside the nets so `rrp.policies.nets.pointer` does not import the `rrp.policies.pointer` family package (which
imports the nets: the package graph must stay acyclic); `rrp.policies.pointer.spec` re-exports every name.
"""
from __future__ import annotations

KNOT_TIMES = (0.1, 0.3, 0.5, 0.7)          # the latent contract's knots (rrp.policies.nets.semantic_latent)
# D-144 R6: the pointer body's `RelBlock`s carry no relation factors by DEFAULT (single "tool" assembly, no
# cross-assembly routing to restrict). `save_checkpoint` (harness/train/pointer) records its `compat_hash` into every
# checkpoint's `versions["factors"]` for provenance / future compatibility checks, same as every other family's
# factor hash. `PolicyConfig.factors=None` resolves to this empty preset, so every existing checkpoint's `UICtx`
# stays byte-identical (docs/relations.md 3.2 zero-bias equivalence: an empty-spec `FactorSite` adds no parameters
# and `.bias()` / `.augment()` return `None` / `(None, None)`).
POINTER_FACTORS_PRESET = "none"
# D-144 R20 follow-up (archived research/tracks/rel-r20.md "lead_questions"): what `UICtx`'s widget self-attention (`ctx>ctx`
# site) offers factors -- R20's own `ui-rel-v1` edge vocab, token hiddens (for `ui.drag_to`'s bilinear pair probe)
# and the screen-geometry fields `geo.*` (R13) reads, so `factors=["preset:ui"]` and/or `geo.pos3d` / `geo.depth3d`
# resolve here. Naming a field only ADDS what a factor is allowed to read at this site (`ops._applies`); it is a
# no-op for every config that does not ask for it, same as rel-geo's own `CTX_CARRIES` extension for the arm.
UI_CARRIES = ("edges:ui-rel-v1", "hidden", "pos3d", "cam_uvd", "zlayer", "parent_id")

# Public featurization sizes: widget slots, label characters, instruction characters, own-event history, widget scalars.
NW, LC, LI, NH, WF = 80, 20, 112, 24, 11
N_ROLE, N_BOUND = 10, 16
N_SYM = 97 + 14                # symbol codes: 0 pad, 1..95 printable, 96 other, 97.. named keys (`sym_of_key`)
N_KEYCLS = 110                 # 0 = no key, 1 + KEY_VOCAB index (tests/unit/test_pointer.py checks it against the env)

# D-146 C2 (architecture 14.6): the pointer copy head. A key class (0 = none, else 1 + KEY_VOCAB index; N_KEYCLS values)
# is a `KEY_BITS`-bit +-1 code in the engineered packet `cw_pointer_eng.v2`; the instruction tokens carry the
# `N_REL` buckets of (rank inside the quoted span - typed count) that let the copy attention find "the next char".
KEY_BITS = 7
assert N_KEYCLS <= 2 ** KEY_BITS
REL_LO, REL_HI = -1, 8                          # exact buckets for r = irank - ntyped in [REL_LO, REL_HI]
N_REL = (REL_HI - REL_LO + 1) + 3               # + typed-long-ago, far-future, not-in-a-quoted-span
ENG_V2_SLOT_W, ENG_V2_KEY0 = 13, 6              # `cw_pointer_eng.v2`: slot width, first key-code field (spec.ENG_LAYOUTS)
