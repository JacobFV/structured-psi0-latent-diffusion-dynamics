"""The relation-factor table (D-144): every factor, field and preset is one declarative entry here. Units add entries
ONLY in their own section (docs/relations.md section 10); operators live in `ops`, never here.
Candidate relations not yet implemented: research/relations_catalog.md (declare them with status="planned")."""
from __future__ import annotations

from rrp.bodies import g1_simple as _G
from rrp.policies.relations.base import (Algebra, FactorDef, FieldDef, ReadoutDef, register_factor, register_field,
                                         register_preset)

# ------------------------------------------------------------------ edge vocabularies (on-disk channel orders)
ARM_REL_VOCAB = ("same_node", "node_in_assembly", "actor_of", "support_of", "patient_of", "target_of", "destination_of",
                 "enables", "maintained", "output_to", "produced", "consumed_by", "pred_arg", "node_actor_of",
                 "role_in_event", "role_points_to", "kin_parent")        # PolicyInput.relations type ids (featurizer REL)
SUPPORT_REL_VOCAB = ("support",)      # `ix.force_flow`'s "edges:support-v1" (rel-geo D-144 addendum, R17 follow-up)
VOCABS = {"arm-rel-v1": ARM_REL_VOCAB, "g1-dim-rel-v1": tuple(_G.RELATIONS), "support-v1": SUPPORT_REL_VOCAB}

# ------------------------------------------------------------------ fields
for _f in (FieldDef("pos3d", 3, "position", "estimated", units="m"),
           FieldDef("cam_uvd", 3, "position", "estimated", frame="camera", units="uv[-1,1], m"),
           FieldDef("orient", 9, "orientation", "public"),
           FieldDef("normal", 3, "direction", "estimated"),
           FieldDef("entity_id", 1, "id", "public"),
           FieldDef("assembly_id", 1, "id", "public"),
           FieldDef("parent_id", 1, "id", "public"),
           FieldDef("time", 1, "time", "public", units="s")):
    register_field(_f)

# ------------------------------------------------------------------ foundation: legacy structure
_EDGE_DOC = {
    "same_node": "action node -> its own morph token (Ψ₀: command dim -> itself)",
    "node_in_assembly": "node / sensor -> its assembly token",
    "actor_of": "manipulator entity <-> event it is actor / cooperating actor of",
    "support_of": "support-role entity <-> event", "patient_of": "patient entity <-> event",
    "target_of": "target (and unmapped roles) entity <-> event", "destination_of": "destination entity <-> event",
    "enables": "event -> event it requires completed", "maintained": "event -> event it requires active",
    "output_to": "event <-> event consuming its output", "produced": "receipt <-> producing event",
    "consumed_by": "receipt <-> consuming event", "pred_arg": "predicate estimate <-> argument entity",
    "node_actor_of": "action node -> event its manipulator acts in (2-hop via bindings)",
    "role_in_event": "role-slot token <-> its event", "role_points_to": "role-slot token -> bound entity token",
    "kin_parent": "joint -> kinematic parent joint (act>act symmetric)",
    "kin_child": "command dim -> kinematic child", "same_assembly": "command dims of one assembly",
    "mirror": "command dim -> its left/right mirror",
}
for _n, _doc in _EDGE_DOC.items():
    register_factor(FactorDef(f"edge.{_n}", "1", field="edges:*", op="edge", form="bias",
                              algebra=Algebra(direction="symmetric" if _n in ("same_assembly",) else "directed"),
                              sources=("given",), params=(("edge", _n),), doc=_doc))

register_factor(FactorDef("msg.incidence", "1", field="edges:arm-rel-v1", op="inert", form="message",
                          doc="incidence messages along pointer edges (ContextEncoder.ptr); control `serialized` = the "
                              "same facts as hashed text features (the unstructured equal-information baseline)"))
register_factor(FactorDef("id.slot_handle", "1", field="entity_id", op="inert", form="embed",
                          doc="learned embedding of the PUBLIC tracker slot address on scene tokens"))
register_factor(FactorDef("route.own_assembly", "1", field="assembly_id", op="same", form="mask",
                          doc="system 0: a node reads only knots of its own assembly (params.also_key_field: key-set "
                              "bool field of always-readable knots, e.g. the legged body assembly)"))
register_factor(FactorDef("route.assembly_reads", "1", field="assembly_id", op="same", form="mask",
                          doc="system 0: a node reads knots of the assemblies allowed by params.reads [n_q, n_k]"))

register_preset("arm", [f"edge.{n}" for n in ARM_REL_VOCAB] + ["msg.incidence"])
register_preset("psi0-dims", [f"edge.{n}" for n in _G.RELATIONS])
register_preset("none", [])
# ------------------------------------------------------------------ R3: arm system 0 routing (node>knot)
register_preset("s0-arm", ["route.own_assembly"])
# rel-geo (D-144 addendum, item 3) found R5 already landed on `main` (`db8b603a`, merged while this unit was in
# flight) with its own `route.assembly_reads` preset for Ψ₀ -- `register_preset("s0-psi0", [{"name":
# "route.assembly_reads", "params": {"reads": _G.reads_table().tolist()}}])`, `_G.reads_table()` a new
# `bodies.g1_simple` function (the real `[M, M]` READS-derived table, computed at the BODY layer specifically to
# avoid the `catalog.py` -> `psi0.nets` circular import this unit had independently flagged as the reason it could
# not supply the table itself). No further action needed here: R5's `s0-psi0` fully resolves the open naming /
# wiring question this item asked rel-geo to "decide"; see research/decisions.md D-144 addendum.

# ------------------------------------------------------------------ R1 / R4 / R5 / R6: probe readouts (probes:<family>)
# arm / dual packet probe (former nets.latent_probes.PacketProbe; query order = its ALL_QUERIES order). Labels are
# privileged except `focus` (public: bound to a role of an ACTIVE event). Loss-table wiring of the legacy label keys and
# the multi-assembly variants: unit R1.
_ARM_PROBE = (
    ("visible", "entity", 1, "bce", "visible", 1.0),
    ("looking_at", "entity", 1, "mse", "gaze", 1 / 30),
    ("focused_on", "entity", 1, "bce", "focus", 1.0),
    ("held_by", "entity×asm", 1, "bce", "held", 1.0),
    ("acting_on", "entity×asm", 1, "bce", "contact", 1.0),
    ("rel_pos", "entity×asm", 6, "gauss", "rel_tcp", 10.0),
    ("observed_effect", "entity", 6, "gauss", "future_disp", 10.0),
    ("subtask", "asm", 12, "ce", "subtask", 1.0),
    ("goal_effect", "entity", 6, "gauss", "goal_effect", 10.0),
)
for _q, _addr, _out, _loss, _lab, _sc in _ARM_PROBE:
    register_factor(FactorDef(f"probe.arm.{_q}", "1", field="packet", op="inert", form="readout", label=_lab,
                              readout=ReadoutDef(_q, _addr, _out, _loss, label=_lab, scale=_sc)))
register_preset("probes:arm-packet-v1", [f"probe.arm.{q[0]}" for q in _ARM_PROBE[:-1]])
# ------------------------------------------------------------------ R5: Ψ₀ system 0 routing + probe (probes:psi0-v1)
# `route.assembly_reads` (foundation) with the G1's fixed READS table (bodies.g1_simple.reads_table; the Realizer's
# dims>knots mask, replacing the former inline `read_mask()` fill).
register_preset("s0-psi0", [{"name": "route.assembly_reads", "params": {"reads": _G.reads_table().tolist()}}])

# psi0 packet probe (former `policies.psi0.nets.PacketProbe`; query order = its old PROBE_QUERIES order). All labels
# privileged (`sim_replay`, unit R5 brief). Addressing on the generic `ReadoutProbe` (docs/relations.md 4): psi0 has
# no `entity` token set, so per-hand queries use `knot×pair` with the 2 pair slots standing for (left, right) hand and
# per-knot-only queries (`lift`, `target_pos`, `base_cmd`) use `knot×pair` too and read only pair slot 0
# (`policies.psi0.nets.run_probe`); `active_hand` / `base_disp` are per-packet (`body`). `base_cmd`'s legacy
# base-assembly-only key mask (`READS["base"]`) is not reproduced by the generic probe (ReadoutProbe has one key mask
# per forward, not per query); tracked in research/tracks/rel-r5.md.
_PSI0_PROBE = (
    ("hand_dist", "knot×pair", 2, "gauss", "hand_dist", 1.0),
    ("contact", "knot×pair", 1, "bce", "contact", 1.0),
    ("lift", "knot×pair", 1, "bce", "lift", 1.0),
    ("target_pos", "knot×pair", 6, "gauss", "target_pos", 1.0),
    ("active_hand", "body", 2, "ce", "active_hand", 1.0),
    ("base_disp", "body", 6, "gauss", "base_disp", 1.0),
    ("base_cmd", "knot×pair", 4, "gauss", "base_cmd", 1.0),
)
for _q, _addr, _out, _loss, _lab, _sc in _PSI0_PROBE:
    register_factor(FactorDef(f"probe.psi0.{_q}", "1", field="packet", op="inert", form="readout", label=_lab,
                              readout=ReadoutDef(_q, _addr, _out, _loss, label=_lab, scale=_sc)))
register_preset("probes:psi0-v1", [f"probe.psi0.{q[0]}" for q in _PSI0_PROBE])
# grasp-region affordance (roadmap #24, D-126; default OFF, unit R5): first-contact point on the target in the
# target's object frame (knot×pair, last knot only, run_probe) + 6-way contact-face class; N_FACES = 6
# (policies.psi0.nets.N_FACES) is a plain literal here (catalog.py stays torch-free; nets.py imports torch).
register_factor(FactorDef("probe.psi0.grasp_pt", "1", field="packet", op="inert", form="readout", label="grasp_pt",
                          readout=ReadoutDef("grasp_pt", "knot×pair", 6, "gauss", label="grasp_pt")))
register_factor(FactorDef("probe.psi0.grasp_face", "1", field="packet", op="inert", form="readout", label="grasp_face",
                          readout=ReadoutDef("grasp_face", "knot×pair", 6, "ce", label="grasp_face")))

# unit R4: legged system 0 routing (own assembly + always-readable body assembly) and packet probe (former
# nets.legged_latent.LeggedProbe; query order = its heads dict order). goal/disp/subtask/fall read every assembly
# (address "asm"); the caller gathers the row at the sample's body assembly index (was `code(asm_code[body_asm])`,
# now exactly `asm_in(asm_code[body_asm])` since "asm" queries every assembly with the same asm_in map).
register_preset("legged-s0", [{"name": "route.own_assembly", "params": {"also_key_field": "body"}}])
_LEGGED_PROBE = (
    ("contact", "knot×asm", 1, "bce", "contact_k", 1.0),
    ("goal", "asm", 4, "gauss", "goal", 1.0),
    ("disp", "asm", 6, "gauss", "disp", 1.0),
    ("subtask", "asm", 4, "ce", "subtask", 1.0),
    ("fall", "asm", 1, "bce", "fall", 1.0),
)
for _q, _addr, _out, _loss, _lab, _sc in _LEGGED_PROBE:
    register_factor(FactorDef(f"probe.legged.{_q}", "1", field="packet", op="inert", form="readout", label=_lab,
                              readout=ReadoutDef(_q, _addr, _out, _loss, label=_lab, scale=_sc)))
register_preset("probes:legged-v1", [f"probe.legged.{q[0]}" for q in _LEGGED_PROBE])

# unit R6: pointer packet probe (former `policies.pointer.PointerProbe`; query names kept identical to its old
# output dict -- slot, rel, phase). M = 1 (the pointer body's single "tool" assembly), so every query addresses
# `knot×asm` (the psi0/legged precedent above): the generic head reads only z's own K*M packet tokens plus fixed
# random handle codes, no widget content -- unlike the pre-R6 probe, which cross-attended the widget descriptors'
# real label/role/geometry. `slot` is a fixed NW=80-way classification of the packet's target widget by SLOT INDEX
# (stable within an episode, research/tracks/pointer.md) instead of by content: a strictly harder, more honest test
# of what `z` itself encodes, and the change that brings pointer's probe into line with every other family's
# "opaque codes only" design (nets/probes.py). No `route.*` factor: the pointer body has no cross-assembly routing
# to restrict (M = 1), so its `RelBlock`s carry the empty preset `none` (registered above) -- there is nothing here
# for R20 (UI factors) to do but ADD entries, not change this preset.
_POINTER_PROBE = (
    ("slot", "knot×asm", 80, "ce", "slot", 1.0),
    ("rel", "knot×asm", 4, "gauss", "rel", 1.0),
    ("phase", "knot×asm", 6, "ce", "phase", 1.0),
)
for _q, _addr, _out, _loss, _lab, _sc in _POINTER_PROBE:
    register_factor(FactorDef(f"probe.pointer.{_q}", "1", field="packet", op="inert", form="readout", label=_lab,
                              readout=ReadoutDef(_q, _addr, _out, _loss, label=_lab, scale=_sc)))
register_preset("probes:pointer-v1", [f"probe.pointer.{q[0]}" for q in _POINTER_PROBE])
# ------------------------------------------------------------------ R13: geometry (geo.*)
# PaPE (sqdiff+diff), rel_rot, align and order over the R12 fields (`pos3d`, `cam_uvd`, `orient`, `normal`). Every
# entry offers `probe` as a source: a `FieldReadouts` head (the foundation hook, `rrp.policies.relations.ops`) reads
# token hiddens at `params.readout_layer` and its (mean, logvar) estimate of the field replaces the field for later
# layers (docs/relations.md section 4). `given` (R12's collate-path values) stays the default everywhere the field is
# already public/estimated on disk; `gt` is the privileged label (deploy guard: unit R7/R14 supplies `StateView` /
# the label, PrivilegedInput blocks it outside training). Labels / scene-part generators (`pos3d`, `cam_uvd`,
# `orient`, `contact_normal`, `table_objects`, `camera_depth`) are unit R14's; `gen` / `label` here just name them.
register_factor(FactorDef(
    "geo.pos3d", "1", field="pos3d", op="sqdiff+diff", form="aug", sources=("given", "probe", "gt"), label="pos3d",
    gen=("table_objects",), readout=ReadoutDef(query="pos3d", address="token", out=6, loss="gauss", label="pos3d",
                                               reads="tokens"),
    params=(("p", 3), ("frame", "world"), ("readout_layer", 0)),
    doc="PaPE (arXiv 2602.01418, Eq. 9) distance/direction bias over `pos3d`, world frame"))
register_factor(FactorDef(
    "geo.depth3d", "1", field="cam_uvd", op="sqdiff+diff", form="aug", sources=("probe", "given", "gt"),
    label="cam_uvd", gen=("camera_depth",),
    readout=ReadoutDef(query="cam_uvd", address="token", out=6, loss="gauss", label="cam_uvd", reads="tokens"),
    params=(("p", 3), ("frame", "world"), ("readout_layer", 0)),
    doc="PaPE over the projected camera-frame `cam_uvd` (uv[-1,1], depth m); default source `probe` -- the "
        "un-tracked (occluded / never-observed) case this factor exists for"))
register_factor(FactorDef(
    "geo.orient", "1", field="orient", op="rel_rot", form="aug", sources=("given", "probe", "gt"), label="orient",
    gen=("table_objects",), readout=ReadoutDef(query="orient", address="token", out=18, loss="gauss", label="orient",
                                               reads="tokens"),
    params=(("readout_layer", 0),),
    doc="relative rotation <R_i B_i, R_j> bias/aug between assembly frames (`orient`, row-major 3x3)"))
register_factor(FactorDef(
    "geo.normal_align", "1", field="normal", op="align", form="aug", sources=("given", "probe", "gt"),
    label="contact_normal",
    readout=ReadoutDef(query="normal", address="token", out=6, loss="gauss", label="contact_normal", reads="tokens"),
    params=(("frame", "world"), ("readout_layer", 0)),
    doc="normal / axis alignment <b_i, n_j> (parallel / perpendicular via b); key-side `normal` field"))
register_factor(FactorDef(
    "geo.above", "1", field="pos3d", op="order", form="bias", sources=("given", "gt"),
    params=(("axis", (0.0, 0.0, 1.0)), ("margin", 0.01)),
    doc="sign((r_j - r_i) . +z) with a dead zone: above / below along the world-up (gravity) axis"))
register_preset("geo", ["geo.pos3d", "geo.depth3d", "geo.orient", "geo.normal_align", "geo.above"])
# ------------------------------------------------------------------ R15: membership / graph (id.*, kin.*)
# ------------------------------------------------------------------ R15: membership / graph (id.*, kin.*)
# entity_id / assembly_id are registered above (foundation); mirror_id is new: a public per-token left/right
# mirror-pair id (mirror_id_i == mirror_id_j <-> i, j are morphological mirror partners; -1 = unpaired), generalizing
# the g1-dim-rel-v1-only `edge.mirror` channel to any morphology whose collate path fills the field (arm, legged).
register_field(FieldDef("mirror_id", 1, "id", "public"))

register_factor(FactorDef("id.same_body", "1", field="entity_id", op="same", form="aug", sources=("given",),
                          doc="1[entity_id_i == entity_id_j]: same rigid body / object / scene entity across a pair "
                              "of tokens (fixed random unit codes per id, exact equality for n_ids <= code_dim)"))
register_factor(FactorDef("id.same_assembly", "1", field="assembly_id", op="same", form="aug", sources=("given",),
                          doc="1[assembly_id_i == assembly_id_j]: same manipulator / limb / body assembly across a "
                              "pair of tokens; the field-level generalization of edge.same_assembly (g1-dim-rel-v1 "
                              "channel only) and route.own_assembly (mask, system 0 routing) to any token set that "
                              "carries a per-token assembly_id (arm, dual, legged, Ψ₀)"))
register_factor(FactorDef("kin.ancestor", "1", field="edges:*", op="ancestor", form="bias", sources=("given",),
                          params=(("edge", "kin_parent"),),
                          doc="transitive closure of the kin_parent graph: 1[j is an ancestor of i] (params.rel: "
                              "'closure' (default) ancestor, 'inverse' descendant, 'sibling' see kin.sibling)"))
register_factor(FactorDef("kin.sibling", "1", field="edges:*", op="hop", form="bias", sources=("given",),
                          params=(("edge", "kin_parent"), ("hops", 2)),
                          doc="kin_parent graph-distance exactly 2 in the undirected (symmetrized) kinematic tree: "
                              "the common case is two nodes sharing an immediate parent (siblings), but a generic "
                              "graph-distance-2 op also lights up grandparent<->grandchild pairs at the same "
                              "undirected distance; params.hops overrides the distance"))
register_factor(FactorDef("kin.mirror", "1", field="mirror_id", op="same", form="aug", sources=("given",),
                          doc="1[mirror_id_i == mirror_id_j] over the public mirror-pair id (unpaired tokens use "
                              "mirror_id = -1 and never match); a token trivially matches itself, same as any other "
                              "`same` factor. Generalizes Ψ₀ dims' edges:g1-dim-rel-v1 `mirror` channel (still "
                              "available as edge.mirror) to morphologies with no such edge vocab (arm, legged), once "
                              "their collate path fills mirror_id"))

register_preset("graph", ["id.same_body", "id.same_assembly", "kin.ancestor", "kin.sibling", "kin.mirror"])
# ------------------------------------------------------------------ R16: contact / grasp / handover (ix.contact, ix.held_by, ix.handover)
# Labels from `rrp.harness.data.relgen.contact` against `StateView.contacts()` (docs/relations.md 5.1, 6, 10;
# research/relations_catalog.md D "physical interaction"). All three are the learned bilinear kernel on ctx token
# hiddens (`hidden` / `bilinear` / `aug`): the bias IS the pair probe (sigmoid(<Ux_i,Vx_j>+c), section 3.2); there is
# no public/estimated "given" source for contact state, so `probe` (learned) is the only deployable source and `gt`
# is training/diagnostics only (deploy guard, section 7). `part grasp_target` (this unit) is the shared scene
# component that makes a graspable object reachable by one or two manipulators, so it is named in `gen` by all three.
_IX_BILINEAR = (
    # name        label              doc
    ("contact",  "contact_pairs",  "two entities in geometric contact (arm, dual, legged foot-ground)"),
    ("held_by",  "held_pairs",     "an object held by a manipulator assembly (>= 2 contacts with its hand bodies, "
                                    "like today's `held` label); arm, dual"),
    ("handover", "handover_pairs", "two manipulator assemblies simultaneously in contact with the same object "
                                    "(a handover in progress); dual only"),
)
for _n, _lab, _doc in _IX_BILINEAR:
    register_factor(FactorDef(f"ix.{_n}", "1", field="hidden", op="bilinear", form="aug",
                              algebra=Algebra(arity=2, direction="symmetric", value="prob", dynamic=True),
                              sources=("probe", "gt"), label=_lab, gen=("grasp_target",),
                              readout=ReadoutDef(_n, "pair", 1, "bce", label=_lab, reads="hidden"),
                              params=(("rank", 8),), doc=_doc))
register_preset("ix", [f"ix.{n}" for n, _, _ in _IX_BILINEAR] + ["ix.support", "ix.force_flow"])
# extended by rel-geo (D-144 addendum) to include R17's `ix.support` / `ix.force_flow` -- forward-referenced here
# (the `FactorDef`s themselves are registered below, in the R17 section) exactly like R13/R18's own forward
# references to not-yet-registered R14 label/part names (unregistered names are inert strings until looked up).
# ------------------------------------------------------------------ R17: support / force flow (ix.support, ix.force_flow)
# labels + part `stack`: rrp.harness.data.relgen.support (research/relations_catalog.md D "support / stacking",
# "force transfer"). `ix.support` is the pairwise bilinear (learned kernel on token hiddens, address "pair", bce
# against the `support_pairs` label); `ix.force_flow` is the `flow` closure (transitive, directed) of the estimated
# / given support graph, bias form, supervised by `support_closure` (the same label module's transitive closure of
# `support_pairs`). Both decouple through the `stack` scene part (`gen`).
register_factor(FactorDef("ix.support", "1", field="hidden", op="bilinear", form="aug",
                          algebra=Algebra(direction="directed", value="prob", dynamic=True),
                          sources=("probe", "gt"), label="support_pairs", gen=("stack",),
                          readout=ReadoutDef("support", "pair", 1, "bce", label="support_pairs", reads="tokens"),
                          params=(("rank", 8), ("emits", ("support-v1", "support"))),
                          doc="a supports b: contact + contact normal within 30 deg of gravity-up at a's top "
                              "(relgen.support.support_matrix); its pair estimate is emitted as the estimated "
                              "`edges:support-v1` graph (FactorSite.emit) that `ix.force_flow` closes"))
register_factor(FactorDef("ix.force_flow", "1", field="edges:support-v1", op="flow", form="bias",
                          algebra=Algebra(direction="directed", transitive=True, value="bool", dynamic=True),
                          sources=("probe", "gt"), label="support_closure", gen=("stack",),
                          params=(("edge", "support"),),
                          doc="upstream / downstream closure of the support graph (relgen.support.support_closure); "
                              "source probe = the closure of `ix.support`'s emitted pair estimate thresholded at "
                              "params.threshold (0.5; FactorSite.emit, deployable); source gt = the privileged "
                              "`support_edges` graph at `<site>#support-v1@gt` (training / diagnostics only)"))
# ------------------------------------------------------------------ R18: task / temporal / epistemic (task.*, time.*)
# `task.next_contact` (docs/relations.md section 6 first wave, section 10 row R18): candidate manipulator/object
# interaction -> bilinear score on TOKEN HIDDENS (field "hidden": no FieldDef needed, matching `ix.*`, section 3.2's
# `bilinear` op), gated by the task summary (the same scene shows several candidate next-contact edges; after the
# task description the bias focuses on the required sequence, docs 3.4). `sources=("probe", "gt")` -- matching
# R16's `ix.*` (the bias IS the pair probe; there is no public/estimated "given" source for interaction state, only
# the learned kernel deployably, `gt` training/diagnostics-only per the deploy guard, section 7). Note `control="gt"`
# itself is never valid on ANY `bilinear` factor (`BilinearOp.controls()` = on/off/zero/rewired only: a bilinear op
# reads hiddens directly, it never resolves a field through `effective_source`); "gt" here is reached only via an
# explicit `FactorSpec(source="gt")` override, which the deploy guard (`assert_deployable`) still honors. Its
# candidate pool -- the soft PUBLIC EdgeSet of manipulator -> graspable / object -> support / destination pairs --
# is emitted in the collate path (`nets.batch.candidate_interaction_edges`, this unit's other owned file), never
# read directly by this factor (a bilinear factor reads hiddens, not edges); the candidates instead feed
# `harness.data.relgen.task.next_contact_sample` so `reveal` / `surprise` (R9, unmodified) supervise it
# progressively. Label `next_contact` (ground truth: which candidate is CURRENTLY touching a manipulator,
# `envs.base.StateView.contacts`) lives in `relgen/task.py`.
register_factor(FactorDef(
    "task.next_contact", "1", field="hidden", op="bilinear", form="aug",
    algebra=Algebra(arity=2, direction="directed", dynamic=True, value="prob"),
    sources=("probe", "gt"), label="next_contact", gen=("reveal", "surprise"), gates=("task",),
    readout=ReadoutDef(query="next_contact", address="pair", out=1, loss="soft_ce", label="next_contact",
                       reads="hidden"),
    params=(("rank", 8),),
    doc="candidate manipulator -> graspable / object -> support / destination bilinear score, sharpened by the "
        "task gate; supervised by progressive reveal / surprise over the R18 candidate-edge set (docs 5.4)."))

# `time.same_track` (section 10 row R18 brief): 1[same persistent track] for multi-step token histories (system 0
# knots, packet-history state, any net that keys history tokens by a track / assembly-over-time id); PUBLIC (the
# track id is the pipeline's own bookkeeping, not simulator truth), so `sources=("given",)` only -- no probe / gt
# ambiguity to resolve, unlike the privileged `ix.*` / `task.*` interaction factors above. Used only by nets that
# carry history tokens (none yet on `main`; a declarative entry, wiring is that net's own unit per docs 3.1).
register_field(FieldDef("track_id", 1, "id", "public"))
register_factor(FactorDef(
    "time.same_track", "1", field="track_id", op="same", form="aug",
    algebra=Algebra(arity=2, direction="symmetric"),
    sources=("given",),
    doc="1[track_i == track_j] over multi-step history tokens (system-0 knots, packet-history state); public."))
register_preset("task", ["task.next_contact", "time.same_track"])   # rel-geo (D-144 addendum, item 3)
# ------------------------------------------------------------------ R19: legged (leg.*)
# Labels from `rrp.harness.data.relgen.body` against `StateView.entities()` / `.contacts()` (docs/relations.md 5.1,
# 10 row R19; research/relations_catalog.md B "locomotion": "footholds, COM <-> support polygon, stability margin").
# `leg.foothold` mirrors R16/R17's bilinear pair pattern exactly (the bias IS the pair probe: no public/estimated
# "given" source for which terrain cell a swinging foot will land on next, so `probe` is the only deployable source
# and `gt` is training/diagnostics only, deploy guard section 7); `gen=("terrain_steps",)`, the scene part that
# supplies the candidate `foothold_cell` entities. `leg.com_support` is a scalar per-sample readout (stability
# margin: signed distance of the COM projection to the support-polygon boundary), matching `probes:legged-v1`'s
# `address="asm"` convention (one value read at the body-assembly row) rather than a pairwise bias -- there is no
# second token for "the robot's own stability" to attend over.
register_factor(FactorDef(
    "leg.foothold", "1", field="hidden", op="bilinear", form="aug",
    algebra=Algebra(arity=2, direction="directed", value="prob", dynamic=True),
    sources=("probe", "gt"), label="foothold_next", gen=("terrain_steps",),
    readout=ReadoutDef("foothold", "pair", 1, "bce", label="foothold_next", reads="hidden"),
    params=(("rank", 8),),
    doc="swinging foot -> nearest candidate terrain cell (relgen.body.foothold_next_fn); planted feet (already in "
        "stance) contribute no true pair"))
register_factor(FactorDef(
    "leg.com_support", "1", field="packet", op="inert", form="readout", label="com_support",
    readout=ReadoutDef("com_support", "asm", 1, "gauss", label="com_support"),
    doc="signed planar margin of the COM projection inside the convex hull of the current stance feet "
        "(relgen.body.support_polygon_margin via com_support_fn); positive = inside, negative = outside "
        "(0-foot stance: large fixed negative margin, relgen.body._NO_SUPPORT_MARGIN)"))
register_preset("legged-r19", ["leg.foothold", "leg.com_support"])
# ------------------------------------------------------------------ R20: UI (ui.*)
# `envs.computerworld` builds the public data this section names (`UI_REL_VOCAB`, `ui_public_fields`, `ui_edges`;
# docs/relations.md section 2's pointer/CW family row: "+pos3d ..., +zlayer, +parent_id, +focus_rank, entity_id").
# `envs` sits below `policies` in the layer order (`tests/unit/test_layering.py`), so importing FROM the env adapter
# here (policies -> envs, downward) is fine; the reverse is not, which is why `envs.computerworld` builds plain
# numpy, never a `TokenSet` / `EdgeSet` (those stay a collate-path concern, matching rel-geo's `edges:support-v1`
# precedent). `parent_id` / `entity_id` are the foundation's own fields already; `zlayer` (named in section 1's
# vocabulary table already) and `focus_rank` are new.
from rrp.envs.computerworld import UI_REL_VOCAB
register_field(FieldDef("zlayer", 1, "scalar", "public", units="dense z-layer rank"))
register_field(FieldDef("focus_rank", 1, "scalar", "public", units="0 = currently focused, -1 = every other widget"))
VOCABS["ui-rel-v1"] = UI_REL_VOCAB

# `ui.label_for` / `ui.focus_next`: the ARM `edge.*` pattern (foundation, above) applied
# to `UI_REL_VOCAB` -- every channel is public / deterministic (built from role, window, scene order, z-layer and
# focusable/disabled, all already in `ObjectDescriptor.attributes`), so `sources` defaults to "given" everywhere;
# "gt" is additionally offered (matching `geo.*`'s pattern, docs section 6) so a privileged `envs.computerworld`
# relgen label can be diagnosed against / substituted for the deployable public edge on the SAME entry, via
# `relgen.ui`'s label of the identical name. `ui.label_for` alone gets `gen=("reveal", "surprise")`: which
# label-role widget a control is bound to is exactly docs 5.4's own worked surprise example ("UI label changes") --
# a candidate classification over a window's label-role widgets, not a fixed fact, so it is the one channel here
# that benefits from progressive reveal / contradiction-after-collapse training (`relgen.ui.label_for_sample`).
_UI_EDGE_DOC = {
    "label_for": "role=\"label\" widget -> the next widget after it (scene order) in the same window: the "
                 "'label immediately precedes the control it describes' layout convention",
    "focus_next": "i, j are consecutive slots of the public tab order (focusable, enabled, boxed widgets in scene "
                  "order; cyclic, the last wraps to the first)",
}
for _n, _doc in _UI_EDGE_DOC.items():
    register_factor(FactorDef(f"ui.{_n}", "1", field="edges:ui-rel-v1", op="edge", form="bias",
                              sources=("given", "gt"), label=_n,
                              gen=("reveal", "surprise") if _n == "label_for" else (),
                              params=(("edge", _n),), doc=_doc))
# D-144 addendum (lead review of R20): relations that are functions of a public per-token field are declared with the
# generic operator over that field, not as hand-built vocabulary channels (docs 3.1: compose primitives). Windows are
# not tokens, so the widget-level containment relation is membership of one window (`same` on `parent_id`); a
# container -> member tree edge (`ancestor` over `parent_id`) needs container tokens and stays a catalog entry.
register_factor(FactorDef("ui.same_window", "1", field="parent_id", op="same", form="bias",
                          algebra=Algebra(direction="symmetric"), sources=("given",),
                          doc="1[i, j are widgets of the same window] (parent_id equality; desktop-level widgets, "
                              "parent_id -1, never match)"))
register_factor(FactorDef("ui.above", "2", field="zlayer", op="order", form="bias",
                          algebra=Algebra(direction="antisymmetric", value="signed"), sources=("given",),
                          params=(("axis", (1.0,)), ("margin", 0.5)),
                          doc="sign(zlayer_j - zlayer_i): +1 = j renders above i, -1 = below, 0 = same layer"))

# `ui.drag_to`: the ix.*/leg.* bilinear pair-probe pattern (docs 3.1: the bias IS the pair probe) -- there is no
# public/estimated "given" source for a drag DESTINATION (only the learned kernel deployably; "gt" is
# training/diagnostics-only, section 7's deploy guard). Label `drag_to` (`relgen.ui.drag_to_fn`): the currently
# focused widget (CW's only public signal of "the widget the pointer is actively interacting with", `scene.focus`)
# paired with the nearest OTHER widget by on-screen position -- the same nearest-candidate pattern R19's own
# `leg.foothold` uses for its privileged pair label.
register_factor(FactorDef(
    "ui.drag_to", "1", field="hidden", op="bilinear", form="aug",
    algebra=Algebra(direction="directed", dynamic=True, value="prob"),
    sources=("probe", "gt"), label="drag_to",
    readout=ReadoutDef("drag_to", "pair", 1, "bce", label="drag_to", reads="tokens"),
    params=(("rank", 8),),
    doc="widget currently interacted with (scene.focus) -> nearest other widget by position, the candidate drop "
        "target of an in-progress drag (relgen.ui.drag_to_fn)"))

register_preset("ui", ["ui.label_for", "ui.same_window", "ui.focus_next", "ui.above", "ui.drag_to"])
