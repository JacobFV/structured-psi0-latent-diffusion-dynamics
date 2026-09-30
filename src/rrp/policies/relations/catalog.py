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
VOCABS = {"arm-rel-v1": ARM_REL_VOCAB, "g1-dim-rel-v1": tuple(_G.RELATIONS)}

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
# ------------------------------------------------------------------ R13: geometry (geo.*)
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
# ------------------------------------------------------------------ R17: support / force flow (ix.support, ix.force_flow)
# ------------------------------------------------------------------ R18: task / temporal / epistemic (task.*, time.*)
# ------------------------------------------------------------------ R19: legged (leg.*)
# ------------------------------------------------------------------ R20: UI (ui.*)
