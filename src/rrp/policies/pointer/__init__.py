"""Pointer-body (ComputerWorld `cw_pointer`) system 0 and packet sources (docs/architecture.md sections 3 and 5).

The pointer body has one assembly (`tool`, M = 1) and the latent contract's knots (0.1, 0.3, 0.5, 0.7 s after
valid_from; validity 0.8 s). At the pointer's 10 Hz control rate knot k covers the control ticks that END in
(t_{k-1}, t_k] (t_{-1} = 0): one tick for k = 0 and two ("early", "late") for k >= 1, so a packet spans 7 ticks.

    spec        constants, `PolicyConfig` (relation factors), `PointerGeometry` (screen / pixel pitch / rate from the env spec)
    packet      engineered packet encoding `cw_pointer_eng.v1`, `pointer_packet`, SCRIPTED `EngineeredSystem0`
    features    the public featurization every learned pointer policy reads (and only that)
    probe       packet probe (`ReadoutProbe` on `probes:pointer-v1`)
    checkpoint  `nets()`, the `RelBlock` checkpoint map, `load_pointer_bundle`
    runtime     `TeacherOracleSource` (ORACLE), `LearnedSystem0`, `PointerSystemI`, `PointerBCPolicy`, `make_pointer_*`

The nets are in `rrp.policies.nets.pointer`; `import rrp.policies.pointer` stays torch-free for the scripted route
(`nets()` and the checkpoint loaders import them lazily). Public names are re-exported here, so the import path of
every consumer (`rrp.policies.pointer:make_pointer_oracle` in the policy registry, tests, trainers) is unchanged.
"""
from rrp.policies.pointer.checkpoint import load_pointer_bundle, load_pointer_module, nets
from rrp.policies.pointer.features import (EventHistory, bound_id, check_bound_ids, codes, collate_public,
                                           public_features, screen_half, sym_of_char, sym_of_key, ui_widget_fields,
                                           widget_features)
from rrp.policies.pointer.packet import (EngineeredSystem0, cmd_source, decode_slot, encode_commands, packet_ticks,
                                         pointer_packet, tick_slot)
from rrp.policies.pointer.probe import (POINTER_PROBE_PRESET, load_pointer_probe_state, new_pointer_probe,
                                        pointer_probe_specs, run_pointer_probe)
from rrp.policies.pointer.runtime import (LearnedSystem0, PointerBCPolicy, PointerSystemI, TeacherOracleSource,
                                          env_widget_table, make_pointer_bc, make_pointer_latent,
                                          make_pointer_oracle, pointer_requirements, target_depth)
from rrp.policies.pointer.spec import (ENG_DIM, ENG_VERSION, KNOT_TIMES, LC, LI, MAX_STEP_PX, N_BOUND, N_KEYCLS, N_ROLE,
                                       N_SYM, NH, NW, PHASES, POINTER_FACTORS_PRESET, POINTER_KINDS, ROLE_IDS,
                                       SLOT_FIELDS, SLOT_W, UI_CARRIES, VALIDITY_S, WF, PointerGeometry, PolicyConfig)
