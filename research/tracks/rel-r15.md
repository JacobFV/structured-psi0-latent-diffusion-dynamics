# track: rel-r15 (relation-factor fanout, unit R15: membership / graph factors)

Branch `track/rel-r15`, worktree `~/work/rrp-wt/rel-r15`, from `origin/main` @ `f38b689a` (R12 merged; foundation
`c5e7d45`). Spec: `docs/relations.md` sections 2-5 and 10 (row R15, deps R12), `research/relations_catalog.md`
sections B ("identity, membership, correspondence") and the hierarchy / mirror rows, `AGENTS.md`. Host only (git /
editing / unit suite; CUDA hidden); no peer smoke needed (the row's acceptance is pure unit math over morphology
fixtures, no training).

## scope

Edited only `src/rrp/policies/relations/catalog.py` (my pre-created `§graph` section) and added new test functions
(a disjoint section) to the shared `tests/unit/test_relations.py`; no row lists a dedicated test file for R15, and
this file already holds the registry's cross-cutting tests (`test.anc` etc.). Did not touch `base.py` or `ops.py` —
every operator this row needs (`same`, `hop`, `ancestor`) already exists in the foundation; no operator/interface
change was needed.

Five catalog entries, per the row's brief ("membership / graph entries using `same`, `hop`, `ancestor` over
`entity_id`, `assembly_id` and the morphology parent field"):

- **`id.same_body`** — `field="entity_id"`, `op="same"`, `form="aug"`. `1[entity_id_i == entity_id_j]`.
- **`id.same_assembly`** — `field="assembly_id"`, `op="same"`, `form="aug"`. `1[assembly_id_i == assembly_id_j]`;
  the field-level generalization of the foundation's `edge.same_assembly` (only fires where a site carries the
  `g1-dim-rel-v1` edge vocab) and `route.own_assembly` (a `mask`, not a bias/aug term) to any token set that carries
  a per-token `assembly_id` field directly — arm, dual, legged, Ψ₀ (docs/relations.md §2's field table).
- **`kin.ancestor`** — `field="edges:*"`, `op="ancestor"`, `form="bias"`, `params={"edge": "kin_parent"}`. Transitive
  closure of the `kin_parent` edge channel (already registered generically as `edge.kin_parent` by the foundation,
  present in both `arm-rel-v1` and `g1-dim-rel-v1`): `1[j is an ancestor of i]`. This is the same wiring pattern the
  foundation's own `test.anc` fixture in `test_relations.py` already exercises (`ClosureOp` extends `EdgeOp`, so the
  op reads the parent relationship off an edge channel, not a raw `parent_id` field — the doc's first-wave table
  (§6) names the field `parent_id`, but the actual `ClosureOp` / `HopOp` implementations in `ops.py` are `EdgeOp`
  subclasses that only read `rc.edges[site]`; I followed the code, which is authoritative over the table, and did
  not touch `ops.py` to reconcile them since that would be an operator change outside this row's scope).
- **`kin.sibling`** — `field="edges:*"`, `op="hop"`, `form="bias"`, `params={"edge": "kin_parent", "hops": 2}`.
  Undirected `kin_parent` graph-distance exactly 2. Documented honestly in the entry's `doc`: at distance 2 this
  includes true siblings (shared immediate parent) *and* grandparent<->grandchild pairs (same undirected distance
  through a 2-edge path down instead of up-then-down) — confirmed exactly by the arm-tree fixture test below, not
  glossed over.
- **`kin.mirror`** — `field="mirror_id"`, `op="same"`, `form="aug"`. New `FieldDef("mirror_id", 1, "id", "public")`
  registered in the same section: a per-token left/right mirror-pair id (`-1` = unpaired). Generalizes Ψ₀ dims' edge
  `mirror` channel (`g1-dim-rel-v1`-only, still available as `edge.mirror`) to morphologies with no such edge vocab
  (arm, legged), once their collate path fills `mirror_id` — no collate path fills it yet on `main` (R12 didn't add
  it), so `kin.mirror` is declared and fully unit-tested against the `same` op's math but is currently inert at
  every real site until a future row wires the field (the same "declare now, wire the field later" pattern the
  foundation already uses for `geo.depth3d` / `ix.contact` in docs/relations.md §6).

Also registered `preset:graph` (`id.same_body, id.same_assembly, kin.ancestor, kin.sibling, kin.mirror`) so a run
config or a test can turn on "every graph factor" in one line, mirroring `preset:arm` / `preset:psi0-dims`.

## tests (red then green)

Added to `tests/unit/test_relations.py` (new section at the end, three new functions; nothing existing touched).
Verified red first: `git stash push -- src/rrp/policies/relations/catalog.py`, ran the file (3 failures,
`FactorError: unknown factor 'kin.ancestor'` etc.), `git stash pop`, iterated to green.

- `test_graph_factors_registered_and_resolve_on_arm_psi0_legged`: all five entries are `status="implemented"`;
  `resolve(["preset:arm", "preset:graph"])` and `resolve(["preset:psi0-dims", "preset:graph"])` both succeed and
  contain all five; `resolve(["preset:graph"])` alone succeeds (the "legged" case: legged token sets carry no edge
  vocabulary at all per docs/relations.md §2, only `assembly_id` / per-family fields); building a `FactorSite` with
  `carries=("assembly_id", "entity_id")` (a legged-shaped site) confirms only `id.same_body` / `id.same_assembly`
  actually apply there — the edge- and `mirror_id`-based factors are silently inert, not an error, which is what
  "factors resolve on arm / Ψ₀ / legged sets" has to mean for a graph vocabulary legged doesn't carry.
- `test_kin_ancestor_and_sibling_on_arm_morphology_fixture`: a small synthetic 7-node arm-style morph tree over
  `edges:arm-rel-v1`'s `kin_parent` channel (root 0; children 1, 2; grandchildren 3, 4, 5, 6). `kin.ancestor`'s
  closure and `kin.sibling`'s hop-2 set are checked against independent reference implementations
  (`_ancestor_closure`, `_undirected_hop_distance`, plain BFS/walk, not calling the ops under test) over the full
  7x7 matrix, plus an explicit enumeration of the hop-2 pairs showing the 3 true sibling pairs and the 4
  grandparent<->grandchild pairs the operator's generic "distance == k" semantics also lights up.
- `test_kin_ancestor_sibling_mirror_and_same_assembly_on_g1_morphology_fixture`: the real G1 morphology
  (`rrp.bodies.g1_simple`, 36 command-dim tokens, real `DIM_PARENT` / `DIM_MIRROR` / `DIM_ASM`, real
  `relation_matrix()`). `kin.ancestor` / `kin.sibling` checked against the same independent references over the
  full 36x36 matrix (not a hand-picked subset), plus a concrete real triple (the left hand's three top-level
  fingertip-chain joints share one immediate parent, `l_wrist_yaw`); `id.same_assembly`'s field-based value checked
  exactly against the body module's own edge-based `same_assembly` channel (two independent implementations of "same
  assembly" must agree); `kin.mirror`'s field-based value checked exactly against `relation_matrix()`'s `mirror`
  edge channel (off-diagonal) plus the `same` op's own diagonal / unpaired-id semantics (a paired token trivially
  matches itself; the five unpaired `base` dims, `mirror_id = -1`, never match anything, including themselves).

## commands run

```
cd ~/work/rrp-wt/rel-r15 && export PYTHONPATH=$PWD/src:$PWD
~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit -q                       # baseline: 613 passed, 39 skipped, exit 0
git stash push -- src/rrp/policies/relations/catalog.py
~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit/test_relations.py -q     # red: 3 failed (unknown factor), 10 passed
git stash pop
~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit/test_relations.py -q     # green: 13 passed
~/work/relational-robot-policy/.venv/bin/python -m pytest tests/unit -q                       # full suite post-change: 616 passed, 39 skipped, exit 0
```

State: **completed** (implementation + tests green; merge to main pending per docs/relations.md 10's merge rule:
`pytest tests/unit -q` exit 0, `git fetch origin && git rebase origin/main`, `git push origin HEAD:main`).
