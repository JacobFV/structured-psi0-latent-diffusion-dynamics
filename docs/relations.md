# relation factors (D-144): one declarative registry for attention-interaction structure, the probes that estimate it and the data that teaches it

Status: design + foundation plan (lead: branch refactor/relations). Architecture context: `docs/architecture.md`
(layers, Env / Policy / Task, harness). Catalog of candidate relations and their decomposition:
`research/relations_catalog.md`. This page is the end state, the migration map, the delete list and the fanout plan.

Why: relational structure is a scaling axis (PaPE-style geometry, typed task graphs, interaction structure,
task-conditioned contact edges, probe-maintained subspaces; the owner's brainstorm lists ~600 candidates). Today each
trick lives in its own net family behind its own flag (`bias_mode`, `structured`, `slot_handles`, `semantic_weight`,
`binding_cf`, `probe_lv_min`, `$RRP_KINFEAT`, anchor options), with six probe classes, three `-inf` routing masks and
two relation vocabularies. End state:

1. a relation is a **small declarative entry** composed from a handful of primitives (field × operator × form ×
   conditioning × source), never a class per relation;
2. it is switched in **one list** (`factors:` in the run config), hashed into the checkpoint and applied by **one
   attention block** shared by every net family (arm, dual, legged, pointer, Ψ₀ head);
3. its **label and data generator** are equally declarative: label functions written once against one privileged
   state interface, composable variation generators and generic transforms (progressive reveal, surprise,
   counterfactual swap, noise / occlusion), assembled by one pipeline stage and one data-mix hook.

## 1. vocabulary

| term | meaning |
|---|---|
| **token set** | tokens of one role in one net: `ctx` (morph nodes, assemblies, scene entities / widgets, image patches, task events / roles / predicates, receipts, sensors), `act` (action nodes or packet assemblies), `knots` (packet knot × assembly), `dims` (Ψ₀ command dims) |
| **field** | a named per-token quantity (`pos3d`, `cam_uvd`, `orient`, `normal`, `extent`, `entity_id`, `assembly_id`, `parent_id`, `track_id`, `time`, `zlayer`, `material_id`, `mass`, ...) or a per-pair graph (`edges:arm-rel-v1`), tagged with provenance |
| **operator** | how a pair (i, j) (or a single token j) turns fields into a relation value: difference, squared difference, relative rotation, alignment, equality, similarity, graph adjacency / hop / ancestor, ordering along an axis, flow along a directed graph, learned bilinear kernel, unary key prior |
| **form** | how the value enters attention: additive `bias`, kernel-compatible q/k `aug`mentation (PaPE), multiplicative `gate`, hard `mask`; plus the non-attention forms `message`, `embed`, `readout` (probe only) |
| **factor** | one registry entry = field + operator + form + algebra metadata + conditioning + sources + label + data generators |
| **site** | one attention call whose logits factors may modify, named `q>k` by its token sets (`ctx>ctx`, `act>ctx`, `act>act`, `dims>dims`, `dims>knots`, ...) |
| **label** | a function of privileged simulator state (via `StateView`) or of public task structure that yields the factor's per-token / per-pair target |

### provenance (the deploy boundary)

```python
Prov = Literal["public", "estimated", "privileged"]
# public     : sensors, public FK of measured joints, declared morphology, supplied task graph/runtime, env-declared layout
# estimated  : output of a public estimator run on public inputs (tracker estimate, the factor's own probe head, a
#              distilled external estimator: depth / normals / segmentation / affordance)
# privileged : simulator ground truth, future outcomes, teacher internals, offline-only estimators.
#              SUPERVISION ONLY: lives in TokenSet.labels, never in TokenSet.fields
LabelProv = Literal["gt", "estimator:<name>", "synthetic:<generator>", "task"]   # recorded on every label
```

A deployable policy never reads `labels`; a factor whose source is `gt` is legal only in training / diagnostics and
labels the run `privileged` (section 7).

## 2. token model (`rrp.policies.relations.base`)

```python
@dataclass(frozen=True)
class FieldDef:                       # registry FIELDS (declarative)
    name: str; dim: int
    kind: Literal["position", "frame", "orientation", "direction", "shape", "scalar", "vector", "id", "membership",
                  "graph", "time", "belief", "symbol"]
    frame: str = ""                   # "base" | "camera:<name>" | "screen" | "world" | "token"
    units: str = ""
    label: str | None = None          # LABELS key producing its privileged ground truth (for probe / gt sources)

@dataclass
class TokenSet:
    name: str
    mask: Tensor                      # [B,T] bool; null identities stay as masked slots
    kind: Tensor                      # [B,T] long -> TOKEN_KINDS
    fields: dict[str, Tensor] = {}    # public/estimated: name -> [B,T,dim]; "<name>.valid" [B,T]; "<name>.var" [B,T,dim]
    labels: dict[str, Tensor] = {}    # privileged per-token targets (supervision only)
    def field(self, name) -> Tensor
    def label(self, name) -> Tensor   # raises PrivilegedInput in deploy mode

@dataclass
class EdgeSet:                        # a graph field between a query set and a key set
    vocab: tuple[str, ...]            # channel names; the order is the featurizer's on-disk order
    data: Tensor                      # [B,Q,K,R] bool (given) or float in [0,1] (estimated / soft)
    prov: Prov = "public"

TOKEN_KINDS = ("pad", "morph_node", "passive_joint", "assembly", "entity", "widget", "image_patch", "task_event",
               "task_role", "predicate", "receipt", "sensor", "action_node", "knot", "command_dim", "text")

@dataclass
class RelCtx:                         # built once per forward by the net from its collate output
    sets: dict[str, TokenSet]; edges: dict[str, EdgeSet]   # edges keyed by site "q>k"
    factors: "FactorSet"; task: Tensor | None = None       # task summary [B,D] for gates
    deploy: bool = False; generator: torch.Generator | None = None   # control RNG, drawn in site order
    estimates: dict = {}              # field estimates produced by factor-owned readouts during this forward
```

Collate functions (one per featurizer family) are the only producers of token sets. They derive fields from the
featurizer's token columns (datasets on disk need no rewrite) and attach labels only on the training path.

| family | token sets | edges (vocab) | public / estimated fields (`+` = added by fanout) |
|---|---|---|---|
| arm / dual flow + E (`features.featurizer`, `features.multi`, `nets.batch`) | `ctx` (4 banks), `act` (nodes; latent path: packet assemblies) | `ctx>ctx`, `act>ctx`, `act>act` (`arm-rel-v1` = the 17 `REL`) | `pos3d` (morph anchors / assembly sites: public FK; scene slots: estimated + `.var` from `position_cov_diag`; receipts), `+cam_uvd`, `+orient`, `entity_id`, `assembly_id` |
| arm system 0 (`policies.system0`) | `act`, `knots` | `act>knots` (routing) | `assembly_id`, `time` |
| legged (`nets.legged_latent`, `features.legged`) | `ctx` (global + joints), `act`, `knots` | `act>knots` (routing) | `assembly_id`, `+pos3d` (feet / base FK), `+time` |
| Ψ₀ (`policies.psi0.nets`, `bodies.g1_simple`) | `dims`, `knots` | `dims>dims` (`g1-dim-rel-v1`: self, parent, child, same_assembly, mirror), `dims>knots` (routing: `READS`) | `assembly_id`, `+pos3d` (palms / pelvis from `psi0_state`) |
| pointer / ComputerWorld (`policies.pointer`) | `ctx` (widgets, pointer, events), `knots` | `+ctx>ctx` (`ui-rel-v1`) | `+pos3d` (screen mm + z-layer depth), `+zlayer`, `+parent_id`, `+focus_rank`, `entity_id` |

## 3. factors: declarative entries over a few primitives

### 3.1 the entry

```python
@dataclass(frozen=True)
class Algebra:                         # metadata; drives validation, controls and default symmetrization
    arity: Literal[1, 2] = 2           # 1 = property (key-side prior / field only); 2 = pairwise (graphs are pairwise)
    direction: Literal["directed", "symmetric", "antisymmetric"] = "directed"
    transitive: bool = False           # closure is precomputed by the op (ancestor, flow)
    value: Literal["bool", "signed", "weighted", "prob", "vector"] = "bool"
    dynamic: bool = False              # changes within an episode (contacts) vs static (morphology)
    range: Literal["local", "global"] = "global"

@dataclass(frozen=True)
class FactorDef:                       # registry FACTORS (a table of these; no subclassing)
    name: str; version: str
    field: str                         # FIELDS key, or "edges:<vocab>"
    op: str                            # OPS key (3.2)
    form: Literal["bias", "aug", "gate", "mask", "message", "embed", "readout"]
    algebra: Algebra = Algebra()
    sources: tuple[str, ...] = ("given",)   # given | probe | estimator:<name> | gt  (first = default)
    label: str | None = None           # LABELS key (supervises `probe`, supplies `gt`)
    gen: tuple[str, ...] = ()          # PARTS / TRANSFORMS keys that make its training data (section 5)
    gates: tuple[str, ...] = ()        # allowed conditioning: task | instruction | goal | embodiment | history
    readout: ReadoutDef | None = None  # probe head (section 4); required when "probe" in sources
    params: dict = {}                  # op/form defaults (edge name, p, frame, rank, axis, hops, also, ...)
    status: Literal["implemented", "planned"] = "implemented"
    doc: str = ""

@dataclass(frozen=True)
class FactorSpec:                      # one entry of the run config's `factors:` list (JSON-able)
    name: str                          # registry name, a glob ("edge.*"), or "preset:<name>"
    control: Control = "on"
    source: str | None = None
    sites: tuple[str, ...] | None = None    # default: every site whose token sets carry the field / edge
    heads: tuple[int, ...] | None = None    # head budget (others get 0)
    gate: str | None = None                 # one of FactorDef.gates
    confidence: bool = False
    weight: float | None = None             # readout loss weight (0 = post hoc only; today's nosem)
    mix: float | None = None                # fraction of each training batch from the factor's data generator
    params: dict = {}                       # overrides of FactorDef.params (enter the compat hash)

FIELDS, OPS, FACTORS: dict[str, ...]; PRESETS: dict[str, list]
def resolve(specs) -> tuple[FactorSpec, ...]
    # expands presets; later entries override earlier matching names (globs over registered names); rejects unknown
    # names, `planned` entries, controls/sources/gates the entry does not support, forms the op cannot realize
def compat_hash(specs) -> str          # "fx-" + sha256[:12] over (name, version, op, form, source, sites, heads, gate,
                                       # confidence, params) of resolved, non-off specs; controls / weight / mix excluded
def provenance(specs) -> list[dict]    # everything, incl. controls and label provenance (checkpoint + Episode)
```

The legacy relations are entries, e.g. `FactorDef("edge.patient_of", "1", field="edges:arm-rel-v1", op="edge",
form="bias", params={"edge": "patient_of"})`; a PaPE depth factor is
`FactorDef("geo.depth3d", "1", field="cam_uvd", op="sqdiff+diff", form="aug", sources=("probe", "given", "gt"),
label="cam_uvd", gen=("camera_depth",), readout=ReadoutDef(query="cam_uvd", address="token", out=6, loss="gauss"),
params={"p": 3, "frame": "world", "readout_layer": 0})`. Registering a new relation from the catalog is one such line
(plus, if needed, one label function and one variation, section 5).

### 3.2 operators (the only code per relation *kind*; ~12 total)

Each operator declares its arity, input field kinds, the forms it can realize and, for `aug`, its separable features
φ_q(i), φ_k(j) with ⟨φ_q(i), φ_k(j)⟩ = value(i, j) up to terms that depend on i only (these cancel in softmax).

| op | value(i, j) | field kinds | forms | separable (aug) features |
|---|---|---|---|---|
| `edge` | A_r[i, j] of a given / estimated graph channel | graph | bias, mask, gate | – |
| `hop` | 1[graph distance(i, j) = k] (k = params.hops) or exp(−dist) | graph, id(parent) | bias | – |
| `ancestor` | transitive closure of a parent field (ancestor / descendant / sibling via params.rel) | id(parent) | bias | – |
| `flow` | 1[j upstream of i] along a directed graph (support / force / causal), closure precomputed | graph | bias | – |
| `same` | 1[id_i = id_j] (same object / part / assembly / track / material) | id | bias, aug | fixed random unit codes e(id): ⟨e(id_i), e(id_j)⟩ (exact for one-hot, ≈ for hashed) |
| `sim` | cos(v_i, v_j) of a property vector | vector, belief | bias, aug | normalized v_i, v_j |
| `diff` | ⟨b_i, P(r_j − r_i)⟩ (direction) | position | aug, bias | q: Pᵀb_i, k: r_j |
| `sqdiff+diff` | PaPE (arXiv 2602.01418, Eq. 9), p-dim compact form: −(r_j−r_i)ᵀ M_i (r_j−r_i) + b̃_iᵀ(r_j−r_i), M_i = P_iᵀ diag(a_i) P_i, a_i = g_h·softplus(W_a x_i) ≥ 0, b_i = W_b x_i, P_i = W_p (frame world) or W_p R_iᵀ (frame query: SE(3)-invariant), b̃_i = P_iᵀ b_i | position (+ orientation for frame query) | aug | q: [−vec(M_i), 2 M_i r_i + b̃_i], k: [vec(r_j r_jᵀ), r_j] (p²+p dims) |
| `rel_rot` | ⟨B_i, R_iᵀ R_j⟩_F (relative rotation; B = −αI recovers the geodesic term tr(R_iᵀR_j)) | orientation | aug | q: vec(R_i B_i), k: vec(R_j) (9 dims) |
| `align` | ⟨b_i, R_iᵀ n_j⟩ or n_i·n_j (normal / axis alignment, parallel / perpendicular via b) | direction | aug | q: R_i b_i, k: n_j |
| `order` | sign((r_j − r_i)·axis) with margin (above/below, left/right, front/behind, along view ray / gravity / motion) | position | bias | – |
| `bilinear` | ⟨U_h x_i, V_h x_j⟩ on token hiddens — the learned kernel; read at the readout layer it is also the edge probe p̂_ij = σ(⟨U x_i, V x_j⟩ + c) (the bias IS the probe: a supervised attention subspace) | hidden | aug, bias | q: U x_i, k: V x_j |
| `unary` | ⟨w_h, φ(f_j)⟩: key-side prior from a property field (salience, graspability, hazard, relevance) | scalar, vector | aug, bias | q: w_h (constant 1-dim), k: φ(f_j) |

Forms wrap an op's value uniformly: `bias` adds `w[f,h] · value` (all `edge`/`bias` factors of a site are ONE stacked
einsum `bqkr,rh->bhqk`); `aug` concatenates `g_h · φ_q` / `φ_k` to q / k (kernel-compatible: SDPA with fixed
`scale = 1/√d`, `qa` pre-multiplied by √d; the dense mask disappears when no `bias` factor is on at the site); `gate`
multiplies another factor's (or a head's) contribution by `σ(w · value + c)`; `mask` adds `-inf` where the value is 0
(hard routing, e.g. system 0 reads only allowed assemblies' knots). Every learned coefficient is zero-initialized so
enabling a factor on a trained model is a no-op at step 0 (zero-bias equivalence; `aug` gates g_h = 0).

Non-attention forms: `message` (incidence messages along pointer edges; today's `structured` path; control
`serialized` = the same facts as hashed text, the equal-information baseline), `embed` (learned embedding of an id
field added to tokens, e.g. `id.slot_handle`), `readout` (probe query only, section 4).

A dense learned-kernel form (MLP on Δr) is deliberately absent: `sqdiff+diff`, `rel_rot`, `align`, `same`, `sim`,
`unary` and `bilinear` cover distance, direction, orientation, identity, similarity, property priors and learned
interaction in kernel-compatible form. Add it only with evidence.

### 3.3 generic controls (applied by the site, uniform across factors)

| control | graph ops (`edge`, `hop`, `ancestor`, `flow`, `order`) | field ops (`diff`, `sqdiff+diff`, `rel_rot`, `align`, `same`, `sim`, `unary`) | `bilinear` / `message` / `embed` / `readout` |
|---|---|---|---|
| `on` | as given | as given | as given |
| `off` | skipped (params kept so checkpoints load) | skipped | skipped |
| `zero` | graph replaced by zeros (bias 0; mask: all allowed) | field replaced by 0 | bilinear: 0 |
| `rewired` | key tokens permuted among valid keys, one permutation per sample and site (degree preserving) | key-side field permuted among valid keys | key-side hidden permuted |
| `shuffled` | edge TYPES permuted among the site's graph channels (endpoints kept) | field values permuted across the set's tokens (both sides) | – |
| `reversed` | transpose (square self-attention sites only; identity elsewhere) | sign flip of antisymmetric ops | – |
| `gt` / `estimated` | graph from the label / from the factor's readout | field from the label / from the readout | – |
| `serialized` | – | – | `message` only |

Controls draw from `RelCtx.generator` in site order, so paired ablations are reproducible. Legacy `bias_mode` is the
control of every `edge.*` entry (section 8).

### 3.4 conditioning, confidence, heads

- **Gates** (`gate: task | instruction | goal | embodiment | history`): `γ_{f,h} = σ(u_{f,h}·c + b_{f,h})` scales the
  factor's contribution; `c` is the corresponding summary vector in `RelCtx` (task-bank mean, instruction embedding,
  goal token, morphology summary, packet-history state). Zero-init `u`. For `aug` it scales `φ_q` per sample, so it
  stays kernel-compatible. This is how "the same scene shows several candidate next-contact edges; after the task
  description the bias focuses on the required sequence" is expressed.
- **Confidence** (`confidence: true`): `c_i = exp(−κ·mean(var_i))` from the field's `.var` or the readout's
  log-variance; contribution × `c_i c_j` (bias: outer product; aug: `φ_q·c_i`, `φ_k·c_j`). `κ` learned (softplus).
- **Heads** (`heads: [...]`): a per-head budget, so factor count scales without new per-layer parameters.

### 3.5 the shared relational attention block (`rrp.policies.nets.attention` + `rrp.policies.relations.site`)

```python
class MHA(nn.Module):                 # parameters / paths unchanged; forward gains q_aug / k_aug
    def forward(self, x, kv=None, kv_cache=None, key_mask=None, bias=None, need_weights=False, q_aug=None, k_aug=None)
class FactorSite(nn.Module):          # one per attention call factors may touch; owns the per-head factor parameters
    def __init__(self, heads: int, dim: int, site: str, specs: Sequence[FactorSpec])
    def keys(self, rc) -> SiteKeys            # cacheable (flow steps): graph bias, masks, key-side φ_k
    def apply(self, x, rc, keys) -> dict       # {"bias", "q_aug", "k_aug"} for MHA.forward (query-side φ_q, gates)
    def contributions(self, x, rc) -> dict[str, Tensor]   # per-factor [B,H,Q,K] logit terms (diagnostics / viz only)
class RelBlock(nn.Module):            # (cross-attn, self-attn, MLP) pre-LN block; module keys n1,x,n2,s,n3,m kept;
                                      # replaces legged / Ψ₀ / system 0 / pointer copies of block/run_block
```

Parameter paths of existing checkpoints are preserved: the stacked `edge` weight is `FactorSite.w` and each FactorSite
sits where today's `StructuralBias` sits (`context.layers.i.bias.w`, `blocks.i.bias_c.w`, `blocks.i.bias_e.w`, Ψ₀
`dims.layers.i.sb.w`). New factors add `...f.<factor>.*` parameters. Loading a checkpoint into a config with extra
factors requires naming them (`load_state_dict(strict=False)` restricted to exactly those zero-init keys);
`versions["factors"]` differs and `require_compatible_versions` reports it.

## 4. probes = readouts in the same registry

Today: `SemanticReadout` (flow aux heads), `PacketProbe` (arm / dual), `AnchorProbe` (W12), `LeggedProbe`, Ψ₀
`PacketProbe`, `PointerProbe`: six copies of one design (fixed random handle codes + query-type embedding + two
cross-attention reads over packet tokens + MLP + per-query heads; Gaussian NLL with a log-variance floor, BCE, CE).

```python
@dataclass(frozen=True)
class ReadoutDef:
    query: str                         # "visible", "held_by", "rel_pos", "tcp_in_own", "contact", "cam_uvd", ...
    address: Literal["entity", "entity×asm", "asm", "knot×asm", "knot×pair", "body", "token", "pair"]
    out: int                           # head width (Gaussian: 2·d)
    loss: Literal["bce", "gauss", "ce", "cos", "mse", "soft_ce"]   # soft_ce: posterior targets (progressive reveal)
    scale: float = 1.0                 # target scaling (today ×10 dm, /30 gaze)
    lv_min: float = -8.0               # log-variance floor (D-085: -4 = bounded)
    reads: Literal["packet", "hidden", "tokens"] = "packet"   # received packet z (D-029), a net's hidden states (aux),
                                                                # token hiddens (factor sources)
class ReadoutProbe(nn.Module):         # (dz, knots, specs, handle sizes, metadata_only, seed)
    def forward(self, z, zmask, **addresses) -> dict[str, Tensor]
def readout_loss(out, labels, specs) -> (total, logs);  def readout_metrics(out, labels, specs) -> dict[str, (sum, n)]
```

- Probe queries are `FactorDef(form="readout", label=..., readout=...)` entries named `probe.<family>.<query>`,
  grouped in presets (`probes:arm-packet-v1` = today's `PacketProbe` queries in order; `probes:anchor-v1`,
  `probes:legged-v1`, `probes:psi0-v1`, `probes:pointer-v1`, `aux:arm-v1`). Weight, `lv_min`, `goal_effect`,
  `w_grasp`, `semantic_weight` / `packet_semantic_weight` / `w_sem` / `aux_weight` become spec `weight` / `params`
  (`params.on = "rep" | "flow" | "both"` for the E path vs the z_hat path). `metadata_only` stays a constructor argument
  (a control probe, recorded in provenance).
- A factor with source `probe` owns a readout on token hiddens at `params.readout_layer`; its estimate (+ variance)
  replaces the field for later layers and its loss maintains a linear subspace (owner: "probe-supervised depth in a
  maintained linear subspace of the latent"). `bilinear` factors own a `pair` readout.
- State-dict compatibility: `ReadoutProbe` keeps `PacketProbe`'s module names and parameter creation order; the other
  families' checkpoints load through a data-level key map in `nets.checkpoint` (on-disk data, like `load_pickle`).

## 5. data generation: labels, variations, transforms, mix (co-equal with the model side)

Owner: each relation may need hand-written data "to decouple a particular invariance in the data that we care to
model"; the data-gen code must be as declarative, DRY and scalable as the factors. One-to-one with factors where possible.

### 5.1 one privileged state interface (`rrp.envs.base.StateView`, capability `privileged_truth`)

```python
@dataclass(frozen=True)
class EntityState:   id: str; kind: str; name: str; pos: ndarray; quat: ndarray | None; vel: ndarray | None
                     extent: ndarray | None; mass: float | None; friction: float | None; material: str | None
                     parent: str | None; assembly: str | None; body: int | None; visible: bool | None; attrs: dict
@dataclass(frozen=True)
class ContactState:  a: str; b: str; pos: ndarray; normal: ndarray; force: ndarray | None; time: float
@dataclass(frozen=True)
class JointState:    name: str; parent: str; child: str; kind: str; axis: ndarray; q: float; qd: float; limits: tuple | None

class StateView(Protocol):            # env.state_view(); privileged; supervision / labelling only
    caps: frozenset[str]              # "poses", "velocities", "contacts", "forces", "joints", "inertia", "materials",
                                      # "camera", "depth_render", "ui_tree", "task_runtime", "terrain"
    time: float; gravity: ndarray
    def entities(self) -> list[EntityState]
    def contacts(self) -> list[ContactState]          # "contacts" (+ "forces")
    def joints(self) -> list[JointState]              # "joints"
    def camera(self, name) -> Camera                  # "camera": K, T_world_cam, width/height; render_depth() with "depth_render"
    def ui_tree(self) -> list[dict]                   # ComputerWorld: node, parent, z, focus rank, role, label, bounds
    def token_entity(self, token_set: str, slot) -> str | None   # the (privileged) association token -> entity id
```

Implemented once per backend: MuJoCo (`Session`, `DualSession`, `LeggedSession`), Warp (batched, per env index),
ComputerWorld (`truth()` scene), SIMPLE (`truth()` palms / pelvis / objects / contacts). A label written against the
view runs in every env whose `caps` cover its `needs` (checked like `negotiate`).

### 5.2 registries (`rrp.harness.data.relgen`)

```python
@dataclass(frozen=True)
class LabelDef:                       # LABELS: one per field / factor label
    name: str; version: str; arity: Literal[1, 2]; needs: frozenset[str]
    fn: Callable[[StateView, TokenIndex], Label]      # TokenIndex: token slots of the set(s) + their entity ids
    prov: str = "gt"                  # "gt" | "task" (public task structure) | "estimator:<name>"
@dataclass
class Label:  value: ndarray; valid: ndarray; prov: str; version: str      # [T,d] or [T,T,d]

@dataclass(frozen=True)
class ScenePart:                      # PARTS: composable scene components, each activating named dynamics / factors
    name: str; version: str
    activates: frozenset[str]         # factor / dynamics names it makes present: {"contact"}, {"contact", "support"},
                                      # {"articulation"}, {"handover"}, {"depth"} ...
    requires: frozenset[str] = frozenset()    # dynamics that must be active too (support requires contact)
    conflicts: frozenset[str] = frozenset()   # parts / dynamics it cannot co-occur with (declared, checked)
    envs: tuple[str, ...] = ()
    build: Callable[[SceneDraft, Generator], None]                # adds its objects / constraints / task events
    vary: Callable[[SceneDraft, Generator, str], list[SceneDraft]] | None = None
                                      # decoupling pairs: k copies differing ONLY in the named factor's value
def compose(parts, env, rng) -> SceneDraft   # the ONE composition operator: union of activates, requires closed,
                                             # conflicts rejected, layout resolved (non-overlapping placement),
                                             # task graph = merged events; no hand-written combinations

@dataclass(frozen=True)
class TransformDef:                   # TRANSFORMS: generic, factor-agnostic sample / label-stream transforms
    name: str; version: str
    fn: Callable[[Episode | Sample, Generator, dict], list[Sample]]
# reveal(schedule)         progressive information: targets become the posterior over candidates given evidence <= t
# surprise(rate, after)    contradiction after collapse: evidence flips, target switches, recovery is measured
# cf_swap(field)           counterfactual swap of a binding / id / attribute (today's binding_cf is cf_swap("binding"))
# noise(field, sigma)      estimator-style noise on a public field (+ var), for confidence scaling
# occlude(p)               drop tokens / mark invisible (null identities)
# subsample(t)             frame / knot subsampling for temporal factors
```

Every generated sample carries its provenance record: `LabelProv` + version per label, the parts and the ACTIVE SET
(the dynamics / factors present in that scene), the transforms applied, env, task, seed. A factor entry's `gen` names
its parts / transforms; `label` names its label. Decoupling pairs are `part.vary(draft, rng, factor)`; they are the
first rung of the curriculum (5.5), not the whole data design.

### 5.3 one pipeline stage and one mix

- `rrp stage run relations_data --config <run>`: for the run's resolved factor list, collect or relabel episodes
  (env × task × seeds from the config; relabelling reads recorded snapshots so no new physics is needed where
  snapshots exist), apply each factor's variations and transforms, write labelled shards
  `artifacts/relgen/<factor>/<version>/` + manifest. No per-factor scripts.
- `mixed_batches(main, curriculum, rng)` (`rrp.harness.data.mix`): each batch = the curriculum's share of composed
  relgen samples + the main (task) data, fraction-exact per batch and deterministic under the seed; labels a sample
  does not carry are masked. This is how a probe keeps its latent subspace while the policy trains.

### 5.4 task-conditioned edges, progressive reveal, surprise (first use of the transforms)

- Candidates: the featurizer emits candidate interaction edges (gripper→graspable, object→support, tool→target,
  source→destination, widget→drop-target) as a soft `EdgeSet` with prior mass; `task.next_contact` (op `bilinear`,
  form `aug`, gate `task`, readout `pair` with `soft_ce`) focuses on the edges the task needs.
- Reveal: target `q_t(e) ∝ π(e)·1[e consistent with evidence ≤ t]` (task text, completed events, observed contacts);
  `soft_ce(q_t, p̂_t)` sharpens the bias only as uncertainty resolves.
- Surprise: contradictory evidence after collapse (grasped object is not the expected one; drop target moves; UI label
  changes); `q_t` switches; metric = steps until KL(q_t‖p̂_t) < ε. Synthetic generator over task graphs + scene
  candidates (no physics) and a relabeller of recorded episodes (contact graph from `StateView.contacts`).

### 5.5 progressive composition curriculum (1 → 2 → k → full world)

Decoupling pairs isolate one dynamic; the full world mixes all of them. Training walks between the two: batches mix
scenes with increasingly many active factors / dynamics (contact → contact + support → contact + support +
articulation → ... → full task scenes), while lower-k data keeps the isolated subspaces maintained.

```python
@dataclass
class Curriculum:                     # lives in the one data-mix stage; JSON-able; recorded in the run provenance
    factors: tuple[str, ...]          # the dynamics / factors in play (from the resolved factor list + parts)
    k_schedule: Callable[[int], int] | list[tuple[int, int]]   # step -> max active-set size (ramp 1 -> K)
    subset_sampler: Literal["coverage", "uniform", "weakest"] = "coverage"
                                      # coverage: least-seen subsets of size k first; weakest: subsets containing the
                                      # factors with the lowest probe competence
    replay: dict[int, float] = {1: 0.2}           # share kept for lower-k levels (maintains isolated subspaces)
    full_world: Callable[[int], float] = ramp(0.0, 0.5)    # rising share of full task scenes (the main data)
    promote: dict[str, float] = {}    # per-factor readout-quality gate: a factor joins larger subsets only once its
                                      # probe metric passes (e.g. {"ix.contact": 0.9 acc, "geo.depth3d": 0.02 m})
    def active_sets(self, step, competence: dict[str, float], rng) -> list[tuple[frozenset[str], float]]
                                      # (active set, batch share) for this step; compose(parts covering the set)
```

- **Composition is an operator, not a list of combos**: `active_sets` picks sets; `compose` picks the parts that cover
  each set (minimal cover; requires closed; conflicts rejected; env compatibility from `ScenePart.envs` and label
  `needs`). A new part or factor joins every level automatically.
- **Promotion by competence**: the stage reads the latest per-factor readout metrics (`readout_metrics`, logged by the
  trainer) and gates promotion; a factor whose probe regresses is re-weighted toward lower-k replay.
- **Provenance**: every sample records its active set; `readout_metrics` are reported per factor × composition depth
  (k) and per active set.
- **Interference metric**: `interference(f, g) = metric_f(sets with f, without g) − metric_f(sets with f and g)` at
  matched k and step; the eval table flags pairs whose interference exceeds a threshold (a factor's probe degrading
  when another is added). Reported by `rrp suite relations-curriculum` from the saved per-sample metrics.
- **Orthogonal transforms**: reveal / surprise / cf_swap / noise / occlude apply at any level (they act on the
  composed sample's label stream), so epistemic training runs from k = 1 upward.
- **Composable now** (see the catalog's first wave): arm/dual MuJoCo {depth, orient, above, contact, held, support,
  force_flow, next_contact, handover(dual)} with parts `table_objects`, `stack`, `grasp_target`, `handover`,
  `camera_depth`; legged {foot contact, foothold, COM/support}; ComputerWorld {label_for, contains, focus, z-order,
  drag}. Articulation / mating / containment need new scene parts (drawer, peg-hole, bin) and are wave-2 catalog work.

## 6. first wave (implement now; everything else is a `planned` catalog entry)

Chosen because our envs can label them today (catalog with the rest: `research/relations_catalog.md`).

| factor | field / op / form | source (default) | label (needs) | envs | gen |
|---|---|---|---|---|---|
| `edge.*` ×17 (arm-rel-v1) | edges / edge / bias | given (public task graph, morphology) | – | arm, dual | cf_swap (binding) |
| `edge.*` ×5 (g1-dim-rel-v1: same_node, kin_parent, kin_child, same_assembly, mirror) | edges / edge / bias | given | – | simple | – |
| `route.own_assembly`, `route.assembly_reads` | assembly_id / same / mask | given | – | arm s0, legged, simple | – |
| `msg.incidence`, `id.slot_handle` | edges / – / message; entity_id / – / embed | given | – | arm, dual | – |
| `geo.pos3d` | pos3d / sqdiff+diff / aug | given (estimated for objects) | `pos3d` (poses) | arm, dual, legged, cw | noise |
| `geo.depth3d` | cam_uvd / sqdiff+diff / aug | probe | `cam_uvd` (poses, camera, depth_render) | arm, dual | part `camera_depth` (vary: same pixel, other depth), noise, occlude |
| `geo.orient` | orient / rel_rot / aug | given (assemblies), probe (objects) | `orient` (poses) | arm, dual, simple | part `table_objects` (vary: yaw) |
| `geo.normal_align` | normal / align / aug | probe | `contact_normal` (contacts) | arm, dual | – |
| `geo.above` | pos3d / order(gravity) / bias | given | – (derived) | arm, dual, legged | part `stack` |
| `id.same_body` | entity_id / same / aug | given | – | all | – |
| `kin.ancestor` | parent_id / ancestor / bias | given (morphology) | – | arm, legged, humanoid, simple | – |
| `ix.contact` | hidden / bilinear / aug | probe | `contact_pairs` (contacts) | arm, dual, legged (foot–ground) | reveal |
| `ix.support` | hidden / bilinear / aug | probe | `support_pairs` (contacts, poses, gravity) | arm, dual (support_insert, stacks) | part `stack` |
| `ix.held_by` | hidden / bilinear / aug | probe | `held_pairs` (contacts; ≥ 2 finger bodies) | arm, dual | – |
| `ix.force_flow` | support graph / flow / bias | probe | `support_pairs` closure | arm, dual stacks | part `stack` |
| `ix.handover` | hidden / bilinear / aug | probe | `handover_pairs` (contacts over time) | dual | – |
| `task.next_contact` | hidden / bilinear / aug, gate task | probe | `next_contact` (task runtime + contacts) | arm, dual, cw (drag) | reveal, surprise |
| `leg.foothold` | pos3d / sqdiff+diff / aug, gate task | probe | `foothold_next` (contacts, terrain) | legged, humanoid (foothold_steps, h_steps) | reveal |
| `ui.label_for`, `ui.contains`, `ui.focus_next`, `ui.above`, `ui.drag_to` | edges(ui-rel-v1) / edge / bias; drag: bilinear | given (public semantic.v1); drag: probe | `ui_tree`, teacher drag target | cw | cf_swap(label), surprise |

## 7. deploy guard

- Every Policy with `requires.privileged == False` calls `factors.assert_deployable()` in `reset()` and runs its nets
  with `RelCtx.deploy = True`: a resolved spec with source / control `gt` raises `PrivilegedInput(<factor>)`, and
  `TokenSet.label()` raises inside the forward. Teachers / oracles may use `gt` and are labelled.
- Collate functions attach labels only with `labels=True` (training); the featurizer never produces labels.
- `versions["factors"] = compat_hash(specs)` and `extra["factors"] = provenance(specs)` in checkpoints; the Episode
  provenance carries the same, so the room and reports show which factors and sources a result used.
- Red/green tests: `geo.pos3d source=gt` trains and raises in deploy mode; a deployed forward over a TokenSet whose
  `labels` are sentinels never touches them.

## 8. migration

### 8.1 relations, masks, flags

| today | factor | notes |
|---|---|---|
| `REL` 0..16 + `StructuralBias` in ContextEncoder / ActionBlock (`bias`, `bias_c`, `bias_e`) | `edge.<name>` ×17, preset `arm` (REL order) | vocab `arm-rel-v1` = the on-disk channel order of `PolicyInput.relations`; hard-coded ids (4, 16, 1, 13, `FOCUS_RELS`) become `REL[...]` lookups |
| `bias_mode` true / none / zero / reversed / rewired | control of `edge.*`: on / off / zero / reversed / rewired | one intended change: `rewired` now also rewires `act>act` (legacy skipped it). All existing configs use `true` |
| `structured` true / false | `msg.incidence` on / `serialized` | |
| `slot_handles` (PolicyConfig, LatentConfig) | `id.slot_handle` | |
| Ψ₀ `RELATIONS` + DimEncoder `StructuralBias` | preset `psi0` = `edge.same_node, edge.kin_parent, edge.kin_child, edge.same_assembly, edge.mirror` | vocab `g1-dim-rel-v1` |
| arm `LatentRealizer`, `LeggedRealizer` `-inf` own-assembly mask; Ψ₀ `READS` mask | `route.own_assembly` (`params.also=["body"]` for legged); `route.assembly_reads` (`params.reads=READS`) | |
| `SemanticReadout` + `aux`, `aux_weight` | `aux:arm-v1` readouts (`reads: hidden`) | |
| `PacketProbe` (+ `goal_effect`) | `ReadoutProbe` + `probes:arm-packet-v1` (+ `probe.arm.goal_effect`); `desired_delta` output alias deleted (its checkpoint key remap is data and stays) | |
| `AnchorProbe`, `LeggedProbe`, Ψ₀ `PacketProbe` (+ grasp head), `PointerProbe` | `probes:anchor-v1`, `probes:legged-v1`, `probes:psi0-v1`, `probes:pointer-v1` | |
| `semantic_weight`, `packet_semantic_weight`, `w_sem`, `probe_lv_min` | readout `weight`, `params.on`, `params.lv_min`; `RunConfig._check_variant` reads the specs | |
| `binding_cf`, `binding_contrast` (`nets/binding_aug.py`) | transform `cf_swap("binding")` with `mix`, `params.contrast` | |
| `$RRP_KINFEAT` | `feat.base_axes` (featurizer option in the factor list; env var deleted; `versions["kinfeat"]` folds into the hash) | |
| `realizer_anchor`, `drop_qd` | stay system 0 input options (not relational) | |

`LatentConfig.version()` must stay identical for every existing representation (bundle compatibility IDs): computed
from the legacy-equivalent dict when the specs equal the legacy defaults; the factor hash enters only when they differ
(test: every `configs/**` latent config hashes as before).

### 8.2 delete list

`StructuralBias`, `transform_relations`; `PolicyConfig.bias_mode / structured / slot_handles`; `N_REL` outside the
featurizer; `nets/latent_probes.py`, `nets/anchor_probes.py`, `LeggedProbe`, Ψ₀ `PacketProbe`, `PointerProbe`,
`SemanticReadout`; the four `block` / `run_block` copies; the three inline `-inf` masks; `nets/binding_aug.py` (→
`cf_swap`); the `$RRP_KINFEAT` env var; `desired_delta` alias outputs.

### 8.3 golden strategy (D-140: goldens, not reruns)

1. Commit 1, before any move: goldens on the CURRENT code with NON-zero relation weights (existing goldens run with
   zero-init `w`, so they cannot see the bias path): tiny FlowPolicy with seeded random `w` at every site, one golden
   per `bias_mode` (true / none / zero / reversed / rewired with `act>act` weights zeroed so the one intended change is
   invisible), over the arm featurizer's real relations and the dual MultiFeaturizer; a TargetEncoder; Ψ₀ DimEncoder
   with random `sb.w`.
2. Every migration commit leaves `tests/data/golden.json` byte-identical (no re-record) and the unit suite green.
3. Checkpoint test: state-dict key lists of the old classes (frozen in the test) load strictly into the new ones.
4. Config test: every `configs/**` / `dags/**` policy block resolves to the same effective factor list as its legacy
   flags; the codemod then rewrites those files to `factors:`; `PolicyConfig.from_dict` maps legacy keys only for
   configs stored inside checkpoints (on-disk data).

## 9. files (end state)

```
src/rrp/policies/relations/__init__.py     docstring only
src/rrp/policies/relations/base.py         Prov, FieldDef, TokenSet, EdgeSet, RelCtx, Algebra, FactorDef, FactorSpec,
                                           FIELDS / OPS / FACTORS / PRESETS, resolve, compat_hash, provenance, guard
src/rrp/policies/relations/ops.py          the operators (3.2) and forms; FactorSite
src/rrp/policies/relations/catalog.py      the FactorDef / FieldDef table (sections per unit) + presets
src/rrp/policies/relations/probe.py        ReadoutDef, ReadoutProbe, readout_loss / readout_metrics
src/rrp/policies/nets/attention.py         MHA (+ q_aug / k_aug), RelBlock
src/rrp/envs/base.py                       + StateView, EntityState, ContactState, JointState, Camera
src/rrp/harness/data/relgen/__init__.py    LabelDef, Label, ScenePart, TransformDef, LABELS / PARTS / TRANSFORMS,
                                           compose, Curriculum, TokenIndex
src/rrp/harness/data/relgen/transforms.py  reveal, surprise, cf_swap, noise, occlude, subsample     (unit G)
src/rrp/harness/data/relgen/{geometry,contact,task,body,ui}.py   label fns + scene parts per family
src/rrp/harness/data/relgen/curriculum.py  Curriculum, active_sets, interference                    (unit R11)
src/rrp/harness/data/mix.py                mixed_batches
```


## 10. fanout plan (Sonnet-sized units; the owner lifted the agent cap, D-143)

**Foundation (lead, this branch; everything below depends on it):** F1 goldens on the current code; F2
`relations/base.py`, `relations/ops.py` with ALL operators of 3.2 and all forms / controls / gates / confidence /
field readouts, `relations/catalog.py` (legacy entries, presets, one pre-created section per unit), `FactorSite`;
F3 `MHA` q/k augmentation, `RelBlock`; delete `StructuralBias` / `transform_relations`; arm/dual flow + E + Ψ₀
DimEncoder on FactorSite; `PolicyConfig.factors` + legacy mapping + configs/dags codemod + factor hash in checkpoints;
deploy guard; `rrp factors list|show`; F4 interface skeletons with tests: `relations/probe.py` (ReadoutProbe,
equivalence-tested against `PacketProbe`, not swapped in), `envs.base.StateView` + dataclasses,
`harness/data/relgen/__init__.py` (LabelDef, Label, ScenePart, TransformDef, registries, `compose` signature),
`harness/data/mix.py` (signature). Units never edit `base.py` / `ops.py`; they add entries in their own section of
`catalog.py` and in their own relgen module. A unit that needs an operator change reports it to the lead instead.

Rules for every unit: worktree `~/work/rrp-wt/rel-<id>` on `track/rel-<id>` from origin/main;
`export PYTHONPATH=$PWD/src:$PWD`; host = git / editing / unit suite only (CUDA hidden; no training or simulation
beyond unit-test fixtures; smokes go to the peer via `scripts/peer_run.sh`); `pytest tests/unit` exit code 0 before
merge; `tests/data/golden.json` untouched unless the brief says otherwise; no shims, no new files beyond those listed;
notes in `research/tracks/rel-<id>.md`; merge with `git fetch origin && git rebase origin/main && git push origin
HEAD:main`; commit trailer as in AGENTS.md. Shared files are shared by disjoint functions only (noted), which rebases
cleanly.

| id | unit | deps | owns | acceptance | est. |
|---|---|---|---|---|---|
| R1 | arm/dual probes → ReadoutProbe | F | `nets/latent_probes.py` (delete), `harness/eval/{hooks,ladder,dual_latent_eval,latent_causal,edit_harness}.py`, `cli/{latent,dual_latent}.py`, `viz/record.py` (probe calls only), `policies/bundles.py` (`load_representation` only), probe lines of `harness/train/latent_train.py` | goldens unchanged; old PacketProbe state dicts load strictly; `probes:arm-packet-v1` metrics equal `probe_metrics` on a fixture batch | 4 h |
| R2 | arm latent config + flags → specs | R1, R9 | `nets/semantic_latent.py`, `nets/binding_aug.py` (delete), `nets/latent_batch.py`, `harness/train/{latent_train,behavior,joint_adapt,sft,adapt,vlm_train}.py`, `harness/data/{packed,dual_latent,chunks}.py`, `harness/eval/latent_counterfactuals.py`, `core/{runconfig,provenance}.py`, `configs/latent/**`, `dags/**` (latent keys) | `LatentConfig.version()` identical for every config under `configs/`; `_check_variant` equivalent; `cf_swap("binding")` batches equal `binding_aug` batches on a fixture; no `semantic_weight` / `probe_lv_min` / `binding_cf` in `src/` | 5 h |
| R3 | arm system 0 on RelBlock + `route.own_assembly` | F | `policies/system0.py`, `policies/latent.py` | `latent.system0.*` goldens unchanged; realizer state dicts load strictly | 2 h |
| R4 | legged nets | F | `nets/{legged_latent,legged_bc}.py`, `policies/legged.py`, `features/legged.py`, `harness/train/{legged_latent_train,legged_bc,legged_dagger}.py`, `harness/eval/legged_latent_eval.py`, `policies/bundles.py` (`load_rep` only) | legged goldens unchanged; LeggedProbe checkpoints load via the key map; `probes:legged-v1` metrics equal `probe_metrics` on a fixture | 4 h |
| R5 | Ψ₀ nets | F | `policies/psi0/{nets,train,data}.py`, `bodies/g1_simple.py` | `test_psi0.py` green; Realizer output byte-identical on a seeded fixture; stage-A checkpoints load (`load_tolerant` paths); `probes:psi0-v1` | 3 h |
| R6 | pointer nets | F + pointer track merged | `policies/pointer.py`, `harness/train/pointer.py` | pointer unit tests green; PointerProbe checkpoints load; `probes:pointer-v1` | 3 h |
| R7 | StateView: MuJoCo | F | `envs/mujoco/{session,dual,legged}.py` (`state_view` + helpers), `tests/unit/test_state_view.py` | contract test on arm / dual / legged fixtures: entities with poses, contacts with normals, joints, camera K/T, `render_depth` shape; caps declared; views never reachable from the featurizer | 3 h |
| R8 | StateView: Warp, ComputerWorld, SIMPLE | F | `envs/warp/tracker_env.py`, `envs/computerworld.py` (`state_view` only), `envs/simple/__init__.py` (`state_view`) | per-backend contract tests (CW on the fixture scene; SIMPLE on a recorded `truth()` dict; Warp on a 2-env CPU batch if available, else skip-marked) | 3 h |
| R9 | generic transforms | F | `harness/data/relgen/transforms.py`, `tests/unit/test_relgen_transforms.py` | reveal targets = Bayes posterior on a toy candidate set; surprise flips after collapse and the recovery metric counts steps; cf_swap / noise (+var) / occlude / subsample are pure and seed-deterministic; provenance appended | 3 h |
| R10 | relgen stage + shards + mix | F | `harness/pipelines/relations.py` (stage `relations_data`), `harness/data/mix.py`, shard/manifest format, `tests/unit/test_relgen_stage.py` | 2-episode fixture run writes shards + manifest with per-sample provenance (active set, label versions); `mixed_batches` fractions exact and seed-deterministic; missing labels masked | 4 h |
| R11 | curriculum + composition + interference | R10 | `harness/data/relgen/curriculum.py`, `relgen/__init__.py` (`compose` body), `rrp suite relations-curriculum` (in `cli`), `tests/unit/test_curriculum.py` | `compose` closes requires, rejects conflicts, minimal cover; k ramp, coverage sampler, replay shares, promotion gate and full-world share follow the schedule exactly (table test); interference computed on synthetic per-sample metrics flags the planted pair | 4 h |
| R12 | arm/dual fields | F | `features/{featurizer,multi,kinfeat}.py` (kinfeat → `feat.base_axes`), `nets/batch.py` (TokenSet fields from token columns), `envs/mujoco/sensors.py` (camera projection) | featurizer goldens unchanged; fields `pos3d` (+`.var`), `cam_uvd`, `orient`, `entity_id`, `assembly_id` with correct provenance on the arm fixture; projection test against MuJoCo camera math; `$RRP_KINFEAT` gone, `feat.base_axes` reproduces its features | 4 h |
| R13 | geometry factors | R12 | `catalog.py` §geo (`geo.pos3d`, `geo.depth3d`, `geo.orient`, `geo.normal_align`, `geo.above`), `tests/unit/test_relations_geo.py` | factors resolve on arm / dual; deploy guard red/green with `source=gt`; field readout (`readout_layer`) wiring test; ≤10 min peer smoke of arm flow with `geo.depth3d source=probe` (loss decreases, depth probe error logged) | 3 h |
| R14 | geometry labels + parts | R7 | `harness/data/relgen/geometry.py` (labels `pos3d`, `cam_uvd`, `orient`, `contact_normal`; parts `table_objects`, `camera_depth` with `vary`) | labels on arm / dual fixtures match hand-computed values; `camera_depth.vary` keeps pixel, changes depth; provenance recorded | 3 h |
| R15 | membership / graph factors | R12 | `catalog.py` §graph (`id.same_body`, `id.same_assembly`, `kin.ancestor`, `kin.sibling`, `kin.mirror` generalized) | closure / hop values on the arm and G1 morphology fixtures; factors resolve on arm / Ψ₀ / legged sets | 2 h |
| R16 | contact / grasp / handover | R7, R12 | `catalog.py` §contact (`ix.contact`, `ix.held_by`, `ix.handover`), `harness/data/relgen/contact.py` (labels `contact_pairs`, `held_pairs`, `handover_pairs`; part `grasp_target`) | labels on grasp / dual-handover fixtures; bilinear readout trains on a synthetic batch (loss decreases, host-cheap) | 3 h |
| R17 | support / force flow | R7, R12 | `catalog.py` §support (`ix.support`, `ix.force_flow`), `harness/data/relgen/support.py` (label `support_pairs` + closure; part `stack` with `vary`) | stack fixture: support pairs and upstream / downstream closure correct; `stack.vary` changes only the order | 3 h |
| R18 | task / temporal / epistemic | R9, R12 | `catalog.py` §task (`task.next_contact`, `time.same_track`), `harness/data/relgen/task.py` (label `next_contact`; candidate-edge emission), `nets/batch.py` (candidate edges only, after R12) | next_contact candidates on pick_place / dual fixtures; reveal and surprise applied to its targets; gate `task` changes the bias between two task texts on a fixture | 4 h |
| R19 | legged labels + foothold | R4, R7 | `catalog.py` §legged (`leg.foothold`, `leg.com_support`), `harness/data/relgen/body.py` (labels `foothold_next`, foot `contact_pairs`, COM / support polygon; part `terrain_steps`), Warp height scan → label block (`envs/warp/*` obs layout) | labels on legged fixtures; the deployable Warp `vec` no longer contains the height scan (privileged layout test) | 4 h |
| R20 | UI factors | R6, R8 | `catalog.py` §ui (`ui.label_for`, `ui.contains`, `ui.focus_next`, `ui.above`, `ui.drag_to`), `harness/data/relgen/ui.py`, `envs/computerworld.py` (public UI fields / `ui-rel-v1` edges) | edge extraction on the CW fixture scene; labels; cf_swap(label) and surprise on UI labels | 3 h |
| R21 | viz: factor / attention inspection | F | `viz/record.py` (factor records), `viz/export.py`, `viz/room/**` | room shows per-factor logit maps per head for a recorded step, the factor list with sources / controls, privileged / estimated badges, competence by composition depth (from R11 outputs when present); exporter unit test | 4 h |

Waves: W1 = R1, R3, R4, R5, R7, R8, R9, R10, R12, R21 (all only need F); W2 = R2, R11, R13, R14, R15, R16, R17, R18,
R19 as their deps merge; W3 = R6, R20 after the pointer track merges. Catalog wave 2 (articulation, mating,
containment, material, tool→target, cause→effect: new scene parts + labels) is planned after W2 reports.

### briefs (one paragraph each; read this page's sections 2–5 and `research/relations_catalog.md` first)

- **R1.** Replace `nets/latent_probes.PacketProbe` everywhere with `relations.probe.ReadoutProbe` configured by the
  preset `probes:arm-packet-v1` (and `probe.arm.goal_effect` when the old config had `goal_effect=True`). Keep the
  state-dict layout (the foundation's equivalence test proves it) so `bundles.load_representation` loads old
  checkpoints strictly; `probe_loss` / `probe_metrics` become `readout_loss` / `readout_metrics` over the specs; drop
  the `desired_delta` output alias but keep the checkpoint key remap in the load path. Do not change what any hook
  reports (metric key names stay).
- **R2.** Move `LatentConfig.semantic_weight / probe_lv_min / binding_cf / binding_contrast / slot_handles` and the
  top-level `packet_semantic_weight` / `aux_weight` into the run config's `factors:` list (readout `weight`,
  `params.on`, `params.lv_min`, transform `cf_swap("binding")` with `mix`, `id.slot_handle`). `LatentConfig.version()`
  must return the same string for every existing config (write the test first: freeze today's versions of all configs
  under `configs/`). Port `binding_aug` onto R9's `cf_swap` and delete it; replace hard-coded relation ids with
  `REL[...]`; rewrite `RunConfig._check_variant` and `TRAINING_FLAG_KEYS` on the specs; codemod configs / dags.
- **R3.** Replace the inline `-inf` own-assembly mask and the private block of `LatentRealizer` with `RelBlock` and
  the `route.own_assembly` factor (preset `s0-arm`); parameter paths must not change; goldens byte-identical.
- **R4.** Legged: `block` / `run_block` → `RelBlock`; the realizer's own/body-assembly mask → `route.own_assembly`
  with `params.also=["body"]`; `LeggedProbe` → `ReadoutProbe` + `probes:legged-v1` with a data-level key map for old
  checkpoints; legged goldens byte-identical.
- **R5.** Ψ₀: `DimEncoder` already runs on FactorSite (foundation); move `Realizer`'s `READS` mask to
  `route.assembly_reads`, the private block to `RelBlock`, the packet probe to `probes:psi0-v1` (grasp head as optional
  specs); keep `load_stage_a` / `load_tolerant` working; `RELATIONS` stays the vocabulary `g1-dim-rel-v1`.
- **R6.** After the pointer track merges: pointer `Block` → `RelBlock` (empty preset), `PointerProbe` → `ReadoutProbe`
  + `probes:pointer-v1`, `--w-sem` / `--lv-min` → specs; old checkpoints load.
- **R7.** Implement `state_view()` on the MuJoCo sessions against `envs.base.StateView`: entities (bodies / objects /
  assemblies with ids matching the scenario's entity ids), contacts (geom pairs → entity pairs, world normal, force via
  `mj_contactForce` when "forces"), joints, cameras (intrinsics from `cam_fovy`, extrinsics from `cam_xpos/xmat`,
  `render_depth` via `mujoco.Renderer` depth mode, EGL only on the peer — host tests use the camera math only), and
  `token_entity` for scene slots. It is capability-gated (`privileged_truth`) and never imported by `policies/`.
- **R8.** Same contract for Warp (per env index), ComputerWorld (`truth()` scene: nodes, parents, z, focus, bounds) and
  SIMPLE (the worker's `truth()` dict); tests on fixtures / recorded dicts only.
- **R9.** Implement the generic transforms in `relgen/transforms.py` as pure functions over samples with a candidate
  set, evidence events and label streams: `reveal(schedule)` produces posterior targets q_t; `surprise(rate, after)`
  injects contradictory evidence after collapse and records the switch time; `cf_swap(field)` swaps bindings /
  ids / attributes consistently across tokens, labels and edges; `noise(field, sigma)` adds noise and sets `.var`;
  `occlude(p)` masks tokens; `subsample`. Each appends its provenance; math tests as in the table.
- **R10.** The one data stage: `harness/pipelines/relations.py` registers `relations_data`, which reads the run
  config's factors, resolves labels / parts / transforms, collects or relabels episodes through `StateView`, writes
  shards `artifacts/relgen/<factor>/<version>/` with a manifest and per-sample provenance (active set, label versions,
  transforms, env, task, seed). `mix.mixed_batches` interleaves curriculum shares with the main data exactly.
- **R11.** `Curriculum` (5.5): k schedule, subset sampler (coverage / uniform / weakest), replay shares, promotion by
  per-factor readout competence, full-world share; `compose(parts)` (requires closure, conflicts, minimal cover, env
  compatibility); `interference(f, g)` from per-sample metrics by active set; `rrp suite relations-curriculum` prints
  competence by composition depth and flags interfering pairs.
- **R12.** Produce the arm / dual token-set fields in the collate path from existing token columns (no dataset
  rewrite): `pos3d` (+`.var` from the slot covariance), `cam_uvd` (project through the declared scene camera; add the
  projection helper to `envs/mujoco/sensors.py`), `orient` (assembly frames; objects unknown → invalid),
  `entity_id`, `assembly_id`, each with `FieldDef` provenance. Replace `$RRP_KINFEAT` by the factor-list option
  `feat.base_axes` (featurizer reads the resolved spec; `versions["kinfeat"]` folds into the hash).
- **R13.** Declare the geometry entries (sqdiff+diff on `pos3d` and `cam_uvd`, rel_rot on `orient`, align on normals,
  order along gravity), wire field readouts (`source=probe`, `readout_layer`) through the foundation hook, write the
  guard and resolution tests, run the peer smoke.
- **R14.** Labels written once against `StateView`: `pos3d` / `cam_uvd` (projection of true positions; depth from
  `render_depth` at the pixel when the cap exists, else analytic), `orient` (true quats), `contact_normal`; scene parts
  `table_objects` (yaw vary) and `camera_depth` (vary depth at fixed pixel) with declared `activates`.
- **R15.** Membership / graph entries using `same`, `hop`, `ancestor` over `entity_id`, `assembly_id` and the morphology
  parent field; tests on the arm and G1 morphologies.
- **R16.** Contact / grasp / handover labels from `StateView.contacts` (held = ≥ 2 finger bodies of one hand
  assembly, like today's `held` label), entries with bilinear readouts, part `grasp_target`.
- **R17.** Support (`a` supports `b` if in contact and the contact normal is within 30° of gravity-up at `a`'s top),
  force-flow closure (upstream / downstream along support), part `stack` (n objects, vary order), entries.
- **R18.** Candidate interaction edges (manipulator→graspable, object→support / destination) emitted in the collate
  path as a soft public EdgeSet; `task.next_contact` (bilinear, gate task, `soft_ce` readout) with reveal / surprise
  targets; `time.same_track` for multi-step token histories (entry + label; used when a net has history tokens).
- **R19.** Legged labels (next foothold cell, per-foot contact, COM projection inside the support polygon), entries
  `leg.foothold` / `leg.com_support`, part `terrain_steps`; move the Warp height scan out of the deployable `vec` into
  a label block (privileged layout test).
- **R20.** ComputerWorld public UI fields (parent, z-layer, focus rank) and `ui-rel-v1` edges in the env adapter,
  UI entries and labels (label_for, contains, focus_next, above, drag_to from the teacher), `cf_swap(label)`.
- **R21.** Record `FactorSite.contributions` for chosen steps into viz records, export per-factor per-head logit maps,
  the factor provenance per run and competence-by-depth tables; room panels with unmistakable source badges.
