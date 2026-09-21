# interactive robot and task workbench

## purpose and stack

Build a researcher-facing control and causal-inspection workbench, not a dashboard theme. Default stack: React + TypeScript + a graph editor such as React Flow, a scene view using Three.js where permitted, and a Python FastAPI/WebSocket backend. Verify current APIs and ARM64/browser support. Keep a low-load peer-rendered image-stream mode for the protected host. Backend runtime is authoritative; UI never computes success by guessing from graph layout.

Use a focused layout: large robot/scene view, adjacent editable event graph, a shared inspector and a bottom playback/resource strip. Stable colors identify estimated/privileged/teacher/learned modes, not research performance. Interaction should be keyboard-accessible with visible errors and no hidden destructive shortcuts. Spend time on useful selection, camera navigation, clear bindings and responsive state—not decorative animation.

## must-work interactions

Load a body/task/model/controller; inspect morphology tree and coarse assemblies; select joints/links/manipulators/objects in either scene or inspector; move an object/target with a gizmo; request end-effector movement; pause/resume/step/reset; record/replay; and display native actions alongside inferred task roles.

Direct joint/target dragging is a user command routed through validated controller limits. Moving an object physically is a simulator intervention and contaminates that autonomous evaluation run. A mode explicitly distinguishes target-goal editing from teleporting the object. Never record either silently as autonomous policy experience.

The task graph shows operator, ordered roles, current receipts, preconditions/invariants/effects, required completed/active events, output bindings, status, reason for waiting/failure, and uncertainty. Users can add/remove events and dependencies, bind role slots, select priorities, request execution, inspect a produced frame and connect it to a downstream event.

Dragging screen position changes layout only. Selecting a node changes inspection only. Requesting execution checks guards and ownership. A future event request cannot mark predecessors complete. A deliberate debug override is separately confirmed, brightly labeled and excluded from normal evaluation metrics.

## versioned edit transaction

Client sends `GraphEdit(expected_version, operations, request_id)`. Backend pauses new policy-chunk dispatch, validates schema/references/cycles/resources, commits the new graph version, invalidates context caches, drops unsent stale chunks and resumes/replans from current observed state. Already executed physical actions are not rolled back by undoing graph text. An actual rewind restores a full snapshot and is labeled a different episode branch.

Repeated request_id is idempotent. A stale expected_version returns a conflict with the current version, without partial mutation. A failed edit leaves the old graph unchanged. WebSocket reconnect rehydrates an authoritative snapshot, then consumes monotonically sequenced updates; clients cannot replay stale commands automatically.

## semantic inspection

Queries: visible, looking_at, focused_on, acting_on, held_by(manipulator), current task(s) for manipulator, relative pose/distance, active reference frame, contact mode, desired next delta, feasibility and uncertainty. Show null/multiple answers and unavailable estimates. Querying an object through two prompts points to one canonical slot; presentation vectors do not create new object identities.

Optional object QA runs only when asked or during training evaluation, not on every frame. Show estimated frames with confidence/age/slip; optional privileged overlays have a separate toggle and cannot feed the policy. Attention visualizations are explicitly labeled diagnostics; expensive full attention export is opt-in and disabled during measured low-latency runs.

## endpoints and events

Implement `/health`, `/api/capabilities`, `/api/robots`, `/api/sessions`, `/api/sessions/{id}/snapshot`, `/api/sessions/{id}/commands`, `/api/sessions/{id}/graph`, `/api/sessions/{id}/probes`, `/api/runs`, `/api/resources`, and `/ws/sessions/{id}`. Typed messages include sequence, session/graph/runtime versions and source (teacher/learned/user/debug). Validate request size, content type, bounded queues and timeouts. Reject arbitrary file paths/commands in robot/task selectors.

Bind 127.0.0.1 only. Authenticate local mutation requests with a random per-session token stored outside logs; validate WebSocket origin; do not expose the backend to a LAN as a convenience. Use SSH tunnels for remote operation. A slow client drops intermediate render frames, never control messages. On disconnect, policy/controller follows its declared hold/cancel behavior, not indefinite playback of stale chunks.

## replay and export

Timeline links video, observations, policy chunks, event receipts, edits, contact frames, resource usage and model/controller revisions. Export selected episode plus graph and reproduction command. Record graph/state transitions as append-only events. Teacher execution, privileged overlays and user interventions remain in the export manifest.

## browser acceptance tests

Playwright tests use a real running backend with a tiny MuJoCo fixture, not static mocks: select a robot; start/step; inspect one joint; move a target; create and connect an event; reject a cycle; reject an unsatisfied execution request; change actor binding and observe changed command routing; edit during a queued chunk and reject the stale chunk; disconnect/reconnect; replay; inspect unknown contact; and verify teacher/learned labels.

Host tests use a separate low-resource software-render browser under the project quota. Pixel-perfect snapshots are secondary to semantic test assertions and recorded interaction videos. All tests run headless and the interactive UI remains usable manually through a documented command.
