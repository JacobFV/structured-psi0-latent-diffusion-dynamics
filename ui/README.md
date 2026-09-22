# RRP workbench UI

React + TypeScript (Vite) front end for the loopback workbench service in `src/rrp/service/`.
The backend is authoritative: the UI only sends requests (commands, versioned graph edits,
probes) and renders backend state; it never computes execution status or success.

- `src/transport/` — HTTP client (`x-rrp-token` on mutations), WebSocket client (snapshot first, then
  sequenced messages; no outgoing queue, so reconnects never replay commands), and
  `types.ts` **generated** by `scripts/export_ui_types.py` from `rrp.service.schemas` (do not edit).
- `src/scene/` — three.js scene from `/scene` geoms posed by snapshot/`state_update` body poses, orbit
  camera, click-to-select, gizmo with an explicit *target goal* vs *teleport object (contaminates
  evaluation)* switch, joint sliders per controller group, EE target entry, low-load stream mode.
- `src/graph/` — React Flow event graph (solid = requires completed, dashed = requires active,
  dotted = output binding), add event/entity, connect/remove dependencies, bind role slots,
  priority, request/cancel execution, receipts, debug override (confirmed, red, contaminates).
- `src/inspector/` — selection, morphology tree, public observation, probes, actor routing /
  controller ownership, message log, opt-in privileged overlay (display only).
- `src/playback/`, `src/resources/` — recorded timeline by control source, export, physics replay,
  read-only broker resources.

Stable provenance colors: teacher = amber "SCRIPTED TEACHER (privileged)", learned = blue,
user = green, debug = red "DEBUG — excluded from evaluation", hold = grey.

## Build (always inside a lease)

```bash
cd ~/work/relational-robot-policy
NB=$HOME/.local/share/vite-plus/bin
PYTHONPATH=src python3 -m rrp.cli ops run --cpu 2 --mem 4G --label ui-build --max-seconds 1800 \
  --env PATH=$NB:/usr/bin:/bin --env HOME=$HOME --env npm_config_cache=$PWD/.cache/npm \
  -- /bin/bash -c "cd $PWD/ui && npm ci && npm run build"
```

`npm run build` type-checks and writes `ui/dist`, which the backend serves at `/` and `/assets`.
After changing `src/rrp/service/schemas.py`, regenerate types: `.venv/bin/python scripts/export_ui_types.py`.

## Manual launch

```bash
cd ~/work/relational-robot-policy
PYTHONPATH=src python3 -m rrp.cli ops run --cpu 1 --mem 2G --label workbench --max-seconds 21600 --detach \
  -- .venv/bin/python -m rrp.cli workbench --port 8765
# then open (loopback only; use an SSH tunnel from another machine):
echo "http://127.0.0.1:8765/?token=$(cat ops/workbench-token)"
```

Add `&lowload=1` to start in server-rendered stream mode (JPEG frames polled at ≤ 4 fps) instead
of the in-browser WebGL scene. Keyboard: all controls are native buttons/inputs (Tab order);
`.` steps one control step when focus is not in a form field. There are no destructive shortcuts;
reset, teleport, remove event and debug override all ask for confirmation.

## Browser acceptance tests (real backend, headless chromium, software GL)

```bash
cd ~/work/relational-robot-policy
NB=$HOME/.local/share/vite-plus/bin
# one-time: browsers into the repo cache
PYTHONPATH=src python3 -m rrp.cli ops run --cpu 2 --mem 4G --label ui-pw-install --max-seconds 1800 \
  --env PATH=$NB:/usr/bin:/bin --env HOME=$HOME --env npm_config_cache=$PWD/.cache/npm \
  --env PLAYWRIGHT_BROWSERS_PATH=$PWD/.cache/ms-playwright -- /bin/bash -c "cd $PWD/ui && npx playwright install chromium"
# run (starts the backend on 127.0.0.1:8791 with a per-run token, runs tests, stops the backend)
PYTHONPATH=src python3 -m rrp.cli ops run --cpu 3 --mem 6G --label ui-browser-tests --max-seconds 1800 \
  --env PATH=$NB:/usr/bin:/bin --env HOME=$HOME --env npm_config_cache=$PWD/.cache/npm \
  --env PLAYWRIGHT_BROWSERS_PATH=$PWD/.cache/ms-playwright -- .venv/bin/python tests/browser/run_browser_tests.py
```

Outputs: `artifacts/receipts/browser/{playwright-report.json,summary.json,workbench-demo.png}` and
`artifacts/video/workbench-demo.webm` (per-test videos stay in `.cache/browser-run/test-results/`).
