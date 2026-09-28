# rrp room (frontend)

Live visualization room for rrp (D-131, contract `viz/CONTRACT.md`). Vite + React + TypeScript + three.js + recharts.
It is read-only: it never starts, stops or edits jobs or files.

## Run

```
cd viz/room
npm install                                   # once (host: run it with nice -n 10)
npm run dev                                   # http://127.0.0.1:3013 (local only)
RRP_ROOM_HOST=0.0.0.0 npm run dev             # the same room on the network, port 3013 (D-132)
```

The Vite plugin (`rrp-api-plugin.ts`) serves `/api/*`. For each document it runs
`python -m rrp.viz.api get <doc> [--live] --max-age S` (repo root, `PYTHONPATH=src`, niced, one export at a time, 15 s cache,
live 10 s), then serves `viz/data/<doc>.json`, with an ETag so an unchanged document is not sent again. Replays come from
`~/work/rrp-data/viz/replays/**/<id>.json[.gz]` and videos from `artifacts/video/` (name allowlist, Range requests); markdown
comes from the contract's allowlist. Environment overrides: `RRP_ROOT`, `RRP_PYTHON`, `RRP_REPLAYS`, `RRP_PSI1Z`.

Static build: `npm run snapshot` (runs the full exporter) and then `npm run build`. `dist/` then carries `data/*.json` and says
"snapshot as of …". Replays and the docs reader need the live API.

Checks: `npm run typecheck`, `npm run build`, and the server-side render check of every view with the real documents and with
the fixtures (the host runs no browser):

```
RRP_NO_SNAPSHOT=1 npx vite build --ssr scripts/render-check.tsx --outDir node_modules/.cache/render-check
node node_modules/.cache/render-check/render-check.js ../..
```

## Data honesty

- Every view shows its mode: `live export` (with its age and git sha), `STALE` (the latest export failed, so the previous file
  is served along with the error), `SNAPSHOT` (static build) or `FIXTURE`.
- Fixtures (`fixtures/`, made by `scripts/make_fixtures.py`) are synthetic and labelled as fixtures everywhere. They are used
  only when the exporter module is absent or when the URL has `fixture=1`. The theatre's fixture demo opens only on request
  (`#theatre?demo=1`) while no replay has been recorded.
- A missing document or signal shows "no data" and the path that was expected. Nothing is interpolated, smoothed or estimated.
  Every source label (scripted teacher, oracle, learned, BC, tracker, synthetic) is spelled out, and signals the recorder
  marks privileged are flagged as display-only.

## Layout (terminal redesign, owner feedback 2026-09-28)

Dark-first "trader terminal", modelled on IBM-2's board: a top bar with the numbered views; a live ticker of real events
(broker lease/admission events, non-ok watchdog samples, DAG node starts/ends, new decisions), which pauses on hover and stays
static under reduced motion; then the view. The landing page `#board` shows everything at once:
- a KPI strip: peer GPU, memory, PSI, admission, leases, DAG done/total per active workstream, arm grasp_v2 semfix|nosem,
  halt effect per legged body (sem|nosem), Ψ₀ step 2, gates, decisions, roadmap, claims, replays;
- seam panels: vitals sparklines, leases with inline memory bars (current, peak, memory.high), DAG node-state bars, the
  results tile wall (Δ against a chosen baseline route), arm lineages, the causal-edit mini forest, the latest training
  sparklines, the robustness break-point strip, Ψ₀, gates, decisions, claims and caveats, and the theatre.

Every panel carries a LIVE/STALE/SNAPSHOT/FIXTURE/NO DATA badge, and clicking it opens its full view. Keys: `1`–`0` switch
views, `/` focuses the first filter, and `[` `]` step through replays in the theatre.

## Views (deep links: `#<view>?<state>`)

1. `#board` (landing; see above). `#overview` (not numbered): claims by status, caveats, roadmap, key numbers from STATUS, latest decisions, workstreams.
2. `#live`: peer vitals, admission, leases (declared vs measured, memory.high and OOM), broker and watchdog events, the
   stale flag, and run-DAG node-state bars per workstream (`tab=dags`).
3. `#results`: heatmap of rate with CI, any row/column dimensions, filters, a delta mode A → B (Newcombe CI) and every
   markdown compare/gate table.
4. `#theatre`: a three.js replay (primitives and inline meshes, contact markers, object and base trails, follow/orbit),
   timelines kept in sync (targets/positions, contacts, packet PCA in 3D, probes, phase, events, edit shading, progress), an
   A/B compare on one clock (`a=`, `b=`, `t=`), the video fallback and the video library (`tab=videos`).
5. `#edits`: forest plots per body · variant · metric with controls and permutation tests.
6. `#training`: logged series per run, grad norm with shaded clip-active steps, clip scale, lr, α, grad-health reports.
7. `#robustness`: break-point curves per factor, motion quality, route comparisons, variant-level paired differences.
8. `#physics`: tracker validation and contact gates, gate criteria, dataset gates, backfill tables.
9. `#psi0`: Ψ₀ reproduction and step-2 runs, P-decisions, the D ↔ P crosswalk, psi1z notes.
10. `#knowledge`: decisions timeline, crosswalk, roadmap, backlog, strategy, STATUS and a docs reader.
