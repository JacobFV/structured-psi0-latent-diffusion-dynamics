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
  only when the exporter module is absent or when the URL has `fixture=1`.
- A missing document or signal shows "no data" and the path that was expected. Nothing is interpolated, smoothed or estimated.
  Every source label (scripted teacher, oracle, learned, BC, tracker, synthetic) is spelled out, and signals the recorder
  marks privileged are flagged as display-only.

## Layout (v4, after IBM-2's workbench)

The left sidebar holds a view picker at the top and that view's controls underneath. There are no top bars and no nested
tabs, and narrow screens get a drawer. The type scale is 10/11/12 px; only the board's headline numbers are larger.

Four views (keys 1–4); older addresses redirect:
1. **Board** (`#board`): one screen at 1440×900: headline tiles (arm v6, new arm, new gripper, legged, Ψ₀ step 2,
   a tiny curriculum tile, status), the policy × env matrix, relation factors, DAGs, leases and latest decisions.
2. **Runs** (`#runs`): the IBM-2 run sidebar (Environment, a Task listbox tinted by outcome, Result, search tokens, the run
   list, "compare with…"). For the selected run it shows one stack of linked panels sharing a time cursor:
   - pipeline flow, morphology, packet structure;
   - 3D, top-down map, gait, joint heatmap, phase portrait;
   - forces, velocities, grasp, energy;
   - packet heatmaps, probe vs truth;
   - events, edit and B − A traces;
   - video, evidence.
3. **Training** (`#training`): peer vitals as charts, then training curves.
4. **Evaluations** (`#evaluations`), four lenses: Success (legged cv2, arm v6, transfer), Route radar, Causal effects,
   Relation factors.

Every panel carries a LIVE/STALE/SNAPSHOT/FIXTURE/NO DATA badge.
