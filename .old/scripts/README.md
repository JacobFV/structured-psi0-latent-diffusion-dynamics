# .old/scripts — finished drivers (D-145 purge unit P3)

Old path `scripts/...`. Moved in by purge unit P3 (`git log --follow -- .old/scripts/<file>`).

| file | what it was | cited by |
|---|---|---|
| `demo/*` (13 files) | sprint demo page: `build_page.py` built `docs/demo/` from raw results; `refresh.sh` / `poll.sh` / `watch_events.sh` kept it current; `render_*.sh` + `side_by_side.py` made the teacher / BC / oracle / R2 triptych videos | `research/tracks/demo.md`, D-078 (mislabel fix) |
| `export_ui_types.py` | generated the workbench's `ui/src/transport/types.ts` from the service schemas | retired with the workbench (D-145; see `.old/ui/README.md`) |

Kept live instead: peer transport, asset fetch and external-env setup moved to `ops/bin/`; the episode renderers and
the replay-spec generator became `rrp video {arm,dual,legged}` and `rrp viz specs`. Scripts named by a paused track's
RESUME section stay in `scripts/` until that track's P4 unit has replaced them with a recipe.
