# .old/scripts — finished drivers (D-145 purge unit P3)

Old path `scripts/...`. Moved in by purge unit P3 (`git log --follow -- .old/scripts/<file>`).

| file | what it was | cited by |
|---|---|---|
| `demo/*` (13 files) | sprint demo page: `build_page.py` built `docs/demo/` from raw results; `refresh.sh` / `poll.sh` / `watch_events.sh` kept it current; `render_*.sh` + `side_by_side.py` made the teacher / BC / oracle / R2 triptych videos | `research/tracks/demo.md`, D-078 (mislabel fix) |
| `export_ui_types.py` | generated the workbench's `ui/src/transport/types.ts` from the service schemas | retired with the workbench (D-145; see `.old/ui/README.md`) |
| `armdiv_chain.sh` | armdiv G1-G3 coordinator: sequential run-dag steps (`pack bcsmoke lsmoke bc1701 lin_sf1 lin_rest bc1702 bckf lin_kf`), one peer GPU lease at a time | `research/tracks/armdiv.md`, D-137 |
| `armdiv_pack.sh`, `armdiv_pack_config.json` | leased `rrp data pack` of the v7div collection to peer disk (`artifacts/packed/latent_pp_v7div_s1_H16`, 1,914,009 rows, done; config = the 65 pool keys of `research/splits/armdiv_pool_v1.json`, horizon 16, stride 1) | `research/tracks/armdiv.md`, D-137 |

Kept live instead: peer transport, asset fetch and external-env setup moved to `ops/bin/`; the episode renderers and
the replay-spec generator became `rrp video {arm,dual,legged}` and `rrp viz specs`. The armdiv chain is now the
`rrp run-dag recipes/armdiv/*.yaml` lines of `research/tracks/armdiv.md` RESUME (P4b).
