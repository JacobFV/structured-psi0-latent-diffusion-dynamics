# .old/src (D-145 purge unit P6; D-146 unit A3)

| path | what it was | decisions | moved in |
|---|---|---|---|
| `.old/src/rrp/viz/workbench/` | the loopback workbench service: FastAPI app (`app.py`), interactive `Session` wrapper (`sessions.py`), probe endpoints, request schemas, policy registry. Served the React UI in `.old/ui/`; launched by `rrp workbench --port 8765` (command deleted). | P10/P11 (`STATUS.md`), D-131 (replaced by the viz room) | D-145 P6 |
| `.old/src/rrp/policies/teachers/dual_coord.py` | W12 coordination-teacher STUBS for `pivot_against_surface` and `carry_tray_level` (privileged, unvalidated, no data ever produced from them; `make_dual_teacher` dispatched to them). The scenes, task graphs and the two-task v2/v3 teachers stay live. | D-126 #22, D-146 item 5 | D-146 unit A3 |

The `service` optional dependency extra (fastapi, uvicorn, websockets, httpx) was deleted with the workbench. The robot registry
`rrp.bodies.catalog.workbench_robots` is NOT part of this service and stays.
