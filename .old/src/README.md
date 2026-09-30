# .old/src (D-145, purge unit P6)

| path | what it was | decisions | moved in |
|---|---|---|---|
| `.old/src/rrp/viz/workbench/` | the loopback workbench service: FastAPI app (`app.py`), interactive `Session` wrapper (`sessions.py`), probe endpoints, request schemas, policy registry. Served the React UI in `.old/ui/`; launched by `rrp workbench --port 8765` (command deleted). | P10/P11 (`STATUS.md`), D-131 (replaced by the viz room) | D-145 P6 |

The `service` optional dependency extra (fastapi, uvicorn, websockets, httpx) was deleted with it. The robot registry
`rrp.bodies.catalog.workbench_robots` is NOT part of this service and stays.
