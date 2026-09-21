"""Extended CLI commands (need the project venv: numpy/mujoco/fastapi/torch)."""
from __future__ import annotations

import argparse
import json
import sys


def cmd_workbench(a):
    from rrp.service.app import serve
    serve(host=a.host, port=a.port)


def cmd_task_validate(a):
    from rrp.tasks.compiler import compile_task
    c = compile_task(json.loads(open(a.path).read()))
    print(json.dumps({"task_id": c.definition.task_id, "events": c.topo_order,
                      "incidences": len(c.incidences), "edges": len(c.edges)}, indent=1))


def cmd_assets_validate(a):
    from rrp.morphology.catalog import workbench_robots
    robots = workbench_robots()
    if a.robot not in robots:
        raise SystemExit(f"unknown robot {a.robot}; known: {sorted(robots)}")
    asm = robots[a.robot]()
    rs = asm.robot_spec
    print(json.dumps({"robot": a.robot, "spec_hash": rs.spec_hash, "validation": asm.validation,
                      "independent_controls": rs.independent_controls(),
                      "generalized_coordinates": rs.generalized_coordinates(),
                      "assemblies": [x.id for x in rs.assemblies], "lineage": rs.lineage}, indent=1))


def register(sub):
    p = sub.add_parser("workbench", help="serve the loopback workbench (backend + built UI)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=cmd_workbench)
    t = sub.add_parser("task", help="task graph tools").add_subparsers(dest="task_cmd", required=True)
    v = t.add_parser("validate")
    v.add_argument("path")
    v.set_defaults(fn=cmd_task_validate)
    s = sub.add_parser("assets", help="asset catalogue tools").add_subparsers(dest="assets_cmd", required=True)
    v = s.add_parser("validate")
    v.add_argument("--robot", required=True)
    v.set_defaults(fn=cmd_assets_validate)
    try:
        from rrp import cli_ml
        cli_ml.register(sub)
    except ImportError:
        pass
