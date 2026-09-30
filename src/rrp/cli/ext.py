"""Extended CLI commands (need the project venv: numpy/mujoco/torch)."""
from __future__ import annotations

import argparse
import json
import sys


def cmd_task_validate(a):
    from rrp.tasks.compiler import compile_task
    c = compile_task(json.loads(open(a.path).read()))
    print(json.dumps({"task_id": c.definition.task_id, "events": c.topo_order,
                      "incidences": len(c.incidences), "edges": len(c.edges)}, indent=1))


def cmd_assets_validate(a):
    from rrp.bodies.catalog import workbench_robots
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
    t = sub.add_parser("task", help="task graph tools").add_subparsers(dest="task_cmd", required=True)
    v = t.add_parser("validate")
    v.add_argument("path")
    v.set_defaults(fn=cmd_task_validate)
    s = sub.add_parser("assets", help="asset catalogue tools").add_subparsers(dest="assets_cmd", required=True)
    v = s.add_parser("validate")
    v.add_argument("--robot", required=True)
    v.set_defaults(fn=cmd_assets_validate)
