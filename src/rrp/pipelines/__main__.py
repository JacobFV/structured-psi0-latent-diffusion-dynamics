"""Entry point of a leased stage job: python -m rrp.pipelines run (--config FILE | --config-b64 B64) [--no-check-inputs]."""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rrp.pipelines")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one stage of a RunConfig and write its manifest")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--config")
    g.add_argument("--config-b64")
    r.add_argument("--root", default=".")
    r.add_argument("--no-check-inputs", action="store_true")
    s = sub.add_parser("stages", help="list the registered stages per family")
    a = ap.parse_args(argv)
    from rrp.pipelines.base import Pipeline
    if a.cmd == "stages":
        from rrp.contracts.runconfig import families, load_family_plugins
        load_family_plugins()
        for fam in ("arm", "legged", "dual") + families()[3:]:
            print(fam, " ".join(Pipeline(fam).stages()))
        return 0
    from rrp.contracts.runconfig import RunConfig
    d = json.loads(base64.b64decode(a.config_b64)) if a.config_b64 else json.loads(Path(a.config).read_text())
    rc = RunConfig.model_validate(d)
    from rrp.pipelines.base import GATE_EXIT, GateFailed
    try:
        body = Pipeline(rc.family).run(rc, root=Path(a.root), check_inputs=not a.no_check_inputs)
    except GateFailed as e:
        print(f"[pipeline] {e}", flush=True)
        return GATE_EXIT
    print(json.dumps(dict(run_id=body["run_id"], config_hash=body["config_hash"], metrics=body["metrics"]), default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
