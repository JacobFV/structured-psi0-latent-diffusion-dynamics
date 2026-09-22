import importlib.util
from pathlib import Path

from rrp.service.schemas import public_message_kinds

REPO = Path(__file__).resolve().parents[2]


def test_ui_protocol_exposes_state_and_intervention_receipts():
    kinds = set(public_message_kinds())
    assert {"session_snapshot", "graph_committed", "command_rejected",
            "probe_result", "resource_update"} <= kinds


def test_committed_ui_types_match_backend_schemas(tmp_path, monkeypatch):
    """ui/src/transport/types.ts must be the exact output of scripts/export_ui_types.py."""
    committed = (REPO / "ui" / "src" / "transport" / "types.ts").read_text()
    spec = importlib.util.spec_from_file_location("export_ui_types", REPO / "scripts" / "export_ui_types.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "REPO", tmp_path)          # write to a scratch tree, compare text
    generated = mod.main()
    assert generated == committed, "regenerate with: .venv/bin/python scripts/export_ui_types.py"
