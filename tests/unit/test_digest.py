"""One digest helper (D-146 round 3, hygiene): `rrp.core.provenance.digest / json_digest / file_digest` are the only
places that hash; nothing else in `src/rrp` imports hashlib. The helper keeps every identity that was hashed inline
before (canonical-JSON options are part of the hashed bytes)."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from rrp.core.provenance import content_hash, digest, file_digest, json_digest

SRC = Path(__file__).resolve().parents[2] / "src" / "rrp"
HOME = "core/provenance.py"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_no_hashlib_outside_the_digest_helper():
    bad = []
    for p in sorted(SRC.rglob("*.py")):
        rel = p.relative_to(SRC).as_posix()
        if rel == HOME:
            continue
        for n in ast.walk(ast.parse(p.read_text())):
            if (isinstance(n, ast.Import) and any(a.name.split(".")[0] == "hashlib" for a in n.names)) or \
                    (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "hashlib"):
                bad.append(f"src/rrp/{rel}:{n.lineno}")
    assert not bad, "hash through rrp.core.provenance.digest / json_digest / file_digest:\n" + "\n".join(bad)


def test_digest_is_sha256_over_the_concatenated_parts():
    assert digest(b"abc", length=None) == _sha(b"abc")
    assert digest(b"a", b"bc") == _sha(b"abc")[:16] == digest(b"abc")
    assert digest(b"abc", length=8) == _sha(b"abc")[:8] and digest() == _sha(b"")[:16]


def test_json_digest_keeps_the_legacy_inline_identities(tmp_path):
    d = {"b": [1, 2.5], "a": {"z": None, "y": "x"}}
    assert json_digest(d) == _sha(json.dumps(d, sort_keys=True).encode())[:16]
    assert json_digest(d, 12) == _sha(json.dumps(d, sort_keys=True).encode())[:12]
    assert json_digest(d, None, indent=1, default=str) == _sha(json.dumps(d, indent=1, sort_keys=True, default=str).encode())
    assert json_digest({"p": Path("x")}, default=str) == _sha(b'{"p": "x"}')[:16]
    assert content_hash(d) == _sha(json.dumps(d, sort_keys=True, separators=(",", ":")).encode())[:16]
    f = tmp_path / "f.bin"
    f.write_bytes(b"xyz" * 1000)
    assert file_digest(f) == digest(b"xyz" * 1000) and file_digest(f, None) == _sha(b"xyz" * 1000)


def test_call_sites_keep_their_identities():
    from rrp.core.robot import combined_hash
    from rrp.harness.eval.registry import config_hash
    from rrp.policies.features.legged import spec_hash16
    from rrp.policies.nets.backbone import image_hash, text_hash
    assert combined_hash(["a", "b"]) == "multi:" + _sha(b"a|b")[:16]
    assert config_hash({"k": object}) == _sha(json.dumps({"k": object}, sort_keys=True, default=str).encode())[:16]
    assert spec_hash16("s") == _sha(b"s")[:16] and text_hash("s") == _sha(b"s")[:16]
    a = np.arange(6, dtype=np.uint8).reshape(2, 3)
    assert image_hash(a) == _sha(b"(2, 3)" + a.tobytes())[:16]
