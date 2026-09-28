"""Shared helpers of the room exporter (viz/CONTRACT.md, D-131): configuration, the document envelope, atomic writes,
markdown parsing (D-entries, tables, bullets), source-label / body / version normalization and the per-file cache.

Pure stdlib file/JSON work (host-light, D-127): no numpy, no torch, no mujoco.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path.home()


@dataclass
class Config:
    repo: Path                                   # the checkout the exporter runs from (its files are "in git")
    out: Path                                    # viz/data
    wt_root: Path | None = HOME / "work/rrp-wt"  # agent worktrees (local copies of artifacts/runs)
    main_checkout: Path | None = HOME / "work/relational-robot-policy"
    rrp_data: Path | None = HOME / "work/rrp-data"
    psi1z: Path | None = HOME / "work/psi1z"
    peer: str = "gb10-direct"
    peer_root: str = "/dev/shm/rrp-brandonin"
    live: bool = False
    now: float = field(default_factory=time.time)
    peer_timeout_s: float = 5.0
    stale_after_s: float = 60.0

    @classmethod
    def from_env(cls, repo: Path, out: Path, live: bool = False) -> "Config":
        e = os.environ
        opt = lambda k, d: (Path(e[k]) if e.get(k) else d)          # noqa: E731
        return cls(repo=Path(repo).resolve(), out=Path(out).resolve(), live=live,
                   wt_root=opt("RRP_VIZ_WT_ROOT", HOME / "work/rrp-wt"),
                   main_checkout=opt("RRP_VIZ_MAIN", HOME / "work/relational-robot-policy"),
                   rrp_data=opt("RRP_VIZ_DATA", HOME / "work/rrp-data"),
                   psi1z=opt("RRP_VIZ_PSI1Z", HOME / "work/psi1z"),
                   peer=e.get("ROBOT_PEER", "gb10-direct"),
                   peer_root=e.get("RRP_PEER_ROOT", "/dev/shm/rrp-brandonin"))

    @property
    def cache_dir(self) -> Path:
        return self.out / "_cache"


def iso(t: float | None) -> str | None:
    if t is None:
        return None
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).astimezone().isoformat(timespec="seconds")


_SHA: dict[str, str | None] = {}


def git_sha(repo: Path) -> str | None:
    k = str(repo)
    if k not in _SHA:
        try:
            _SHA[k] = subprocess.run(["git", "-C", k, "rev-parse", "HEAD"], capture_output=True, text=True,
                                     timeout=3).stdout.strip() or None
        except Exception:
            _SHA[k] = None
    return _SHA[k]


def envelope(name: str, cfg: Config, sources, **body) -> dict:
    """Every document: {schema: rrp-viz/<name>/v1, generated_at, git_sha, sources: [paths]} + body."""
    srcs = sorted({str(s) for s in sources if s})
    return {"schema": f"rrp-viz/{name}/v1", "generated_at": iso(cfg.now), "generated_at_unix": round(cfg.now, 3),
            "git_sha": git_sha(cfg.repo), "sources": srcs, **body}


def write_json(path: Path, obj, indent=None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(obj, indent=indent, separators=None if indent else (",", ":"), ensure_ascii=False, default=str)
    tmp.write_text(data)
    os.replace(tmp, path)
    return len(data)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- markdown
D_ID = re.compile(r"\bD-(\d{3})\b")
P_ID = re.compile(r"\bP-(\d{3})\b")
W_ID = re.compile(r"\bW(\d{1,2})\b")
ROADMAP_REF = re.compile(r"(?:roadmap\s*)?#(\d{1,3})\b")
ENTRY_HEAD = re.compile(r"^## ([DP])-(\d{3})\s+(\d{4}-\d{2}-\d{2})?\s*(.*)$")


def d_refs(text: str) -> list[str]:
    return sorted({f"D-{m}" for m in D_ID.findall(text or "")})


def p_refs(text: str) -> list[str]:
    return sorted({f"P-{m}" for m in P_ID.findall(text or "")})


def parse_entries(text: str, source_file: str, prefix: str = "D") -> list[dict]:
    """Parse '## D-NNN YYYY-MM-DD title' entries (rrp) or '## P-NNN …' (psi1z). Body = markdown up to the next '## '."""
    out, cur, start = [], None, 0
    lines = text.splitlines()

    def close(end):
        if cur is not None:
            body = "\n".join(lines[cur["_start"] + 1:end]).strip("\n")
            cur["body"] = body
            cur["line_end"] = end
            refs = d_refs(body + " " + cur["title"])
            cur["refs_d"] = [r for r in refs if r != cur["id"]]
            cur["refs_p"] = [r for r in p_refs(body + " " + cur["title"]) if r != cur["id"]]
            cur["workstreams"] = sorted({f"W{m}" for m in W_ID.findall(body + " " + cur["title"])},
                                        key=lambda w: int(w[1:]))
            cur["paths"] = sorted(set(re.findall(r"`((?:artifacts|research|docs|dags|scripts|src|configs)/[^`\s]+)`",
                                                 body)))[:60]
            del cur["_start"]
            out.append(cur)

    for i, ln in enumerate(lines):
        if ln.startswith("## "):
            close(i)
            cur = None
            m = ENTRY_HEAD.match(ln)
            if m and m.group(1) == prefix:
                cur = {"id": f"{prefix}-{m.group(2)}", "date": m.group(3), "title": m.group(4).strip(),
                       "source_file": source_file, "line": i + 1, "_start": i}
    close(len(lines))
    return out


_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    parts, buf, i = [], "", 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            buf += "|"
            i += 2
            continue
        if s[i] == "|":
            parts.append(buf.strip())
            buf = ""
        else:
            buf += s[i]
        i += 1
    parts.append(buf.strip())
    return parts


def parse_tables(text: str, source_file: str) -> list[dict]:
    """GitHub markdown tables with their nearest heading and line number."""
    lines = text.splitlines()
    out, heading, i = [], None, 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            heading = ln.lstrip("#").strip()
        if ln.lstrip().startswith("|") and i + 1 < len(lines) and _SEP.match(lines[i + 1]):
            header = _cells(ln)
            rows, j = [], i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(_cells(lines[j]))
                j += 1
            out.append({"heading": heading, "header": header, "rows": rows, "source_file": source_file,
                        "line": i + 1})
            i = j
            continue
        i += 1
    return out


def parse_sections(text: str) -> list[dict]:
    """Headings (any level) with their body text and line numbers."""
    lines = text.splitlines()
    heads = [(i, len(ln) - len(ln.lstrip("#")), ln.lstrip("#").strip()) for i, ln in enumerate(lines)
             if ln.startswith("#") and ln.lstrip("#").startswith(" ")]
    out = []
    for k, (i, lvl, title) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        out.append({"title": title, "level": lvl, "line": i + 1, "body": "\n".join(lines[i + 1:end]).strip("\n")})
    return out


def bullets(body: str) -> list[str]:
    """Top-level '- ' bullets, continuation lines joined."""
    out = []
    for ln in body.splitlines():
        if ln.startswith("- "):
            out.append(ln[2:].strip())
        elif out and ln.startswith("  ") and ln.strip():
            out[-1] += " " + ln.strip()
    return out


# ---------------------------------------------------------------- normalization
ARM_BODY = re.compile(r"\b((?:panda|parm\d+[ls]?|ur5e|sawyer|xarm7|iiwa\w*|kinova\w*|gen3\w*|fr3)_(?:pg2|tf3))\b")
ARM_SHORT = re.compile(r"(?:^|[_/\-])(panda|parm\d+[ls]?|ur5e|sawyer|xarm7)(?=$|[_/\-.])")
LEGGED_BODIES = ("anymal_c", "hexapod6_long", "hexapod6", "go2", "t1", "g1", "h1", "pquad4", "sprawl4", "sprawl8",
                 "spot", "a1", "cassie")
_LEG_RE = re.compile(r"(?<![a-z0-9])(" + "|".join(LEGGED_BODIES) + r")(?![a-z0-9])")
_TOK_SPLIT = re.compile(r"[/_\-.=]")


def bodies_in(s: str) -> list[str]:
    s = s or ""
    arms = ARM_BODY.findall(s)
    legs = [] if arms else [m for m in _LEG_RE.findall(s.replace("-", "_"))]
    seen, out = set(), []
    for b in arms + legs:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def family_of(body: str | None, path: str = "") -> str | None:
    p = path.lower()
    if "psi1z" in p or "psi0" in p:
        return "psi0"
    if "dual" in p or "handover" in p or "support_insert" in p or "w12" in p:
        return "dual"
    if body is None:
        return "arm" if ARM_SHORT.search(p) else None
    if ARM_BODY.fullmatch(body):
        return "arm"
    if body in LEGGED_BODIES:
        return "legged"
    return None


def source_label(raw) -> str | None:
    """Normalize a recorded source/route description to the contract's labels. Unknown stays as recorded; None stays None."""
    if raw is None:
        return None
    if isinstance(raw, list):
        labs = sorted({source_label(r) for r in raw if r is not None} - {None})
        return labs[0] if len(labs) == 1 else ("mixed:" + "|".join(labs) if labs else None)
    if isinstance(raw, dict):
        for k in ("source_label", "label", "source", "kind", "name"):
            if k in raw:
                return source_label(raw[k])
        return None
    s = str(raw).strip()
    lo = s.lower()
    if lo in ("teacher", "scripted_teacher", "scripted") or lo.startswith("scripted_teacher"):
        return "scripted_teacher"
    if lo.startswith("oracle") or "oracle diagnostic" in lo or lo.startswith("target_encoder_oracle"):
        return "oracle" if ":" not in s else "oracle:" + s.split(":", 1)[1].strip()
    if lo.startswith(("learned:", "bc:", "learned_tracker:", "learned_latent")):
        return s
    if lo == "bc":
        return "bc"
    if lo in ("generated", "r2"):
        return "learned:generated"
    if lo == "learned_tracker":
        return "learned_tracker"
    if lo in ("random", "mock", "privileged"):
        return lo
    return s


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def rnd(x, nd=4):
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, nd)
    return x


# ---------------------------------------------------------------- per-source cache (by path, mtime, size)
class FileCache:
    """path -> (mtime_ns, size, sha1). Content-derived products are cached by sha1 in `products`."""

    VERSION = 4

    def __init__(self, path: Path):
        self.path = path
        d = read_json(path, {}) or {}
        ok = d.get("version") == self.VERSION
        self.stat: dict[str, list] = d.get("stat", {}) if ok else {}
        self.products: dict[str, dict] = d.get("products", {}) if ok else {}
        self.dirty = False
        self.used: set[str] = set()

    def digest(self, p: Path, st: os.stat_result | None = None) -> str | None:
        st = st or p.stat()
        k = str(p)
        e = self.stat.get(k)
        if e and e[0] == st.st_mtime_ns and e[1] == st.st_size:
            return e[2]
        try:
            h = sha1_bytes(p.read_bytes())
        except OSError:
            return None
        self.stat[k] = [st.st_mtime_ns, st.st_size, h]
        self.dirty = True
        return h

    def product(self, kind: str, h: str):
        self.used.add(f"{kind}:{h}")
        return self.products.get(f"{kind}:{h}")

    def put(self, kind: str, h: str, value):
        self.products[f"{kind}:{h}"] = value
        self.used.add(f"{kind}:{h}")
        self.dirty = True

    def save(self, prune_kinds: tuple[str, ...] = ()):
        if prune_kinds:
            for k in [k for k in self.products if k.split(":", 1)[0] in prune_kinds and k not in self.used]:
                del self.products[k]
                self.dirty = True
        if self.dirty:
            write_json(self.path, {"version": self.VERSION, "stat": self.stat, "products": self.products})
            self.dirty = False
