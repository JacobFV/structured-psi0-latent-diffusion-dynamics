"""YAML subset for DAG files (PyYAML is not a project dependency; it is used when installed).

Supported: block mappings and sequences by indentation (spaces only), "- " items (including "- key: value" maps),
flow sequences [a, b] and flow mappings {a: 1, b: [2, 3]} (nesting allowed), scalars (int, float, true/false,
null/~, 'single' and "double" quoted strings), comments (# outside quotes), blank lines. Not supported (rejected):
anchors/aliases, tags, multi-line strings (| >), multiple documents.
"""
from __future__ import annotations

import json
import re


class YamlError(ValueError):
    pass


def load(text: str):
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except ImportError:
        return loads(text)


def _strip_comment(line: str) -> str:
    out, q = [], None
    for i, ch in enumerate(line):
        if q:
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


_INT = re.compile(r"[-+]?\d+")
_FLOAT = re.compile(r"[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?")


def _scalar(s: str):
    s = s.strip()
    if s == "" or s in ("null", "~", "Null", "NULL"):
        return None
    if s in ("true", "True", "TRUE"):
        return True
    if s in ("false", "False", "FALSE"):
        return False
    if s[0] in "'\"":
        if len(s) < 2 or s[-1] != s[0]:
            raise YamlError(f"unterminated string {s!r}")
        return json.loads(s) if s[0] == '"' else s[1:-1].replace("''", "'")
    if s[0] in "[{":
        v, rest = _flow(s, 0)
        if s[rest:].strip():
            raise YamlError(f"trailing text after flow value: {s!r}")
        return v
    if s[0] in "&*!|>":
        raise YamlError(f"unsupported YAML feature in {s!r}")
    if _INT.fullmatch(s):
        return int(s)
    if _FLOAT.fullmatch(s):
        return float(s)
    return s


def _flow(s: str, i: int):
    """Parse a flow collection starting at s[i] ('[' or '{'); returns (value, index after it)."""
    open_ch = s[i]
    close = "]" if open_ch == "[" else "}"
    i += 1
    items: list = []
    d: dict = {}
    while True:
        while i < len(s) and s[i] in " \t":
            i += 1
        if i >= len(s):
            raise YamlError(f"unterminated flow collection: {s!r}")
        if s[i] == close:
            return (items if open_ch == "[" else d), i + 1
        key = None
        if open_ch == "{":
            j = _scan_token(s, i, ":")
            key = _scalar(s[i:j])
            i = j + 1
            while i < len(s) and s[i] in " \t":
                i += 1
        if s[i] in "[{":
            val, i = _flow(s, i)
        else:
            j = _scan_token(s, i, "," + close)
            val = _scalar(s[i:j])
            i = j
        if open_ch == "[":
            items.append(val)
        else:
            d[key] = val
        while i < len(s) and s[i] in " \t":
            i += 1
        if i < len(s) and s[i] == ",":
            i += 1


def _scan_token(s: str, i: int, stops: str) -> int:
    q = None
    while i < len(s):
        ch = s[i]
        if q:
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch in stops:
            return i
        i += 1
    raise YamlError(f"expected one of {stops!r} in {s!r}")


def _split_key(text: str):
    """'key: value' -> (key, value-or-''); None if the line is not a mapping entry."""
    q = None
    for i, ch in enumerate(text):
        if q:
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch in "[{":
            return None
        elif ch == ":" and (i + 1 == len(text) or text[i + 1] == " "):
            return _scalar(text[:i]), text[i + 1:].strip()
    return None


def loads(text: str):
    lines = []
    for n, raw in enumerate(text.splitlines(), 1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise YamlError(f"line {n}: tabs in indentation")
        s = _strip_comment(raw)
        if s.strip() in ("---", "..."):
            continue
        if s.strip():
            lines.append((len(s) - len(s.lstrip(" ")), s.strip(), n))
    if not lines:
        return None
    val, i = _block(lines, 0, lines[0][0])
    if i != len(lines):
        raise YamlError(f"line {lines[i][2]}: unexpected indentation")
    return val


def _block(lines, i, ind):
    if lines[i][1].startswith("- ") or lines[i][1] == "-":
        return _seq(lines, i, ind)
    return _map(lines, i, ind)


def _map(lines, i, ind):
    d = {}
    while i < len(lines) and lines[i][0] == ind:
        _, text, n = lines[i]
        kv = _split_key(text)
        if kv is None or text.startswith("- "):
            raise YamlError(f"line {n}: expected 'key: value'")
        k, v = kv
        if k in d:
            raise YamlError(f"line {n}: duplicate key {k!r}")
        i += 1
        if v == "":
            if i < len(lines) and lines[i][0] > ind:
                d[k], i = _block(lines, i, lines[i][0])
            elif i < len(lines) and lines[i][0] == ind and lines[i][1].startswith("- "):
                d[k], i = _seq(lines, i, ind)          # "key:\n- a" (sequence at the key's indentation)
            else:
                d[k] = None
        else:
            d[k] = _scalar(v)
    if i < len(lines) and lines[i][0] > ind:
        raise YamlError(f"line {lines[i][2]}: unexpected indentation")
    return d, i


def _seq(lines, i, ind):
    out = []
    while i < len(lines) and lines[i][0] == ind and (lines[i][1].startswith("- ") or lines[i][1] == "-"):
        _, text, n = lines[i]
        rest = text[1:].strip()
        i += 1
        if rest == "":
            if i < len(lines) and lines[i][0] > ind:
                v, i = _block(lines, i, lines[i][0])
            else:
                v = None
            out.append(v)
            continue
        kv = _split_key(rest)
        if kv is None:
            out.append(_scalar(rest))
            continue
        # "- key: value" starts a mapping whose further keys are indented to the key's column
        sub_ind = ind + (len(text) - len(rest))
        synthetic = [(sub_ind, rest, n)] + []
        j = i
        while j < len(lines) and lines[j][0] >= sub_ind:
            j += 1
        v, k = _map(synthetic + lines[i:j], 0, sub_ind)
        if k != 1 + (j - i):
            raise YamlError(f"line {n}: bad mapping item")
        out.append(v)
        i = j
    return out, i
