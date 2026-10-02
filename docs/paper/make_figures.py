#!/usr/bin/env python3
"""Generate the paper's figures and generated tables.  Run from the repository root:

    PYTHONPATH=src:<dir with matplotlib, Pillow>:$HOME/work/ext/cw-site python3 docs/paper/make_figures.py \
        {fig2|fig3|table_relations|table_envs|all}

Outputs (docs/paper/): figures/fig2.{pdf}, figures/fig2_preview.png, figures/fig2_panel_*.png, figures/fig2_panels.json,
figures/fig3_envs.{pdf,png}, figures/table_relations_counts.json, table_relations.tex (macros \\relsummary, \\relfull,
\\relplanned), table_envs.tex.  The tables need only the repository environment; Fig. 2 and Fig. 3 additionally need
matplotlib / Pillow, which the repository environment does not ship, so put them on PYTHONPATH from a throwaway
`pip install --target`.  Fig. 3 needs MuJoCo with EGL; Fig. 2 also reads the ComputerWorld wheel (cw-site, see
research/tracks/pointer.md).  Run under the ops broker: `python -m rrp.cli ops run --cpu 2 --mem 4G --label paper_build -- ...`.

Sections: [U3] relation tables, [U2] environments table and Fig. 3 (source-tagged environment tiles), [U1] Fig. 2.
Fig. 2 shows factor VALUES computed by the real relation code or read from simulator state; it is not trained attention.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent


# =================================================================== [U3] relation tables

REPO = Path(__file__).resolve().parents[2]      # repository root (this file is docs/paper/make_figures.py)
CATALOG_MD = REPO / "research" / "relations_catalog.md"

# ----------------------------------------------------------------------------------------------- display constants
GROUPS = [
    ("structure", "structure / edges"),
    ("geometry", "geometry"),
    ("kinematics", "kinematics / body"),
    ("interaction", "physical interaction"),
    ("task", "task / procedure"),
    ("locomotion", "locomotion"),
    ("ui", "UI"),
    ("time", "time"),
    ("probe", "probe readouts"),
    ("other", "properties / affordance / language"),   # planned-only families
]
SUMMARY_LABEL = {"structure": "structure", "geometry": "geometry", "kinematics": "kinematics/body",
                 "interaction": "interaction", "task": "task/procedure", "locomotion": "locomotion", "ui": "UI",
                 "time": "time", "probe": "probe readouts", "other": "prop./aff./lang."}
TASK_EDGES = {"actor_of", "patient_of", "target_of", "destination_of", "support_of", "enables", "maintained",
              "output_to", "produced", "consumed_by", "pred_arg", "role_in_event", "role_points_to", "node_actor_of"}
KIN_EDGES = {"kin_parent", "kin_child", "mirror", "limb_adjacent", "foot_of", "over_cell"}
PREFIX_GROUP = {"msg": "structure", "route": "structure", "id": "structure", "geo": "geometry", "kin": "kinematics",
                "ix": "interaction", "task": "task", "rel": "task", "leg": "locomotion", "ui": "ui", "time": "time",
                "probe": "probe", "prop": "other", "aff": "other", "lang": "other", "vel": "other"}
NET_ABBR = {"arm": "arm", "dual": "dual", "legged": "leg", "humanoid": "hum", "psi0": "psi0", "pointer": "ptr"}
NET_ORDER = list(NET_ABBR)
ENV_ABBR = {"mujoco/arm": "arm", "mujoco/dual": "dual", "mujoco/legged": "mj-leg", "warp/legged": "wp-leg",
            "simple": "simple", "computerworld": "cw"}
ENV_ORDER = list(ENV_ABBR)
ATTN_FORMS = ("bias", "aug", "gate", "mask")
OPS_RE = ["sqdiff+diff", "rel_rot", "bilinear", "ancestor", "unary", "align", "order", "same", "edge", "flow", "hop",
          "sim", "diff"]
FORMS_RE = ["bias", "aug", "gate", "mask", "readout", "message", "embed"]
ENTRY_RE = re.compile(r"^(geo|ix|id|kin|task|time|ui|edge|msg|route|leg|probe|prop|aff|rel|lang)\.[\w*<>.]+$")

# Catalog rows that carry a wave tag but name no entry: intended entry name (proposed; shown in italics). Keyed by
# (section letter, lower-cased family cell, tag). Rows sharing a name merge into one table row.
PROPOSED_NAMES = {
    ("A", "surface directions", "W2"): "geo.surface_normal",
    ("A", "shape and scale", "P"): "geo.extent",
    ("A", "visibility", "P"): "geo.occludes",
    ("A", "symmetry / lattice", "P"): "id.same_part_class",
    ("B", "segmentation", "P"): "id.same_segment",
    ("B", "hierarchy", "W2"): "id.scene_parent",
    ("B", "equivalence", "P"): "id.same_part_class",
    ("C", "mirror / morphology correspondence", "P"): "kin.cross_morphology",
    ("D", "force transfer", "W2"): "ix.impulse_flow",
    ("D", "collisions over time", "W2"): "ix.collision_seq",
    ("E", "affordance pairs", "W2"): "aff.affords_at",
    ("E", "compatibility", "P"): "aff.compatible",
    ("E", "safety regions", "P"): "aff.safe_region",
    ("G", "motion", "P"): "time.co_motion",
    ("I", "semantic similarity", "P"): "lang.sim",
}
# a wave cell whose tags cannot be split generically
LABEL_OVERRIDES = {"geo.surface_normal": ["surface_normal"]}     # the catalog row lists two labels for two entries
SEGMENT_OVERRIDES = {
    ("D", "fields and flows"): [("X", "stress / strain / pressure / momentum / energy fields (rigid-body sims)"),
                                ("P", "`prop.vel` unary velocity field")],
}

# ----------------------------------------------------------------------------------------------- LaTeX helpers
_UNI = {"Ψ₀": r"$\Psi_0$", "→": r"$\to$", "↔": r"$\leftrightarrow$", "×": r"$\times$", "≥": r"$\geq$",
        "−": "--", "–": "--", "—": "---", "≈": r"$\approx$", "²": r"$^2$", "₀": r"$_0$", "Ψ": r"$\Psi$", "·": r"$\cdot$"}


def esc(s: str) -> str:
    """Escape plain text for LaTeX."""
    s = str(s).replace("Ψ₀", "\x00")
    out = []
    for ch in s:
        if ch == "\x00":
            out.append(_UNI["Ψ₀"])
        elif ch in _UNI:
            out.append(_UNI[ch])
        elif ch in "&%#$_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch == "<":
            out.append(r"\textless{}")
        elif ch == ">":
            out.append(r"\textgreater{}")
        else:
            out.append(ch)
    return "".join(out)


def tt(s: str) -> str:
    """Monospace identifier with break opportunities after `.`, `_` and `,`."""
    body = esc(s).replace("\\_", "\\_\\allowbreak{}").replace(".", ".\\allowbreak{}").replace(", ", ",\\allowbreak\\ ")
    return r"\texttt{" + body + "}"


def md_inline(s: str) -> str:
    parts = re.split(r"(`[^`]*`)", s)
    return "".join(tt(p[1:-1]) if len(p) > 1 and p.startswith("`") and p.endswith("`") else esc(p) for p in parts)


YES, NO = r"\textbf{Y}", r"\textcolor{gray}{N}"


def yn(b: bool) -> str:
    return YES if b else NO


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short=8", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


# ----------------------------------------------------------------------------------------------- registries
def load_registries():
    from rrp.policies.relations import base as B
    B._ensure_catalog()
    import rrp.harness.data.relgen as R
    R.load_families()
    L, P, T = B.data_registries()
    return B, R, L, P, T


def group_of(name: str) -> str:
    pre, _, rest = name.partition(".")
    if pre == "edge":
        if rest in TASK_EDGES:
            return "task"
        if rest in KIN_EDGES:
            return "kinematics"
        return "structure"
    return PREFIX_GROUP.get(pre, "other")


def classify_rows(B, R, L, P, T, notes: list):
    """One dict per FACTORS entry (all statuses), classified from the registries."""
    cov = R.coverage()
    env_caps = {e: frozenset(c) for e, c in cov["envs"].items()}
    rows = []
    for name, d in sorted(B.FACTORS.items()):
        suffix = name.split(".", 1)[1]
        # effective relgen label: declared; else the field when it is a relgen label and the factor has a non-given
        # source (`lab = d.label or d.field` in base._check_family); else a label named after the entry
        label, how = None, None
        if d.label in L:
            label, how = d.label, "declared"
        elif d.label is None and d.field in L and any(s in d.sources for s in ("gt", "probe")):
            label, how = d.field, "field"
        elif d.label is None and suffix in L:
            label, how = suffix, "entry-name"
        gen_parts = [g for g in d.gen if g in P]
        gen_tr = [g for g in d.gen if g in T]
        act_parts = [] if gen_parts else sorted(p for p, sp in P.items() if suffix in sp.activates)
        rel_parts = gen_parts + act_parts
        data_gen = bool(d.gen) or label is not None
        # envs: label caps (label_runs_in) intersected with the envs of the relevant scene parts (a transform, or no
        # part at all, leaves every env)
        envs = []
        if data_gen:
            for env, caps in env_caps.items():
                ok_label = True if label is None else R.label_runs_in(label, caps)
                ok_part = (any((not P[p].envs) or env in P[p].envs for p in rel_parts)
                           if rel_parts and not gen_tr else True)
                if ok_label and ok_part:
                    envs.append(env)
        if d.status == "implemented" and data_gen and label is not None and not envs:
            all_caps = frozenset().union(*env_caps.values())
            notes.append(f"{name}: label {label!r} needs caps {sorted(L[label].needs - all_caps)} that no registered env's "
                         f"StateView provides -> data envs: none (the generator exists, no env can run it yet)")
        fam = cov["factors"].get(name, {}).get("families", {}) if d.status == "implemented" else {}
        nets = [f for f in NET_ORDER if fam.get(f) is True]
        row = dict(name=name, group=group_of(name), op=d.op, form=d.form, field=d.field, sources=list(d.sources),
                   label=label, label_how=how, declared_label=d.label, gen=list(d.gen), parts=rel_parts,
                   transforms=gen_tr, data_gen=data_gen, attn=d.form in ATTN_FORMS,
                   readout=bool(d.readout) or d.form == "readout", status=d.status, nets=nets, envs=envs,
                   gates=list(d.gates))
        rows.append(row)
        if d.status != "implemented":
            continue
        if how in ("field", "entry-name"):
            notes.append(f"{name}: data gen Y only via the {how} label {label!r} (no declared `label`)"
                         + (f" and scene part(s) {rel_parts}" if rel_parts else ""))
        if d.form in ("message", "embed"):
            notes.append(f"{name}: form {d.form!r} is neither an attention-logit term nor a readout (net-internal "
                         f"structure): attention N, readout N")
        if d.form == "mask":
            notes.append(f"{name}: form 'mask' counted as attention (hard routing mask, not an additive bias)")
        if d.form == "readout" and label is not None:
            notes.append(f"{name}: form 'readout' with a relgen label ({label}) -> data gen Y, readout Y, attention N")
        if d.label is not None and d.label not in L and not d.gen:
            row["legacy_label"] = True
    legacy = [r["name"] for r in rows if r.get("legacy_label")]
    if legacy:
        notes.append(f"{len(legacy)} probe.* factors: readout targets are legacy packet labels of the task-data "
                     f"pipeline, not relgen LABELS -> data gen N (readout Y)")
    return rows


# ----------------------------------------------------------------------------------------------- catalog parsing
TAG_RE = re.compile(r"(?<![\w`.])(W1|W2|P|X|M)(?![\w`.])")


def parse_catalog(path: Path = CATALOG_MD):
    """-> (planned entries by name, x_rows, m_rows, w1 (name, family) pairs, epistemic mechanisms, problems)."""
    lines = path.read_text().splitlines()
    sec, hdr = None, None
    planned: dict[str, dict] = {}
    x_rows, m_rows, w1_names, mech, problems = [], [], [], [], []
    for ln in lines:
        if ln.startswith("## "):
            sec, hdr = ln[3:].strip(), None
            continue
        if not ln.startswith("|") or sec is None:
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if hdr is None:
            hdr = [c.lower() for c in cells]
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        m = re.match(r"^([A-K])\. ", sec)
        if m is None or m.group(1) == "K":
            continue
        letter = m.group(1)
        row = dict(zip(hdr, cells))
        wave = row.get("wave") or row.get("status") or ""
        cands = row.get("candidates", "")
        if "family" in row:
            family = row["family"]
        elif letter == "J":
            family = cands.split(",")[0].strip()
        else:                                                    # H / I tables have no family cell
            family = re.sub(r"^[A-K]\.\s*", "", sec).split("(")[0].strip()
            if letter == "I":
                family = "text grounding" if "bilinear" in row.get("decomposition", "") else "semantic similarity"
        segs = SEGMENT_OVERRIDES.get((letter, family.lower()))
        if segs is None:
            segs = []
            for seg in wave.split(";"):
                tags = list(TAG_RE.finditer(seg))
                if not tags:
                    if segs:                                      # continuation of the previous segment
                        segs[-1] = (segs[-1][0], segs[-1][1] + "; " + seg.strip())
                    continue
                t = tags[0]
                segs.append((t.group(1), (seg[:t.start()] + " " + seg[t.end():]).strip()))
        for tag, text in segs:
            names = [n for n in re.findall(r"`([^`]+)`", text) if ENTRY_RE.match(n)]
            if tag == "W1":
                w1_names.extend((n, family) for n in names)
                if letter == "J":
                    mech.append(family)
                continue
            if tag == "M":
                m_rows.append((letter, family))
                continue
            if tag == "X":
                x_rows.append((letter, family))
                continue
            proposed = not names
            if proposed:
                nm = PROPOSED_NAMES.get((letter, family.lower(), tag))
                if nm is None:
                    problems.append(f"catalog row {letter}/{family}/{tag}: no entry name and no PROPOSED_NAMES mapping")
                    nm = f"?{family}"
                names = [nm]
            decomp = row.get("decomposition", "")
            flat = (decomp + " " + text).replace("sqdiff+diff", "SQ")
            ops = [o for o in OPS_RE if re.search(rf"(?<![\w+]){'SQ' if o == 'sqdiff+diff' else re.escape(o)}(?!\w)", flat)]
            forms = [f for f in FORMS_RE if re.search(rf"\b{f}\b", decomp)]
            label_cell = next((v for k, v in row.items() if k.startswith("label")), "")
            labels = re.findall(r"`([^`]+)`", label_cell)
            need = [p.split(";")[0].strip() for p in re.findall(r"\(([^)]*)\)", text) if p.strip()]
            need = [n for n in need if not any(n != o and n.replace("needs ", "").replace("a ", "") in o.replace("needs ", "").replace("a ", "") for o in need)]
            envs = [s.strip() for s in re.sub(r"\([^)]*\)", "", row.get("envs", "")).split(",")
                    if s.strip() and s.strip() not in ("–", "-")]
            for nm in names:
                labels_nm = LABEL_OVERRIDES.get(nm, labels)
                e = planned.setdefault(nm, dict(name=nm, wave=tag, families=[], ops=[], forms=[], labels=[], need=[],
                                                envs=[], proposed=proposed))
                if tag == "W2":
                    e["wave"] = "W2"
                for k, v in (("families", [family]), ("ops", ops), ("forms", forms), ("labels", labels_nm),
                             ("need", need), ("envs", envs)):
                    for x in v:
                        if x not in e[k]:
                            e[k].append(x)
    return planned, x_rows, m_rows, w1_names, mech, problems


# ----------------------------------------------------------------------------------------------- table emission
def fmt_list(items, mapping=None, order=None):
    if not items:
        return "--"
    if order:
        items = sorted(items, key=order.index)
    return ",\\allowbreak ".join(esc(mapping.get(i, i) if mapping else i) for i in items)


def fmt_source(srcs):
    return ", ".join(r"gt$^\dagger$" if s == "gt" else esc(s) for s in srcs)


def collapse(rows):
    """Merge edge.* and probe.<net>.* rows with an identical classification into one row each."""
    keyed = {}
    for r in rows:
        parts = r["name"].split(".")
        pre = "edge." if parts[0] == "edge" else (f"probe.{parts[1]}." if parts[0] == "probe" and len(parts) == 3 else None)
        sig = (pre, r["op"], r["form"], tuple(r["sources"]), r["data_gen"], tuple(r["parts"]), tuple(r["transforms"]),
               r["attn"], r["readout"], tuple(r["nets"]), tuple(r["envs"]))
        if pre is None:
            sig += (r["name"],)
        keyed.setdefault(sig, dict(r, members=[], prefix=pre))["members"].append(r["name"])
    return list(keyed.values())


def name_cell(r):
    m = sorted(r["members"])
    if r["prefix"] is None or len(m) == 1:
        return tt(m[0])
    short = [x[len(r["prefix"]):] for x in m]
    return tt(r["prefix"] + "{" + ", ".join(short) + "}") + f" ({len(m)})"


def impl_table(rows, sha, totals):
    col = (r"@{}>{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{0.54in} p{0.37in} "
           r">{\raggedright\arraybackslash}p{0.82in} >{\raggedright\arraybackslash}p{1.12in} c c "
           r">{\raggedright\arraybackslash}p{1.04in} >{\raggedright\arraybackslash}p{0.7in}@{}")
    out = [r"\begin{table*}[t]", r"\centering\fontsize{6.8}{7.7}\selectfont", r"\setlength{\tabcolsep}{2pt}",
           r"\begin{tabularx}{\textwidth}{" + col + "}", r"\toprule",
           r"Factor & Operator & Form & Source(s) & Data gen (parts / transforms) & Attn & Read & Nets & Data envs \\",
           r"\midrule"]
    first = True
    for key, disp in GROUPS:
        grp = [r for r in rows if r["group"] == key and r["status"] == "implemented"]
        if not grp:
            continue
        if not first:
            out.append(r"\midrule")
        first = False
        out.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\textsc{{{esc(disp)}}} \;({len(grp)} implemented)}}\\")
        for r in collapse(grp):
            if r["data_gen"]:
                items = [esc(p) for p in r["parts"]] + [r"\textit{" + esc(t) + "}" for t in r["transforms"]]
                gen = YES + (": " + ", ".join(items) if items else (": " + esc(r["label"]) if r["label"] else ""))
            else:
                gen = NO
            out.append(" & ".join([name_cell(r), esc(r["op"]), esc(r["form"]), fmt_source(r["sources"]), gen,
                                   yn(r["attn"]), yn(r["readout"]), fmt_list(r["nets"], NET_ABBR, NET_ORDER),
                                   (fmt_list(r["envs"], ENV_ABBR, ENV_ORDER) if r["envs"] else "none$^\\ddagger$") if r["data_gen"] else "--"]) + r"\\")
    out += [r"\bottomrule", r"\end{tabularx}", r"\par\smallskip",
            r"{\scriptsize Generated by \texttt{make\_figures.py} from the registries at \texttt{" + sha + r"}.}",
            r"\caption{\textbf{Every implemented relation factor.} "
            + esc(f"{totals['implemented']} factors: {totals['gen']} generate or label data, {totals['attn']} act on "
                  f"attention logits or masks, {totals['readout']} supervise a readout (several per factor possible). ")
            + r"Rows with identical classification are merged (\texttt{edge.*}, \texttt{probe.<net>.*}; count in "
              r"parentheses). gt$^\dagger$ = gt (training only): simulator truth, never a deployable input. Data gen: a "
              r"data generator is declared or a relgen label exists (scene parts upright, transforms italic); probe "
              r"targets are task-pipeline packet labels, hence N. Attn: form $\in$ \{bias, aug, gate, mask\}. Read: a "
              r"readout is declared. Nets: families whose \texttt{resolve(family=...)} accepts the factor. Data envs: "
              r"label caps and scene part available; $^\ddagger$ no registered environment provides the caps yet.}",
            r"\label{tab:relations}", r"\end{table*}"]
    return "\n".join(out) + "\n"


def planned_table(planned, reg_planned, x_rows, sha, totals):
    col = (r"@{}>{\raggedright\arraybackslash}p{1.15in} p{0.88in} >{\raggedright\arraybackslash}p{1.3in} "
           r">{\raggedright\arraybackslash}p{0.8in} >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{0.7in}@{}")
    out = [r"\begin{table*}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{2.4pt}",
           r"\begin{tabularx}{\textwidth}{" + col + "}", r"\toprule",
           r"Intended entry & Status & Catalog family & Operator $\times$ form & Needs (label caps; scene part) & Envs \\",
           r"\midrule"]
    by_group: dict[str, list] = {}
    for e in list(reg_planned) + list(planned.values()):
        by_group.setdefault(group_of(e["name"]), []).append(e)
    first = True
    for key, disp in GROUPS:
        items = sorted(by_group.get(key, []), key=lambda e: e["name"])
        if not items:
            continue
        if not first:
            out.append(r"\midrule")
        first = False
        out.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textsc{{{esc(disp)}}} \;({len(items)} planned)}}\\")
        for e in items:
            nm = (r"\textit{" + tt(e["name"]) + "}") if e["proposed"] else tt(e["name"])
            status = {"W2": "planned (W2)", "P": "planned (P)", "reg": "planned (registry)"}[e["wave"]]
            opform = esc(" + ".join(e["ops"]) or "--") + (r" $\times$ " + esc(", ".join(e["forms"])) if e["forms"] else "")
            need = "; ".join(([", ".join(tt(x) for x in e["labels"])] if e["labels"] else []) + [md_inline(x) for x in e["need"]]) or "--"
            out.append(" & ".join([nm, status, esc("; ".join(e["families"])), opform, need,
                                   esc(", ".join(e["envs"])) if e["envs"] else "--"]) + r"\\")
    xs = "; ".join(sorted({f for _, f in x_rows}))
    out += [r"\midrule",
            rf"\multicolumn{{6}}{{@{{}}p{{\dimexpr\textwidth-4.8pt\relax}}}}{{\textsc{{out of scope}} ({len(x_rows)} families; "
            rf"no physics or sensing for them in our environments): {esc(xs)}.}}\\",
            r"\bottomrule", r"\end{tabularx}", r"\par\smallskip",
            r"{\scriptsize Generated by \texttt{make\_figures.py} from \texttt{research/relations\_catalog.md} and the "
            r"registries at \texttt{" + sha + r"}.}",
            r"\caption{\textbf{Planned and out-of-scope relations.} "
            + esc(f"{totals['planned_entries']} planned entries ({totals['planned_registry']} declared in the registry with "
                  f"status planned, {totals['planned_catalog']} from the candidate catalog, waves W2 and P); "
                  f"{totals['xfam']} out-of-scope families. ")
            + r"Italic names are proposed (the catalog names no entry). None is implemented, so no result in this paper "
              r"depends on them. Meta properties of relations (catalog section K and "
            + esc(f"{totals['meta_rows']} rows") + r" tagged M, e.g.\ joint states, costs) are not factors and are excluded; "
            + esc(f"{totals['epistemic']} epistemic mechanisms") + r" (confidence scaling, soft edges, the reveal / surprise / "
              r"noise / occlude transforms) are implemented but are not attention factors.}",
            r"\label{tab:relations-planned}", r"\end{table*}"]
    return "\n".join(out) + "\n"


def summary_table(rows, planned, reg_planned, sha):
    plan_by = {k: 0 for k, _ in GROUPS}
    for e in list(reg_planned) + list(planned.values()):
        plan_by[group_of(e["name"])] += 1
    lines, tot, cnt = [], [0] * 5, {}
    for k, _ in GROUPS:
        g = [r for r in rows if r["group"] == k and r["status"] == "implemented"]
        vals = [len(g), sum(r["data_gen"] for r in g), sum(r["attn"] for r in g), sum(r["readout"] for r in g), plan_by[k]]
        cnt[k] = dict(zip(["implemented", "data_gen", "attention", "readout", "planned"], vals))
        if any(vals):
            tot = [a + b for a, b in zip(tot, vals)]
            lines.append(" & ".join([esc(SUMMARY_LABEL[k])] + [str(v) for v in vals]) + r"\\")
    out = [r"\begin{table}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
           r"Family & Impl. & Data gen & Attn & Readout & Planned \\", r"\midrule", *lines, r"\midrule",
           " & ".join([r"\textbf{total}"] + [rf"\textbf{{{v}}}" for v in tot]) + r"\\", r"\bottomrule", r"\end{tabular}",
           r"\par\smallskip{\scriptsize Generated by \texttt{make\_figures.py} from the registries at \texttt{" + sha + r"}.}",
           r"\caption{\textbf{Relation factors by family.} Counts are computed from the registry; a factor can count in "
           r"several columns. Planned = declared in the registry or catalogued (waves W2, P), not implemented. The "
           r"full per-factor table is Table~\ref{tab:relations}.}", r"\label{tab:relations-summary}", r"\end{table}"]
    return "\n".join(out) + "\n", cnt, tot


# ----------------------------------------------------------------------------------------------- entry point
def relations_table(out_path) -> dict:
    out_path = Path(out_path)
    (out_path.parent / "figures").mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    B, R, L, P, T = load_registries()
    sha = git_sha()
    rows = classify_rows(B, R, L, P, T, notes)
    planned, x_rows, m_rows, w1_names, mech, problems = parse_catalog()
    reg_planned = [dict(name=r["name"], wave="reg", ops=[r["op"]], forms=[r["form"]], labels=[], need=[], envs=[],
                        proposed=False,
                        families=[next((f for n, f in w1_names if n == r["name"]), "registry") + " (declared)"])
                   for r in rows if r["status"] == "planned"]
    for nm, fam in sorted(set(w1_names)):       # catalog W1 names must exist in the registry
        if not any(fnmatch.fnmatchcase(k, nm) for k in B.FACTORS):
            notes.append(f"catalog lists `{nm}` as W1 (family {fam!r}) but no such factor is registered")
    for nm in sorted(planned):
        if nm in B.FACTORS:
            notes.append(f"catalog plans `{nm}` but it is already registered ({B.FACTORS[nm].status})")
    notes += problems
    impl = [r for r in rows if r["status"] == "implemented"]
    totals = dict(implemented=len(impl), registry_total=len(rows), gen=sum(r["data_gen"] for r in impl),
                  attn=sum(r["attn"] for r in impl), readout=sum(r["readout"] for r in impl),
                  planned_registry=len(reg_planned), planned_catalog=len(planned),
                  planned_entries=len(reg_planned) + len(planned), xfam=len(x_rows), meta_rows=len(m_rows),
                  epistemic=len(mech))
    summ_tex, by_family, tot = summary_table(rows, planned, reg_planned, sha)
    out_path.write_text(
        "% generated by make_figures.py table_relations from the registries; do not edit\n"
        "\\newcommand{\\relsummary}{%\n" + summ_tex + "}\n"
        "\\newcommand{\\relfull}{%\n" + impl_table(rows, sha, totals) + "}\n"
        "\\newcommand{\\relplanned}{%\n" + planned_table(planned, reg_planned, x_rows, sha, totals) + "}\n")
    counts = dict(git_sha=sha, totals=totals, by_family=by_family,
                  registries=dict(factors=len(B.FACTORS), labels=len(L), parts=len(P), transforms=len(T)),
                  forms={f: sum(1 for r in impl if r["form"] == f) for f in sorted({r["form"] for r in impl})},
                  planned=sorted([e["name"] for e in reg_planned] + list(planned)),
                  out_of_scope=sorted(f for _, f in x_rows), meta=sorted(f for _, f in m_rows),
                  epistemic_mechanisms=mech, ambiguous=sorted(set(notes)), rows=rows)
    (out_path.parent / "figures" / "table_relations_counts.json").write_text(json.dumps(counts, indent=1, sort_keys=True, default=list))
    return counts




# =================================================================== [U2] environments table
# the net family each env's relational model uses (docs/relations.md; recipes/*: `family:` of the policy)
ENV_FAMILY = {"arm": "arm", "dual": "dual", "quad": "legged", "humanoid": "humanoid", "warp": "legged",
              "simple": "psi0", "cw": "pointer"}
# factor-name prefix -> a short word for the table
PREFIX_WORD = {"edge": "edges", "route": "routes", "msg": "msg", "id": "ids", "geo": "geometry", "kin": "kinematics",
               "ix": "interaction", "task": "task", "leg": "gait", "ui": "UI", "probe": "probe"}


def _envs_deps():
    global ENVS, FACTORS, FAMILIES, resolve, SIMPLE_TASKS, tasks_in, LEGGED_ASSETS, PROCEDURAL
    from rrp.harness.data.relgen import load_families
    load_families()
    import rrp.policies.relations.catalog  # noqa: F401  (registers FACTORS / FAMILIES)
    from rrp.bodies.legged import LEGGED_ASSETS, PROCEDURAL
    from rrp.envs.base import ENVS
    from rrp.policies.relations.base import FACTORS, FAMILIES, resolve
    from rrp.tasks.spec import SIMPLE_TASKS, tasks_in


def resolving(family: str) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for name, d in sorted(FACTORS.items()):
        if d.status != "implemented":
            continue
        try:
            resolve([name], family=family)
        except Exception:
            continue
        out[name.split(".")[0]] += 1
    return dict(out)


def fam_cell(fam: str) -> str:
    r = resolving(fam)
    tot = sum(r.values())
    structural = sum(v for k, v in r.items() if k != "probe")
    return f"{fam}<nl>{tot} ({structural}+{r.get('probe', 0)})"


def strip_grip(k: str) -> str:
    return re.sub(r"_(pg2|tf3)$", "", k)


def arm_bodies(repo: Path) -> str:
    sp = repo / "research" / "splits"
    pool = json.load(open(sp / "armdiv_pool_v1.json"))
    prim = json.load(open(sp / "primary_v1.json"))
    keys = list(pool["source_train_robots_v6"]) + list(pool["new_procedural"]) + list(pool["new_menagerie"])
    base = sorted({strip_grip(k) for k in keys})
    proc = [b for b in base if b.startswith("pa2s")]
    named = [b for b in base if not b.startswith("pa2s")]
    parm = sorted(b for b in named if b.startswith("parm"))
    other = [b for b in named if not b.startswith("parm")]
    held = sorted({strip_grip(r) for r in prim["targets"]["held_out_arm_family"]["robots"]}
                  | {r for r in prim["targets"]["held_out_compatible_attachment_combination"]["robots"]})
    return (f"{', '.join(other)}, parm5/6/7 variants ({len(parm)}), {len(proc)} procedural pa2s* "
            f"(pg2 / tf3 grippers); sealed targets {', '.join(held)}")


def legged_bodies() -> tuple[str, str]:
    quad = [k for k, v in LEGGED_ASSETS.items() if v["kind"] == "quadruped"]
    quad += [k for k in PROCEDURAL if k.startswith(("hexapod", "pquad", "sprawl"))]
    return ", ".join(quad), ""


def humanoid_bodies(repo: Path) -> str:
    h = json.load(open(repo / "research" / "splits" / "humanoid_v1.json"))
    src = list(h["source_train_bodies"])
    assert all(LEGGED_ASSETS[k]["kind"] == "humanoid" for k in src)
    return ", ".join(src) + " + phum procedural"


def envs_esc(s: str) -> str:
    """Escape plain text for LaTeX; <dots>, <Psi0>, <br>, <sl>, <zb> are the only raw-LaTeX tokens."""
    s = s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace(r"\_", r"\_\allowbreak{}")
    return s.replace("<br>",r"\newline\scriptsize\ttfamily ").replace("<nl>", r"\newline ").replace("<zb>", r"\allowbreak{}").replace("<sl>", r"/\allowbreak{}").replace("<dots>", r"\ldots{}").replace("<Psi0>", r"$\Psi_0$")


def build_envs(repo: Path) -> str:
    _envs_deps()
    assert set(ENVS) >= {"mujoco/arm", "mujoco/dual", "mujoco/legged", "warp/legged", "simple", "computerworld"}
    arm_t, dual_t = tasks_in("mujoco/arm"), tasks_in("mujoco/dual")
    leg_all = tasks_in("mujoco/legged")
    leg_t = [t for t in leg_all if not t.startswith("h_")]
    hum_t = [t for t in leg_all if t.startswith("h_")]
    held_t = [t for t in hum_t if t in ("h_steps_carry", "h_gap_cart")]
    warp_t = tasks_in("warp/legged")
    cw_t = [t.split("/")[1] for t in tasks_in("computerworld")]
    simple_n = len(SIMPLE_TASKS)
    n_repro = sum(1 for v in SIMPLE_TASKS.values() if v[2].startswith("reproduced"))
    quad, _ = legged_bodies()
    rows = [
        ("arm <br>mujoco<sl>arm", arm_bodies(repo), ", ".join(arm_t), fam_cell("arm"),
         "teacher + learned; transfer split primary_v1 / armdiv"),
        ("dual <br>mujoco<sl>dual", "two mounted arms (pairs of the arm pool) or one dual body", ", ".join(dual_t), fam_cell("dual"),
         "parked, D-146: eval of external checkpoints; scripted teachers only"),
        ("quad <br>mujoco<sl>legged", quad, ", ".join(leg_t), fam_cell("legged"),
         "teacher + frozen tracker"),
        ("humanoid <br>mujoco<sl>legged", humanoid_bodies(repo),
         f"{len(hum_t) - len(held_t)} h_* tasks (h_steps, h_gap, h_walk, <dots>); held out: {', '.join(held_t)}",
         fam_cell("humanoid"), "teacher + frozen tracker; campaign in progress"),
        ("tracker <br>warp<sl>legged", "same legged and humanoid bodies, GPU-batched", ", ".join(["locomotion"] + [t for t in warp_t if t != "locomotion"]),
         "legged / humanoid<nl>(as above)", "tracker training only"),
        ("SIMPLE <br>simple", "g1_simple (G1, 36-d command)", f"{simple_n} G1Wholebody* tasks", fam_cell("psi0"),
         f"<Psi0> SIMPLE on Isaac Sim; path-traced renderer caveat; step-1 reproduced on {n_repro} of {simple_n} tasks"),
        ("CWorld <br>computer<zb>world", "cw_pointer (cursor, keys)", ", ".join(cw_t), fam_cell("pointer"),
         "scripted teacher; procedural strings v3"),
    ]
    lines = [r"% generated by make_figures.py table_envs from rrp code and frozen splits; do not edit",
             r"\begin{table*}[t]\footnotesize\setlength{\tabcolsep}{2.5pt}\renewcommand{\arraystretch}{1.1}",
             r"\caption{Environments, bodies, tasks, the relation families whose factors resolve in each env's net family "
             r"(implemented factors: total, net-side + probe), and the status of each track.}",
             r"\label{tab:envs}",
             r"\begin{tabular}{@{}" + "".join(r">{\raggedright\arraybackslash}p{%s\textwidth}" % w for w in ("0.12", "0.30", "0.22", "0.10", "0.18")) + r"@{}}",
             r"\toprule", r"env & bodies & tasks & relation family & status \\ \midrule"]
    for env, bodies, tasks, fam, status in rows:
        lines.append(" & ".join(envs_esc(c) for c in (env, bodies, tasks, fam, status)) + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"




# =================================================================== [U1] Fig. 2 (the attention logit, stretched)
# FIGURE 2 of the rrp paper: the attention logit, stretched.
# 
#     logit(i,j) = <q_i,k_j>/sqrt(d) + b_dist + b_dir + b_kin + b_contact + b_support/flow + b_held + g_task*b_next + b_ui
# 
# Every panel shows ONE term for ONE query token on a REAL simulator state. The values are computed by the rrp factor
# operators (`FactorSite.contributions`, `OPS[...].value`), NOT by a trained network and NOT by a re-implementation:
#   * bias-form terms (kin.ancestor, ix.force_flow, ui.label_for, ui.above) -> FactorSite(...).contributions
#   * aug-form PaPE (geo.pos3d)            -> FactorSite(...).contributions with hand-set (illustrative) coefficients;
#                                             asserted against the closed form -(r_j-r_i)^T M (r_j-r_i) + b^T (r_j-r_i)
#   * probe-sourced pair terms (ix.contact, ix.held_by, task.next_contact, ix.support) -> the relgen LABEL (simulator
#     truth), which is the supervision target of the probe; the deployed term is the probe's estimate (not trained here).
# Run (read-only against the repo):
# Deterministic: fixed seeds, no RNG other than the repo's name-seeded generators.
import textwrap
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
try:                                  # heavy / optional deps of Fig. 2 and Fig. 3 only (the tables do not need them)
    import torch
    import mujoco
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle, FancyArrowPatch
    FIG_DEPS_ERROR = None
except ImportError as _e:             # pragma: no cover
    FIG_DEPS_ERROR = _e
    LinearSegmentedColormap = Rectangle = FancyArrowPatch = None

F2_SEED = 3000001
F2_W, F2_H = 600, 450
GEOM, GRAPH, AMBER, VIOLET = "#2BB3C0", "#3A3F47", "#F2A33A", "#8C7CF0"
PAPER, MIDGREY = "#F7F7F5", "#8A8F98"
FAMILY_NAME = {GEOM: "geometry", GRAPH: "kinematics/UI", AMBER: "contact/support/grasp", VIOLET: "procedure"}


def _mix(c0, c1, t):
    c0, c1 = np.array(matplotlib.colors.to_rgb(c0)), np.array(matplotlib.colors.to_rgb(c1))
    return c0 * (1 - t) + c1 * t


def ramp(color):                      # sequential: pale -> family colour
    return LinearSegmentedColormap.from_list("seq", [_mix("#F7F7F5", color, 0.10), _mix("#F7F7F5", color, 0.55), color])


def diverging(pos, neg):              # signed: neg <- pale -> pos
    return LinearSegmentedColormap.from_list("div", [neg, "#F2F2EF", pos])


# =============================================================================================== camera / render
class Cam:
    def __init__(self, lookat, dist, az, el, fovy=30.0):
        self.lookat = np.array(lookat, float)
        self.dist, self.az, self.el, self.fovy = float(dist), float(az), float(el), float(fovy)


def pose_from_cam(c: Cam):
    az, el = np.radians(c.az), np.radians(c.el)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pos = c.lookat - c.dist * fwd
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, -up, fwd])                 # world -> camera (x right, y down, z forward)
    return pos, R


class View:
    def __init__(self, cam: Cam, w=F2_W, h=F2_H):
        self.pos, self.R = pose_from_cam(cam)
        self.f = (h / 2) / np.tan(np.radians(cam.fovy) / 2)
        self.w, self.h, self.fovy = w, h, cam.fovy
        self.t = -self.R @ self.pos

    def project(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        pc = P @ self.R.T + self.t
        z = pc[:, 2]
        uv = pc[:, :2] / np.maximum(z[:, None], 1e-6) * self.f + np.array([self.w / 2, self.h / 2])
        return uv, z

    def ray_dirs(self):
        u, v = np.meshgrid(np.arange(self.w) + 0.5, np.arange(self.h) + 0.5)
        d = np.stack([(u - self.w / 2) / self.f, (v - self.h / 2) / self.f, np.ones_like(u)], -1).reshape(-1, 3)
        d = d @ self.R                              # camera -> world (R orthonormal)
        return d / np.linalg.norm(d, axis=1, keepdims=True)


CLAY = np.array([0.90, 0.90, 0.88, 1.0])


def style_model(m, color_of):
    """Matte clay look; `color_of(body_name, geom_type, geom_id)` -> rgba or None (hidden). Mutates ONLY this process's model copy."""
    if m.nmat:
        m.mat_specular[:] = 0.0
        m.mat_shininess[:] = 0.0
        m.mat_reflectance[:] = 0.0
    for g in range(m.ngeom):
        bn = m.body(int(m.geom_bodyid[g])).name
        rgba = None if int(m.geom_group[g]) >= 3 else color_of(bn, int(m.geom_type[g]), g)
        if rgba is None:
            m.geom_rgba[g, 3] = 0.0
            m.geom_group[g] = 5                        # not rendered, not ray-cast
        else:
            m.geom_rgba[g] = rgba
            m.geom_matid[g] = -1
    m.vis.headlight.ambient[:] = [0.40, 0.40, 0.40]
    m.vis.headlight.diffuse[:] = [0.40, 0.40, 0.40]
    for li in range(m.nlight):                      # key light(s) of the source scene: keep, but tone down
        m.light_diffuse[li] *= 0.35
        m.light_specular[li] = 0.0
        m.light_castshadow[li] = 0
    m.vis.headlight.specular[:] = [0, 0, 0]
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, F2_W)
    m.vis.global_.offheight = max(m.vis.global_.offheight, F2_H)
    m.vis.quality.shadowsize = 4096
    m.vis.map.znear = 0.005
    m.vis.map.zfar = 20.0


def render_view(m, d, cam: Cam, seg=False):
    r = mujoco.Renderer(m, F2_H, F2_W)
    mc = mujoco.MjvCamera()
    mc.type = mujoco.mjtCamera.mjCAMERA_FREE
    mc.lookat[:] = cam.lookat
    mc.distance, mc.azimuth, mc.elevation = cam.dist, cam.az, cam.el
    m.vis.global_.fovy = cam.fovy
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[:3] = 1
    opt.sitegroup[:] = 0
    opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
    r.update_scene(d, camera=mc, scene_option=opt)
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
    r.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
    rgb = r.render().copy()
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=mc, scene_option=opt)
    sg = r.render().copy()
    r.close()
    return rgb, sg


def ray_world_points(m, d, view: View):
    """EXACT world point behind every pixel (mj_multiRay from the camera centre over the rendered geom groups)."""
    dirs = view.ray_dirs()
    n = len(dirs)
    geomid = np.zeros(n, np.int32)
    dist = np.zeros(n)
    gg = np.array([1, 1, 1, 0, 0, 0], np.uint8)
    normal = np.zeros(n * 3)
    mujoco.mj_multiRay(m, d, np.asarray(view.pos, np.float64), np.ascontiguousarray(dirs, np.float64).reshape(-1), gg, 1, -1, geomid, dist, normal, n, 20.0)
    hit = geomid >= 0
    pts = view.pos[None] + np.where(hit, dist, 0.0)[:, None] * dirs
    return pts.reshape(F2_H, F2_W, 3), geomid.reshape(F2_H, F2_W), hit.reshape(F2_H, F2_W)


# =============================================================================================== relational building blocks
def _rel():
    from rrp.policies.relations import base as RB
    from rrp.policies.relations import ops as RO
    return RB, RO


def tset(name, pos=None, extra=None, mask=None):
    RB, _ = _rel()
    n = None if pos is None else len(pos)
    fields = {}
    if pos is not None:
        fields["pos3d"] = torch.as_tensor(np.asarray(pos, np.float32))[None]
    for k, v in (extra or {}).items():
        fields[k] = torch.as_tensor(np.asarray(v, np.float32))[None]
        n = fields[k].shape[1] if n is None else n
    return RB.TokenSet(name, torch.ones(1, n, dtype=torch.bool) if mask is None else torch.as_tensor(mask)[None], fields=fields)


def pape_site(a_vec, b_vec, q_sets, site="ctx>pix", dim=4):
    """FactorSite(geo.pos3d) with the learned `_Coeff` weights zeroed and hand-set a, b (g=1, W_p=I): the REAL PapeOp kernel."""
    RB, RO = _rel()
    specs = RB.resolve(["geo.pos3d"])
    fs = RO.FactorSite(1, dim, site, specs, ("pos3d", "hidden"))
    m = fs.f["geo__pos3d"]
    a_vec = np.asarray(a_vec, np.float64)
    with torch.no_grad():
        m.Wp.copy_(torch.eye(3)[None])
        m.g.fill_(1.0)
        m.a.weight.zero_()
        m.b.weight.zero_()
        # softplus(bias) = a  (a = 0 is realised as a very negative bias)
        inv = np.where(a_vec > 1e-9, np.log(np.expm1(np.maximum(a_vec, 1e-9))), -40.0)
        m.a.bias.copy_(torch.as_tensor(inv, dtype=torch.float32))
        m.b.bias.copy_(torch.as_tensor(np.asarray(b_vec), dtype=torch.float32))
    a_eff = torch.nn.functional.softplus(m.a.bias).detach().numpy().astype(np.float64)
    return fs, a_eff, np.asarray(b_vec, np.float64)


def pape_values(rq, rk, a_vec, b_vec, site="ctx>pix"):
    """[Q,K] closed-form PaPE term from the real PapeOp kernel (per-query constant r_i^T M r_i + b^T r_i removed; it is
    softmax-invariant). Asserts equality with -(r_j-r_i)^T M (r_j-r_i) + b^T (r_j-r_i)."""
    RB, RO = _rel()
    q, k = site.split(">")
    assert q != k, "query and key token sets need distinct names"
    fs, a_eff, b = pape_site(a_vec, b_vec, None, site)
    rc = RB.RelCtx(sets={q: tset(q, rq), k: tset(k, rk)})
    xq = torch.zeros(1, len(rq), 4)
    xk = torch.zeros(1, len(rk), 4)
    kern = fs.contributions(rc, xq, xk)["geo.pos3d"][0, 0].double().numpy()                      # [Q,K]
    rq64, rk64 = np.asarray(rq, np.float64), np.asarray(rk, np.float64)
    c = (rq64 ** 2 * a_eff[None]).sum(1) + rq64 @ b                                             # r_i^T M r_i + b^T r_i
    val = kern - c[:, None]
    dlt = rk64[None] - rq64[:, None]
    closed = -(dlt ** 2 * a_eff).sum(-1) + dlt @ b
    err = float(np.abs(val - closed).max())
    scale = max(1.0, float(np.abs(closed).max()))
    assert err < 2e-4 * scale, f"PaPE kernel vs closed form: max err {err}"
    return val, err


def label_matrix(sv, ids, name):
    from rrp.harness.data import relgen as RG
    lab = RG.LABELS[name].fn(sv, RG.TokenIndex(sets={"ctx": ids}))
    return np.asarray(lab.value, np.float64), np.asarray(lab.valid)


# =============================================================================================== scenes
def arm_session(n_distractors=2, seed=F2_SEED):
    from rrp.bodies.catalog import workbench_robots
    from rrp.envs.mujoco.scenario import BUILDERS
    from rrp.envs.mujoco.session import Session
    sc = BUILDERS["pick_place"](workbench_robots()["panda_pg2"](), seed, n_distractors=n_distractors)
    s = Session(sc, seed=seed)
    s.reset()
    return s


def arm_until(phase, seed=F2_SEED, max_ticks=900, extra_ticks=0):
    """Scripted privileged teacher (PickPlaceTeacher) stepped until it enters `phase` (+extra ticks)."""
    from rrp.policies.teachers.arm import PickPlaceTeacher
    s = arm_session(seed=seed)
    T = PickPlaceTeacher(s, robot=0)
    k = 0
    while k < max_ticks:
        s.step(T.act())
        k += 1
        if T.phase == phase:
            break
    for _ in range(extra_ticks):
        s.step(T.act())
        k += 1
    return s, T, k


def entity_table(sv):
    ents = list(sv.entities())
    return [e.id for e in ents], np.array([e.pos for e in ents], np.float64), [e.kind for e in ents]


def arm_color_fn(object_bodies):
    def f(bn, typ, g):
        if typ == mujoco.mjtGeom.mjGEOM_PLANE:
            return np.array([0.80, 0.81, 0.82, 1.0])
        if bn in object_bodies:
            return np.array([0.93, 0.93, 0.92, 1.0])
        if bn.startswith("r0_") or bn.startswith("r1_"):
            return np.array([0.62, 0.64, 0.67, 1.0])
        if bn == "world":
            return np.array([0.78, 0.79, 0.80, 1.0])
        return np.array([0.70, 0.71, 0.72, 1.0])
    return f


# =============================================================================================== panel container
class Panel:
    def __init__(self, n, factor, slug, color, short, formula, badge, tag, img):
        self.n, self.factor, self.slug, self.color = n, factor, slug, color
        self.short, self.formula, self.badge, self.tag, self.img = short, formula, badge, tag, img
        self.marks = []
        self.cbar = None
        self.meta = {}


def blend(base, heat_rgb, mask, alpha=0.78):
    """Heat over the render, keeping the render's shading (heat * relative luminance) so 3-D shape stays readable."""
    b = base.astype(np.float32)
    shade = np.clip(b.mean(-1, keepdims=True) / (0.80 * 255.0), 0.55, 1.12)
    out = b.copy()
    tgt = heat_rgb * 255.0 * shade
    out[mask] = out[mask] * (1 - alpha) + tgt[mask] * alpha
    return out.clip(0, 255).astype(np.uint8)


def stats(values, names=None):
    v = np.asarray(values, float)
    j = int(np.argmax(v))
    return dict(min=float(v.min()), max=float(v.max()), argmax=(names[j] if names is not None else j))


def ring(P, uv, color="#E4572E", r=15):
    P.marks.append(dict(kind="ring", x=float(uv[0]), y=float(uv[1]), r=r, color=color))


def dot(P, uv, face, r=6, edge="#1D1F22"):
    P.marks.append(dict(kind="dot", x=float(uv[0]), y=float(uv[1]), r=r, face=face, edge=edge))


def text(P, uv, s, color="#1D1F22", size=1.0, ha="center", va="center", bg=True):
    P.marks.append(dict(kind="text", x=float(uv[0]), y=float(uv[1]), s=s, color=color, size=size, ha=ha, va=va, bg=bg))


def line(P, uvs, color="#1D1F22", lw=1.2, arrow=False):
    P.marks.append(dict(kind="line", pts=[[float(a), float(b)] for a, b in uvs], color=color, lw=lw, arrow=arrow))



# =============================================================================================== shared scene machinery
def arm_at_tick(n, seed=F2_SEED, setup=None):
    """PickPlaceTeacher (scripted privileged teacher) stepped exactly `n` ticks from reset; `setup(session)` may teleport objects first."""
    from rrp.policies.teachers.arm import PickPlaceTeacher
    s = arm_session(seed=seed)
    T = PickPlaceTeacher(s, robot=0)
    if setup is not None:
        setup(s)
    for _ in range(n):
        s.step(T.act())
    return s, T


BG = np.array([247, 247, 245], np.uint8)


def render_scene(m, d, cam, color_fn, min_agree=0.97):
    """Matte render + exact per-pixel world points (mj_multiRay) + geom ids; asserts ray/segmentation agreement > 97 %."""
    style_model(m, color_fn)
    mujoco.mj_forward(m, d)
    rgb, sg = render_view(m, d, cam)
    view = View(cam)
    pts, gid, hit = ray_world_points(m, d, view)
    both = hit & (sg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    agree = float((sg[..., 0][both] == gid[both]).mean())
    assert agree > min_agree, f"ray/segmentation agreement {agree:.3f}"
    rgb = rgb.copy()
    rgb[~hit] = BG
    body = np.where(hit, m.geom_bodyid[np.maximum(gid, 0)], -1)
    return dict(rgb=rgb, view=view, pts=pts, gid=gid, hit=hit, body=body, agree=agree, m=m)


def paint_bodies(sc, body_vals, cmap, vmin, vmax, alpha=0.88):
    """Heat on every pixel whose body is in body_vals {body_id: value}."""
    heat = np.zeros((F2_H, F2_W, 3))
    mask = np.zeros((F2_H, F2_W), bool)
    for b, v in body_vals.items():
        mk = sc["body"] == b
        mask |= mk
        heat[mk] = cmap(float(np.clip((v - vmin) / max(vmax - vmin, 1e-9), 0, 1)))[:3]
    return blend(sc["rgb"], heat, mask, alpha)


def entity_bodies(s, m):
    """{entity id: [body ids]} from the session's public body->entity map."""
    out = {}
    for bn, eid in s._body_entity_map().items():
        out.setdefault(eid, []).append(int(m.body(bn).id))
    return out


def uv_of(sc, P3):
    uv, z = sc["view"].project(np.asarray(P3, float))
    return uv


def palm_uv(s, sc):
    """Projected centre of the gripper's palm body (the 'gripper' entity's root body)."""
    return uv_of(sc, s.data.xpos[s.model.body("r0_wrist_palm").id])[0]


def in_frame(uv, pad=10):
    return pad < uv[0] < F2_W - pad and pad < uv[1] < F2_H - pad


ARM_CAM = Cam(lookat=[0.48, -0.08, 0.05], dist=0.55, az=168, el=-40, fovy=34.0)
ARM_FLOOR = lambda objs: arm_color_fn(objs)


def arm_scene(s, cam=ARM_CAM):
    objs = {o.sim_body for o in s.scenario.objects}
    sc = render_scene(s.model, s.data, cam, arm_color_fn(objs))
    sv = s.state_view()
    ids, pos, kinds = entity_table(sv)
    return sc, sv, ids, pos, kinds


# =============================================================================================== panels 1, 2 (PaPE, geo.pos3d)
def build_pape_panels(tick_n=29):
    s, T = arm_at_tick(tick_n)
    sc, sv, ids, pos, kinds = arm_scene(s)
    qi = ids.index("cube")
    rq = pos[qi:qi + 1]
    uvq = uv_of(sc, rq)[0]
    uvt = uv_of(sc, pos)
    pts = sc["pts"]
    floor = sc["hit"] & ~np.isin(sc["body"], [b for b in range(s.model.nbody) if s.model.body(b).name.startswith(("r0_", "r1_"))])
    out = []
    specs = [
        dict(n=1, slug="dist", a=np.array([40.0, 40.0, 40.0]), b=np.zeros(3), cmap=ramp(GEOM), vmin=-4.0, vmax=0.0,
             formula=r"$-(r_j-r_i)^{\top}M\,(r_j-r_i)$", short="b_dist", fmt="{:.1f}", cb=("0", "-4")),
        dict(n=2, slug="dir", a=np.zeros(3), b=np.array([12.0, 0.0, 0.0]), cmap=diverging(GEOM, "#9AA0A8"), vmin=-2.5, vmax=2.5,
             formula=r"$\tilde b^{\top}(r_j-r_i)$", short="b_dir", fmt="{:+.1f}", cb=("+", "-")),
    ]
    for sp in specs:
        vfun = lambda P3: pape_values(rq, P3, sp["a"], sp["b"])[0][0]
        vals = np.zeros((F2_H, F2_W))
        vals[floor] = vfun(pts[floor])
        t = np.clip((vals - sp["vmin"]) / (sp["vmax"] - sp["vmin"]), 0, 1)
        img = blend(sc["rgb"], sp["cmap"](t)[..., :3], floor, 0.80)
        tok, err = pape_values(rq, pos, sp["a"], sp["b"], site="ctx>pix")
        P = Panel(sp["n"], "geo.pos3d", f"geo_pos3d_{sp['slug']}", GEOM, sp["short"], sp["formula"],
                  "illustrative coefficients; learned per head", "teacher state (scripted)", img)
        P.cbar = (sp["cb"][0], sp["cb"][1], sp["cmap"])
        ring(P, uvq)
        for k, (i, uv) in enumerate(zip(ids, uvt)):
            if k != qi and in_frame(uv):
                dot(P, uv, "#FFFFFF", r=4)
                text(P, uv + np.array([0, -14]), sp["fmt"].format(tok[0, k]), size=0.9)
        pv = vals[floor]
        P.meta = dict(
            env="mujoco/arm", body="panda_pg2", task="pick_place(n_distractors=2)", seed=F2_SEED, tick=tick_n, phase=T.phase,
            query="cube", source_kind="given pos3d (simulator poses of the entity tokens) + hand-set PaPE coefficients (illustrative)",
            scene_source="scripted privileged teacher (PickPlaceTeacher), state replayed from reset",
            coefficients=dict(a=sp["a"].tolist(), b=sp["b"].tolist(), g=1.0, W_p="I"),
            term_over="every non-robot pixel (exact mj_multiRay world point per pixel through the real PapeOp kernel); numbers = token values",
            stats=dict(token=dict(min=float(tok.min()), max=float(tok.max()), argmax=ids[int(tok[0].argmax())]),
                       pixel=dict(min=float(pv.min()), max=float(pv.max()))),
            token_values={i: float(tok[0, k]) for k, i in enumerate(ids)},
            closed_form_max_abs_err=err, ray_seg_agreement=sc["agree"],
            code_path="FactorSite(['geo.pos3d']).contributions -> PapeOp.features (W_p=I, a=softplus(bias), b=bias, g=1, weights zeroed); per-query constant r_i^T M r_i + b^T r_i removed (softmax-invariant); asserted == -(r_j-r_i)^T M (r_j-r_i) + b^T (r_j-r_i)",
        )
        out.append(P)
    return out


# =============================================================================================== panel 3 (kin.ancestor on a G1 humanoid)
def build_kin_panel():
    from rrp.bodies import g1_simple as G
    from rrp.envs.mujoco import humanoid_scenes as HS
    from rrp.policies.relations.base import EdgeSet, RelCtx, TokenSet, resolve, get_factor, spec
    from rrp.policies.relations.ops import FactorSite, OPS
    m, sp, meta = HS.task_model("g1", "h_steps", {"h_frac": 0.02})
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    n = G.ACTION_DIM
    rel = torch.from_numpy(G.relation_matrix()).permute(1, 2, 0).unsqueeze(0)            # [1,T,T,R], real g1-dim-rel-v1
    rc = RelCtx(sets={"dims": TokenSet("dims", torch.ones(1, n, dtype=torch.bool))}, edges={"dims>dims": EdgeSet(G.RELATIONS, rel)})
    site = FactorSite(1, 8, "dims>dims", resolve(["kin.ancestor"]), ("edges:g1-dim-rel-v1",))
    assert [s_.name for s_ in site.specs] == ["kin.ancestor"], site.specs
    with torch.no_grad():
        site.f["kin__ancestor"].w.fill_(1.0)
    contrib = site.contributions(rc, torch.zeros(1, n, 8), torch.zeros(1, n, 8))["kin.ancestor"][0, 0].numpy()       # [Q,K]
    # independent check: walk the parent chain of the real morphology
    parent = {i: int(G.DIM_PARENT[i]) for i in range(n)}

    def chain(i):
        out, j, seen = set(), parent.get(i, -1), set()
        while j >= 0 and j not in seen:
            out.add(j); seen.add(j); j = parent.get(j, -1)
        return out
    for i in range(n):
        assert set(np.nonzero(contrib[i])[0].tolist()) == chain(i), f"kin.ancestor != parent-chain walk at dim {i}"
    qd = G.DIM_NAMES.index("l_wrist_yaw")
    anc = contrib[qd]
    # dim -> link body (arm / waist dims have a joint in the rendered model; finger / base dims do not)
    dim_body = {}
    for i, (nm, jn, *_r) in enumerate(G._DIMS):
        if jn is None:
            continue
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "r0_" + jn)
        if j >= 0:
            dim_body[i] = int(m.jnt_bodyid[j])
    cam = Cam(lookat=[0.0, 0.05, 0.86], dist=1.55, az=205, el=-6, fovy=34.0)
    base = lambda bn, t, g: None if bn == "world" else np.array([0.70, 0.72, 0.75, 1.0])
    sc = render_scene(m, d, cam, base, min_agree=0.94)     # thin mesh limbs: silhouette pixels differ between raster and ray
    dark = np.array(matplotlib.colors.to_rgb(GRAPH))
    heat = np.zeros((F2_H, F2_W, 3)); mask = np.zeros((F2_H, F2_W), bool)
    for i, b in dim_body.items():
        if anc[i] > 0:
            mk = sc["body"] == b
            mask |= mk; heat[mk] = dark
    img = blend(sc["rgb"], heat, mask, 0.92)
    P = Panel(3, "kin.ancestor", "kin_ancestor", GRAPH, "b_kin", r"$\mathbb{1}[\,j\ \mathrm{ancestor\ of}\ i\,]$",
              "given: kin_parent graph (body morphology)", "simulator state (G1 keyframe)", img)
    qb = dim_body[qd]
    ring(P, uv_of(sc, d.xpos[qb])[0], r=17)
    P.meta = dict(
        env="mujoco/humanoid (rrp.envs.mujoco.humanoid_scenes)", body="g1 (36 command-dim tokens, g1-dim-rel-v1)", task="h_steps keyframe 0 (static pose, no policy)",
        seed=F2_SEED, tick=0, query="l_wrist_yaw", source_kind="given (kinematic tree)", scene_source="simulator keyframe (no controller, no motion)",
        stats=dict(min=float(anc.min()), max=float(anc.max()), argmax=G.DIM_NAMES[int(np.argmax(anc))], n_ancestors=int(anc.sum()),
                   ancestors=[G.DIM_NAMES[j] for j in np.nonzero(anc)[0]]),
        drawn_dims=sorted(G.DIM_NAMES[i] for i in dim_body), not_drawn="finger and base dims have no link in the rendered MuJoCo model (still tokens, values in the op output)",
        parent_chain_check="all 36 rows equal an independent parent-chain walk of G.DIM_PARENT", ray_seg_agreement=sc["agree"],
        code_path="FactorSite(['kin.ancestor']).contributions -> ClosureOp.value over RelCtx edges 'dims>dims' = real bodies.g1_simple.relation_matrix(); site.f['kin__ancestor'].w = 1",
    )
    return P


# =============================================================================================== panels 4, 6 (ix.contact, ix.held_by)
def lab(sv, ids, name):
    from rrp.harness.data import relgen as RG
    return np.asarray(RG.LABELS[name].fn(sv, RG.TokenIndex(sets={"ctx": ids})).value, np.float64)


def build_contact_panel(tick_n=40):
    s, T = arm_at_tick(tick_n)
    cam = Cam(lookat=[0.497, -0.017, 0.035], dist=0.22, az=120, el=-20, fovy=34.0)
    sc, sv, ids, pos, kinds = arm_scene(s, cam)
    M = lab(sv, ids, "contact_pairs")[..., 0]
    gi, ci = ids.index("gripper"), ids.index("cube")
    eb = entity_bodies(s, s.model)
    cm = ramp(AMBER)
    vals = {b: M[gi, ids.index(e)] for e, bs in eb.items() if e in ids and e != "gripper" for b in bs}
    img = paint_bodies(sc, vals, cm, 0, 1)
    P = Panel(4, "ix.contact", "ix_contact", AMBER, "b_contact", r"$\mathrm{contact}(i,j)$",
              "label (sim truth) -> trains the probe; deployed term = probe estimate", "teacher state (scripted)", img)
    cps = [c for c in sv.contacts() if {c.a, c.b} == {"gripper", "cube"}]
    for c in cps:
        uv = uv_of(sc, c.pos)[0]
        if in_frame(uv, 4):
            dot(P, uv, "#FFFFFF", r=3.2, edge="#7A4A00")
    ring(P, palm_uv(s, sc), r=15)
    P.meta = dict(
        env="mujoco/arm", body="panda_pg2", task="pick_place(n_distractors=2)", seed=F2_SEED, tick=tick_n, phase=T.phase, query="gripper",
        source_kind="label (relgen contact_pairs, simulator contact truth; privileged)", scene_source="scripted privileged teacher state",
        stats=dict(min=float(M[gi].min()), max=float(M[gi].max()), argmax=ids[int(M[gi].argmax())]),
        token_values={e: float(M[gi, k]) for k, e in enumerate(ids)}, n_contact_points_drawn=len(cps), ray_seg_agreement=sc["agree"],
        code_path="relgen.LABELS['contact_pairs'].fn(StateView, TokenIndex) = the supervision target of the ix.contact bilinear probe (deployed bias = learned probe estimate; NOT computed here); white dots = StateView.contacts() positions",
    )
    return P


def build_held_panel(tick_n=58):
    s, T = arm_at_tick(tick_n)
    cam = Cam(lookat=[0.50, -0.017, 0.12], dist=0.26, az=120, el=-15, fovy=34.0)
    sc, sv, ids, pos, kinds = arm_scene(s, cam)
    M = lab(sv, ids, "held_pairs")[..., 0]
    gi, ci = ids.index("gripper"), ids.index("cube")
    eb = entity_bodies(s, s.model)
    cm = ramp(AMBER)
    vals = {b: M[ci, ids.index(e)] for e, bs in eb.items() if e in ids and e != "cube" for b in bs}
    img = paint_bodies(sc, vals, cm, 0, 1)
    P = Panel(6, "ix.held_by", "ix_held_by", AMBER, "b_held", r"$\mathrm{held\_by}(i,j)$",
              "label (sim truth) -> trains the probe; deployed term = probe estimate", "teacher state (scripted)", img)
    ring(P, uv_of(sc, pos[ci])[0], r=15)
    P.meta = dict(
        env="mujoco/arm", body="panda_pg2", task="pick_place(n_distractors=2), lift phase", seed=F2_SEED, tick=tick_n, phase=T.phase, query="cube",
        source_kind="label (relgen held_pairs, simulator truth; privileged)", scene_source="scripted privileged teacher state (the dual-arm handover is replaced by this single-arm lift)",
        stats=dict(min=float(M[ci].min()), max=float(M[ci].max()), argmax=ids[int(M[ci].argmax())]),
        token_values={e: float(M[ci, k]) for k, e in enumerate(ids)}, cube_z=float(pos[ci][2]), ray_seg_agreement=sc["agree"],
        code_path="relgen.LABELS['held_pairs'].fn(StateView, TokenIndex) = supervision target of the ix.held_by bilinear probe (deployed bias = learned probe estimate; NOT computed here)",
    )
    return P


# =============================================================================================== panel 5 (ix.force_flow on a stack)
def build_flow_panel():
    from rrp.harness.data import relgen as RG
    from rrp.harness.data.relgen import support as SUP
    from rrp.policies.relations.base import EdgeSet, RelCtx, resolve
    from rrp.policies.relations.ops import FactorSite

    def setup(s):
        for k, (b, z) in enumerate((("distractor0", 0.022), ("distractor1", 0.0665), ("cube", 0.111))):
            s.teleport_object(b, [0.48, -0.10, z + 0.002 * k])

    s, T = arm_at_tick(10, setup=setup)
    cam = Cam(lookat=[0.48, -0.10, 0.07], dist=0.34, az=165, el=-22, fovy=34.0)
    sc, sv, ids, pos, kinds = arm_scene(s, cam)
    n = len(ids)
    A = lab(sv, ids, "support_pairs")[..., 0]                                  # A[a, b] = a is directly below / supports b
    C_label = lab(sv, ids, "support_closure")[..., 0]
    idx = {e: k for k, e in enumerate(ids)}
    bottom = max((e for e in ids if e in ("cube", "obj:distractor0", "obj:distractor1")), key=lambda e: -pos[idx[e]][2])
    rc = RelCtx(sets={"ctx": tset("ctx", pos)}, edges={"ctx>ctx#support-v1@gt": EdgeSet(("support",), torch.as_tensor(A, dtype=torch.float32)[None, :, :, None], prov="privileged")})
    site = FactorSite(1, 3, "ctx>ctx", resolve([{"name": "ix.force_flow", "source": "gt"}]), ("edges:support-v1",))
    assert [s_.name for s_ in site.specs] == ["ix.force_flow"], site.specs
    with torch.no_grad():
        site.f["ix__force_flow"].w.fill_(1.0)
    flow = site.contributions(rc, torch.zeros(1, n, 3), torch.zeros(1, n, 3))["ix.force_flow"][0, 0].numpy()
    assert np.array_equal(flow > 0.5, C_label > 0.5), "ix.force_flow (ClosureOp over gt support edges) != relgen support_closure label"
    qb = idx[bottom]
    eb = entity_bodies(s, s.model)
    cm = ramp(AMBER)
    vals = {b: flow[qb, idx[e]] for e, bs in eb.items() if e in idx and e not in ("gripper", "target") for b in bs}
    img = paint_bodies(sc, vals, cm, 0, 1)
    P = Panel(5, "ix.force_flow", "ix_force_flow", AMBER, "b_flow", r"$\mathrm{flow}^{+}(\mathrm{support})(i,j)$",
              "label (sim truth) -> trains the probe; deployed term = probe estimate", "teleported stack + settle", img)
    ring(P, uv_of(sc, pos[qb])[0], r=15)
    order = sorted([e for e in ids if e in ("cube", "obj:distractor0", "obj:distractor1")], key=lambda e: pos[idx[e]][2])
    for e in order:
        uv = uv_of(sc, pos[idx[e]])[0]
        if e != bottom:
            text(P, uv + np.array([36, 0]), f"{flow[qb, idx[e]]:.0f}", size=1.0)
    P.meta = dict(
        env="mujoco/arm", body="panda_pg2", task="pick_place cubes teleported into a 3-stack (Session.teleport_object, logged as an intervention) then 10 ticks of settling", seed=F2_SEED, tick=10,
        query=bottom, source_kind="gt: privileged support_pairs edges (relgen) closed by the REAL flow op; deployed term = closure of the ix.support probe estimate",
        scene_source="figure-only intervention scene (teleported stack); arm held by the teacher, far above",
        stack_bottom_to_top=order, stats=dict(min=float(flow[qb].min()), max=float(flow[qb].max()), argmax=ids[int(flow[qb].argmax())]),
        closure_check="FactorSite flow output == relgen.support_closure label for all token pairs", token_values={e: float(flow[qb, k]) for k, e in enumerate(ids)},
        ray_seg_agreement=sc["agree"],
        code_path="FactorSite([{'name':'ix.force_flow','source':'gt'}]).contributions -> ClosureOp.value over rc.edges['ctx>ctx#support-v1@gt'] = relgen support_pairs; convention: A[a,b] = a supports b, so the query row reaches everything it (transitively) carries",
    )
    return P


# =============================================================================================== panel 7 (task.next_contact)
def build_next_panel(tick_n=29):
    from rrp.harness.data import relgen as RG
    import rrp.harness.data.relgen.transforms  # noqa: F401  (registers TRANSFORMS)
    from rrp.harness.data.relgen import TRANSFORMS
    from rrp.harness.data.relgen import task as TK
    s, T = arm_at_tick(tick_n)
    sc, sv, ids, pos, kinds = arm_scene(s)
    cands = ["cube", "obj:distractor0", "obj:distractor1"]
    # evidence: a synthetic progressive-reveal schedule (task text rules out the blue cube at t=0, the yellow one at t=1)
    evidence = [{"t": 0, "excludes": ["obj:distractor0"]}, {"t": 1, "excludes": ["obj:distractor1"]}]
    smp = TK.next_contact_sample(cands, sv, evidence=evidence)
    outs = {}
    for t in (0, 1):
        [o] = TRANSFORMS["reveal"].fn(smp, np.random.default_rng(0), {"schedule": [t]})
        outs[t] = o
    lab_key = next(iter(outs[0]["labels"]))
    post = {t: {c: float(v) for c, v in zip(cands, np.asarray(outs[t]["labels"][lab_key]["value"]).reshape(-1))} for t in outs}
    t_show = 0
    q = post[t_show]
    eb = entity_bodies(s, s.model)
    cm = ramp(VIOLET)
    vals = {b: q[e] for e, bs in eb.items() if e in q for b in bs}
    img = paint_bodies(sc, vals, cm, 0, 1)
    gi = ids.index("gripper")
    P = Panel(7, "task.next_contact", "task_next_contact", VIOLET, "g_task b_next", r"$g_{\rm task}\,\mathrm{next}(i,j)$",
              "label (sim truth) -> trains the probe; deployed term = probe estimate", "teacher state; synthetic reveal", img)
    P.cbar = ("1", "0", cm)
    for e in cands:
        uv = uv_of(sc, pos[ids.index(e)])[0]
        text(P, uv + np.array([0, -16]), f"{q[e]:.2f}", size=1.0)
    ring(P, palm_uv(s, sc) if in_frame(palm_uv(s, sc), 14) else np.array([F2_W * 0.55, 45.0]), r=15)
    now = np.asarray(RG.LABELS["next_contact"].fn(sv, RG.TokenIndex(sets={"ctx": ids})).value)[:, 0]
    P.meta = dict(
        env="mujoco/arm", body="panda_pg2", task="pick_place(n_distractors=2), approach", seed=F2_SEED, tick=tick_n, phase=T.phase, query="gripper (manipulator)",
        source_kind="label: relgen task.next_contact_sample + TRANSFORMS['reveal'] (R9, unmodified) posterior over candidate targets; evidence schedule is SYNTHETIC",
        scene_source="scripted privileged teacher state", evidence=evidence, posterior_by_t=post, shown_t=t_show,
        stats=dict(min=float(min(q.values())), max=float(max(q.values())), argmax=max(q, key=q.get)),
        next_contact_label_now={e: float(now[k]) for k, e in enumerate(ids)}, ray_seg_agreement=sc["agree"],
        code_path="relgen.task.next_contact_sample(candidates, StateView, evidence) -> TRANSFORMS['reveal'].fn(schedule=[t]) posterior q(e) ~ prior * 1[e not excluded]; the deployed bias is the task-gated bilinear probe (NOT computed here)",
    )
    return P


# =============================================================================================== panel 8 (ui.label_for + ui.above)
CROP = (122, 215, 392, 418)                      # x0, y0, x1, y1 of the 960x640 desktop frame -> 600x450 panel image


def build_ui_panel():
    from rrp.envs.computerworld import UI_REL_VOCAB, make_env, scene_widgets, ui_edges, ui_public_fields
    from rrp.policies.relations.base import EdgeSet, RelCtx, TokenSet, resolve
    from rrp.policies.relations.ops import FactorSite
    from PIL import Image
    env = make_env(task="cw/fill_form", seed=F2_SEED)
    table = env.slots.assign(scene_widgets(env.scene()))
    img0 = np.asarray(env.render())[..., :3]
    E = ui_edges(table)                                                                  # [T,T,R] bool
    fld = ui_public_fields(table)
    n = len(table)
    ts = TokenSet("ui", torch.ones(1, n, dtype=torch.bool), fields={"zlayer": torch.as_tensor(np.asarray(fld["zlayer"], np.float32)).view(1, n, 1)})
    rc = RelCtx(sets={"ui": ts}, edges={"ui>ui": EdgeSet(UI_REL_VOCAB, torch.as_tensor(E, dtype=torch.bool)[None])})
    site = FactorSite(1, 4, "ui>ui", resolve(["ui.label_for", "ui.above"]), ("edges:ui-rel-v1", "zlayer"))
    assert {s_.name for s_ in site.specs} == {"ui.label_for", "ui.above"}, site.specs
    with torch.no_grad():
        site.w.fill_(1.0)
        site.f["ui__above"].w.fill_(1.0)
    con = site.contributions(rc, torch.zeros(1, n, 4), torch.zeros(1, n, 4))
    lf, ab = con["ui.label_for"][0, 0].numpy(), con["ui.above"][0, 0].numpy()
    qi = int(np.nonzero(lf.any(1))[0][0])           # first widget that is a label_for source
    x0, y0, x1, y1 = CROP
    crop = Image.fromarray(img0[y0:y1, x0:x1]).resize((F2_W, F2_H), Image.LANCZOS)
    img = np.asarray(crop)
    sx, sy = F2_W / (x1 - x0), F2_H / (y1 - y0)

    def bx(w):
        a, b, c, d_ = w["box"]
        return (a - x0) * sx, (b - y0) * sy, (c - x0) * sx, (d_ - y0) * sy
    P = Panel(8, "ui.label_for", "ui_label_for", GRAPH, "b_ui", r"$\mathrm{label\_for}(i,j),\ \mathrm{above}(i,j)$",
              "given: public UI tree (no probe)", "computerworld (rendered); grey frame: ui.above", img)
    a_, b_, c_, d__ = bx(table[qi])
    P.marks.append(dict(kind="rect", x0=a_ - 3, y0=b_ - 3, x1=c_ + 3, y1=d__ + 3, color="#E4572E", lw=2.2, z=6))
    for j in np.nonzero(lf[qi])[0]:
        a, b, c, d2 = bx(table[j])
        P.marks.append(dict(kind="rect", x0=a, y0=b, x1=c, y1=d2, color=GRAPH, lw=1.6, fill=True, fc=GRAPH, alpha=0.40, z=3))
        P.marks.append(dict(kind="rect", x0=a, y0=b, x1=c, y1=d2, color=GRAPH, lw=1.6, z=5))
        text(P, ((a + c) / 2, (b + d2) / 2), "1", size=1.3)
    shown = []
    for j in range(n):
        if j == qi or lf[qi, j] or ab[qi, j] == 0:
            continue
        a, b, c, d2 = bx(table[j])
        if c < 0 or a > F2_W or d2 < 0 or b > F2_H:
            continue
        a, b, c, d2 = max(a, 3), max(b, 3), min(c, F2_W - 3), min(d2, F2_H - 3)
        if (c - a) * (d2 - b) > 0.5 * F2_W * F2_H:       # a container covering the crop: draw its edge only
            P.marks.append(dict(kind="rect", x0=a, y0=b, x1=c, y1=d2, color="#6B7078", lw=1.2, z=2))
        else:
            P.marks.append(dict(kind="rect", x0=a, y0=b, x1=c, y1=d2, color="#6B7078", lw=1.0, z=2))
        if (c - a) * (d2 - b) <= 0.5 * F2_W * F2_H and c - a >= 70 and d2 - b >= 24:   # label only boxes it fits inside
            text(P, (min(max(c - 34, 40), F2_W - 40), b + 16), f"above {ab[qi, j]:+.0f}", size=0.9, ha="center")
        shown.append((j, float(ab[qi, j])))
    sel = [(j, float(ab[qi, j])) for j in range(n) if ab[qi, j] != 0]
    P.meta = dict(
        env="computerworld (rrp.envs.computerworld, cw-site)", body="desktop UI (public widget tree)", task="cw/fill_form", seed=F2_SEED, tick=0,
        query=f"slot {qi} (role=label, box={table[qi]['box']})", source_kind="given (public role / window / scene order / z-layer); no probe, no privileged input",
        scene_source="rendered computerworld desktop frame (cropped); no teacher, no policy",
        stats=dict(label_for=dict(min=float(lf[qi].min()), max=float(lf[qi].max()), argmax=int(lf[qi].argmax())),
                   above=dict(min=float(ab[qi].min()), max=float(ab[qi].max()), argmax=int(ab[qi].argmax()), n_nonzero=int((ab[qi] != 0).sum()))),
        label_for_partners=[int(j) for j in np.nonzero(lf[qi])[0]], crop=list(CROP), above_drawn=shown, n_slots=n,
        code_path="FactorSite(['ui.label_for','ui.above']).contributions -> EdgeOp (public ui_edges, channel label_for) and OrderOp (public zlayer); site.w = 1, f['ui__above'].w = 1",
    )
    return P
# =============================================================================================== drawing
MONO = "DejaVu Sans Mono"


def draw_panel(ax, P: Panel, fs, frame_lw, show_title=False):
    """One panel into `ax` (pixel coordinates 0..W, 0..H). fs = base font size (pt); frame_lw in pt."""
    ax.imshow(P.img, extent=(0, F2_W, F2_H, 0), interpolation="bilinear", zorder=0)
    ax.set_xlim(0, F2_W)
    ax.set_ylim(F2_H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(P.color)
        sp.set_linewidth(frame_lw)
    pxpt = 600.0 / (ax.get_position().width * ax.figure.get_size_inches()[0] * 72.0)     # image px per pt in this axes
    for mk in P.marks:
        k = mk["kind"]
        if k == "ring":
            ax.add_patch(plt.Circle((mk["x"], mk["y"]), mk["r"] * 1.05, fill=False, ec="white", lw=fs * 0.55, zorder=5))
            ax.add_patch(plt.Circle((mk["x"], mk["y"]), mk["r"] * 1.05, fill=False, ec=mk["color"], lw=fs * 0.28, zorder=6))
        elif k == "dot":
            ax.add_patch(plt.Circle((mk["x"], mk["y"]), mk["r"], fc=mk["face"], ec=mk["edge"], lw=fs * 0.12, zorder=4))
        elif k == "text":
            kw = dict(bbox=dict(boxstyle="round,pad=0.12", fc=(1, 1, 1, 0.78), ec="none")) if mk["bg"] else {}
            ax.text(mk["x"], mk["y"], mk["s"], color=mk["color"], fontsize=fs * 0.82 * mk["size"], ha=mk["ha"], va=mk["va"],
                    family=MONO, zorder=7, **kw)
        elif k == "line":
            pts = np.array(mk["pts"])
            if mk["arrow"]:
                ax.add_patch(FancyArrowPatch(pts[0], pts[-1], arrowstyle="-|>", mutation_scale=fs * 1.3, lw=mk["lw"] * fs * 0.2,
                                             color=mk["color"], zorder=5, shrinkA=0, shrinkB=0))
            else:
                ax.plot(pts[:, 0], pts[:, 1], color=mk["color"], lw=mk["lw"] * fs * 0.2, zorder=3, solid_capstyle="round")
        elif k == "rect":
            ax.add_patch(Rectangle((mk["x0"], mk["y0"]), mk["x1"] - mk["x0"], mk["y1"] - mk["y0"], fill=mk.get("fill", False),
                                   fc=mk.get("fc", "none"), ec=mk["color"], lw=mk["lw"] * fs * 0.2, zorder=mk.get("z", 3), alpha=mk.get("alpha", 1)))
    # top-left provenance chip (scene source), bottom badge (term source)
    ax.text(8, 8, P.tag, color="white", fontsize=fs * 0.74, ha="left", va="top", family=MONO, zorder=8,
            bbox=dict(boxstyle="square,pad=0.25", fc=(0.11, 0.12, 0.13, 0.82), ec="none"))
    ax.text(8, F2_H - 8, "\n".join(textwrap.wrap(P.badge, 40 if fs < 6 else 52)), color="white", fontsize=fs * 0.80, ha="left", va="bottom", zorder=8, wrap=False,
            bbox=dict(boxstyle="square,pad=0.25", fc=P.color if P.color != "#F2A33A" else "#B87414", ec="none", alpha=0.92))
    if P.cbar is not None:
        hi, lo, cm = P.cbar
        ins = ax.inset_axes([0.885, 0.50, 0.03, 0.30])
        ins.imshow(np.linspace(1, 0, 64)[:, None], cmap=cm, aspect="auto", extent=(0, 1, 0, 1))
        ins.set_xticks([])
        ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_linewidth(0.3)
        for yy, s_ in ((0.80, hi), (0.50, lo)):
            ax.text(0.875, yy, s_, color="#1D1F22", fontsize=fs * 0.7, ha="right", va="center", family=MONO, zorder=8, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.1", fc=(1, 1, 1, 0.75), ec="none"))


def save_panel_png(P: Panel, path):
    fig = plt.figure(figsize=(3.0, 2.25), dpi=200)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    # standalone: image fills the frame; factor id (mono) + formula in a header band drawn over the image
    draw_panel(ax, P, fs=7.0, frame_lw=1.5)
    ax.text(F2_W - 8, 8, P.factor, color=P.color if P.color != "#F2A33A" else "#B87414", fontsize=7.4, ha="right", va="top",
            family=MONO, zorder=8, bbox=dict(boxstyle="square,pad=0.25", fc=(1, 1, 1, 0.9), ec=P.color, lw=0.8))
    ax.text(F2_W - 8, 52, P.formula, color="#1D1F22", fontsize=7.4, ha="right", va="top", zorder=8,
            bbox=dict(boxstyle="square,pad=0.25", fc=(1, 1, 1, 0.85), ec="none"))
    fig.savefig(path, dpi=200)
    plt.close(fig)




def composite(panels, path_pdf, path_png=None):
    """The whole figure* (7.1 in wide): equation row in mathtext, panels alternating above / below their term, leader lines."""
    FW, FH = 7.1, 4.62
    fig = plt.figure(figsize=(FW, FH), dpi=200)
    fig.patch.set_facecolor("white")
    rend = fig.canvas.get_renderer()
    ACOL = {AMBER: "#B87414"}
    col = lambda P: ACOL.get(P.color, P.color)

    def A(x, y, w, h):
        return fig.add_axes([x / FW, y / FH, w / FW, h / FH])

    pw, ph = 1.64, 1.64 * 0.75
    xcol = [0.08 + 1.72 * k for k in range(4)]
    y_eq = 2.31
    top_y, bot_y = FH - 0.05 - ph, 0.05
    # ---- equation row: measure the pieces, then place them left to right so the row spans the full width
    pieces = [(r"$\mathrm{logit}(i,j)\;=\;\dfrac{\langle q_i,k_j\rangle}{\sqrt{d}}$", "#1D1F22")]
    terms = [r"$b_{\rm dist}$", r"$b_{\rm dir}$", r"$b_{\rm kin}$", r"$b_{\rm contact}$", r"$b_{\rm supp/flow}$", r"$b_{\rm held}$",
             r"$g_{\rm task}\,b_{\rm next}$", r"$b_{\rm ui}$"]
    for t in terms:
        pieces += [("$+$", "#6B7078"), (t, None)]
    fs0 = 10.0
    ws = []
    for tx, _ in pieces:
        t = fig.text(0, 0, tx, fontsize=fs0)
        ws.append(t.get_window_extent(rend).width / fig.dpi)
        t.remove()
    gap0 = 0.05
    avail = FW - 0.10
    scale = min(1.12, (avail - gap0 * (len(pieces) - 1)) / sum(ws))
    fs = fs0 * scale
    xs, x = [], 0.05
    for w_ in ws:
        xs.append(x)
        x += w_ * scale + gap0
    term_cx = []
    for k, (tx, c) in enumerate(pieces):
        P = None
        if k >= 1 and (k % 2 == 0):
            P = panels[k // 2 - 1]
        fig.text(xs[k] / FW, y_eq / FH, tx, fontsize=fs, ha="left", va="center", color=(col(P) if P else c))
        if P is not None:
            term_cx.append(xs[k] + ws[k] * scale / 2)
    lead_cx = xs[0] + ws[0] * scale * 0.62
    fig.text(lead_cx / FW, (y_eq - 0.33) / FH, "learned, data-dependent (no heat)", fontsize=4.8, color="#555", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.25", fc="#EEEEEC", ec="#BBB", lw=0.5))
    # ---- panels + label blocks + leaders
    for P in panels:
        top = (P.n % 2 == 1)
        c = (P.n - 1) // 2 if top else (P.n - 2) // 2
        x0, y0 = xcol[c], (top_y if top else bot_y)
        ax = A(x0, y0, pw, ph)
        draw_panel(ax, P, fs=5.6, frame_lw=1.44)
        cxp = x0 + pw / 2
        l1 = y0 - 0.15 if top else y0 + ph + 0.15
        l2 = y0 - 0.30 if top else y0 + ph + 0.30
        fig.text(cxp / FW, l1 / FH, P.formula, fontsize=6.6, ha="center", va="center")
        fig.text(cxp / FW, l2 / FH, P.factor, fontsize=5.6, ha="center", va="center", family=MONO, color=col(P))
        ey = y_eq + (0.13 if top else -0.13)
        ye = l2 + (-0.08 if top else 0.08)
        tcx = term_cx[P.n - 1]
        fig.add_artist(plt.Line2D([tcx / FW, cxp / FW], [ey / FH, ye / FH], color=P.color, lw=0.8, alpha=0.95, transform=fig.transFigure))
    fig.savefig(path_pdf)
    if path_png:
        fig.savefig(path_png, dpi=200)
    plt.close(fig)


CAPTION = r"""% Figure 2 caption (generated by fig2.py; the figure itself is figures/fig2.pdf)
\caption{\textbf{The attention logit, one term at a time.}
Each panel stretches one term of $\mathrm{logit}(i,j)$ for a single query token (ring) over a real simulator state; frame colour is the relation family
(geometry, kinematics/UI, contact/support/grasp, procedure). The content term $\langle q_i,k_j\rangle/\sqrt{d}$ is learned and data-dependent, so it has no panel.
\emph{The values shown are computed by the factor operators of \texttt{rrp.policies.relations} on real simulator states; they are not trained attention.}
Panels 1--2 run the PaPE kernel with hand-set, illustrative coefficients (the model learns them per head); panels 3 and 8 are public, given structure (G1 kinematic tree; computerworld widget tree).
For probe-sourced pair terms (panels 4--7) the colour shows the simulator-truth label that trains the probe; the deployed term is the probe's estimate, which is not shown.
Scenes: scripted-teacher Panda pick-and-place states (1, 2, 4, 6, 7), a teleported three-cube stack (5), a G1 keyframe (3), a rendered desktop UI (8).
Planned relation factors are omitted.}
"""


def fig2(out_dir=None):
    if FIG_DEPS_ERROR is not None:
        raise SystemExit(f"fig2 needs torch, mujoco and matplotlib on PYTHONPATH: {FIG_DEPS_ERROR}")
    t0 = time.time()
    out_dir = os.path.abspath(out_dir or OUT)
    figs = os.path.join(out_dir, "figures")
    os.makedirs(figs, exist_ok=True)
    from rrp.harness.data import relgen as RG
    RG.load_families()
    from rrp.policies.relations import base as RB
    from rrp.policies.relations import catalog  # noqa: F401
    torch.manual_seed(F2_SEED)
    np.random.seed(F2_SEED)
    panels = {}
    for P in build_pape_panels():
        panels[P.n] = P
    for fn in (build_kin_panel, build_contact_panel, build_flow_panel, build_held_panel, build_next_panel, build_ui_panel):
        P = fn()
        panels[P.n] = P
    plist = [panels[k] for k in sorted(panels)]
    assert [p.n for p in plist] == list(range(1, 9))
    for P in plist:
        save_panel_png(P, os.path.join(figs, f"fig2_panel_{P.n}_{P.factor.replace('.', '_')}.png"))
    composite(plist, os.path.join(figs, "fig2.pdf"), os.path.join(figs, "fig2_preview.png"))
    planned = sorted(n for n, d in RB.FACTORS.items() if getattr(d, "status", "implemented") == "planned")
    rec = []
    for P in plist:
        m = dict(P.meta)
        rec.append(dict(panel=P.n, factor=P.factor, term=P.short, family=FAMILY_NAME[P.color], frame_color=P.color,
                        png=f"figures/fig2_panel_{P.n}_{P.factor.replace('.', '_')}.png", badge=P.badge, scene_tag=P.tag, **m))
    doc = dict(
        figure="fig2", seed=F2_SEED, runtime_s=round(time.time() - t0, 1),
        honesty="values are computed by rrp factor operators (FactorSite.contributions / OPS value) or are relgen simulator-truth labels, on real simulator states; NOT trained attention",
        omitted_planned_factors=planned, panels=rec,
    )
    with open(os.path.join(figs, "fig2_panels.json"), "w") as f:
        json.dump(doc, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return doc


# =================================================================== [U2] Fig. 3 (environment tiles)
# Figure 3: the environments, one tile each, with an unmistakable source tag.
# 
# 
# Tiles (480x360, rendered with MuJoCo EGL; no Isaac Sim):
#   arm       mujoco/arm   pick_place, panda_pg2, seed 0   frame mid-episode of the SCRIPTED TEACHER (privileged)
#   dual      mujoco/dual  handover (parked, D-146)        frame mid-episode of the SCRIPTED TEACHER (privileged)
#   quad      mujoco/legged go2                            RESET STATE (no controller running)
#   humanoid  mujoco/legged h_steps, t1                    RESET STATE (scenario built without a session: no t1 tracker on this host)
#   cworld    computerworld cw/fill_form                   RESET STATE (no controller running)
# Deterministic: fixed seeds, fixed tick, no randomness outside the seeded env. `fig3(out_dir)` returns the PDF path.
F3_W, F3_H = 480, 360
F3_SEED = 0
F3_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# source tag -> (text lines, badge colour); teacher/scripted marks follow AGENTS.md (unmistakable)
F3_TAGS = {
    "teacher": (("SCRIPTED TEACHER", "(privileged)"), (194, 87, 122)),
    "reset": (("RESET STATE", "(no controller)"), (110, 110, 110)),
    "rl_tracker": (("RL EXPERT + TRACKER",), (43, 179, 192)),
}


def _free_cam(env, offset, distance: float, azimuth: float, elevation: float):
    """A free camera aimed at the body of the first free joint plus `offset` (metres)."""
    import mujoco
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    m = env.model
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    root = env.data.xpos[m.jnt_bodyid[free[0]]] if free else env.data.qpos[:3]
    c.lookat[:] = np.asarray(root, float) + np.asarray(offset, float)
    c.distance, c.azimuth, c.elevation = distance, azimuth, elevation
    return c


def _render(env, camera, w: int = F3_W, h: int = F3_H) -> np.ndarray:
    import mujoco
    m = env.model
    r = mujoco.Renderer(m, h, w)
    names = [m.camera(i).name for i in range(m.ncam)]
    if isinstance(camera, mujoco.MjvCamera):
        r.update_scene(env.data, camera=camera)
    elif camera in names:
        r.update_scene(env.data, camera=camera)
    else:
        r.update_scene(env.data)
    img = r.render().copy()
    r.close()
    return img


def _first_cam(env, prefs) -> str | None:
    names = [env.model.camera(i).name for i in range(env.model.ncam)]
    return next((c for c in prefs if c in names), None)


def _teacher_frame(env, task: str, teacher_cls_or_factory, stop_tick: int, cam_prefs) -> np.ndarray:
    """Run the scripted teacher through `harness.rollout` and keep the frame at `stop_tick` (or the last one)."""
    from rrp.harness import hooks as F3_H
    from rrp.harness.rollout import rollout
    from rrp.policies.teachers import TeacherPolicy
    teacher = teacher_cls_or_factory(env)
    pol = TeacherPolicy(task, lambda e: teacher, "fig3:teacher", ("joint_position", "gripper"))
    cam = _first_cam(env, cam_prefs)
    keep: dict = {}
    tick = [0]

    def frame(i, e, act, step):
        tick[0] += 1
        if tick[0] <= stop_tick:
            keep["img"] = _render(e, cam)

    rollout(lambda _sd: env, pol, F3_H.budget_task(task, env.spec.env_id), [F3_SEED], batch=1, max_steps=stop_tick,
            hooks=[F3_H.EndWhen(lambda i, e: bool(teacher.done) or tick[0] >= stop_tick), F3_H.Recorder(on_step=frame)])
    return keep["img"]


def tile_arm() -> tuple[np.ndarray, str]:
    from rrp.envs.base import make_env
    from rrp.policies.teachers.arm import PickPlaceTeacher
    env = make_env("mujoco/arm", task="pick_place", body="panda_pg2", seed=F3_SEED)
    return _teacher_frame(env, "pick_place", PickPlaceTeacher, 170, ("front",)), "teacher"


def tile_dual() -> tuple[np.ndarray, str]:
    from rrp.policies.teachers.dual import TEACHERS
    from rrp.policies.teachers.dual_validate import make_session
    env = make_session("handover", "panda_pg2__ur5e_pg2", F3_SEED)
    return _teacher_frame(env, "handover", TEACHERS["handover"], 200, ("front",)), "teacher"


def tile_quad() -> tuple[np.ndarray, str]:
    from rrp.envs.base import make_env
    env = make_env("mujoco/legged", task="foothold_steps", body="go2", seed=F3_SEED)
    return _render(env, _free_cam(env, (0.0, 0.0, 0.0), 1.8, 135.0, -18.0)), "reset"


def tile_humanoid() -> tuple[np.ndarray, str]:
    """t1 on the h_steps terrain. No frozen t1 tracker exists on this host, so the scene is built without a session
    (scenario model + home keyframe), which is all a reset-state tile needs."""
    import mujoco
    from types import SimpleNamespace
    from rrp.envs.mujoco.humanoid_scenes import build_h_steps
    sc = build_h_steps("t1", F3_SEED)
    d = mujoco.MjData(sc.model)
    if sc.model.nkey:
        mujoco.mj_resetDataKeyframe(sc.model, d, 0)
    mujoco.mj_forward(sc.model, d)
    env = SimpleNamespace(model=sc.model, data=d)
    return _render(env, _free_cam(env, (0.9, 0.0, 0.3), 3.6, 150.0, -14.0)), "reset"


def tile_cworld() -> tuple[np.ndarray, str]:
    from rrp.envs.base import make_env
    env = make_env("computerworld", task="cw/fill_form", body="cw_pointer", seed=F3_SEED)
    return np.asarray(env.render())[..., :3], "reset"


F3_TILES = [
    ("arm", "arm: pick_place, panda_pg2", tile_arm),
    ("dual", "dual: handover (parked)", tile_dual),
    ("quad", "quadruped: go2", tile_quad),
    ("humanoid", "humanoid: t1, h_steps", tile_humanoid),
    ("cworld", "computerworld: fill_form", tile_cworld),
]


def fig3_decorate(img: np.ndarray, label: str, tag: str):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img).convert("RGB")
    if im.size != (F3_W, F3_H):
        im = im.resize((F3_W, F3_H), Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")
    big, small = ImageFont.truetype(F3_FONT, 24), ImageFont.truetype(F3_FONT, 19)
    lines, col = F3_TAGS[tag]
    bw = max(d.textlength(t, font=big if i == 0 else small) for i, t in enumerate(lines)) + 20
    bh = 12 + 27 * len(lines)
    d.rectangle([0, 0, bw, bh], fill=col + (235,))
    for i, t in enumerate(lines):
        d.text((8, 5 + 27 * i), t, font=big if i == 0 else small, fill=(255, 255, 255))
    lw = d.textlength(label, font=small) + 20
    d.rectangle([0, F3_H - 30, lw, F3_H], fill=(255, 255, 255, 225))
    d.text((8, F3_H - 27), label, font=small, fill=(40, 40, 40))
    d.rectangle([0, 0, F3_W - 1, F3_H - 1], outline=col + (255,), width=4)
    return im


def fig3_render_tiles(out_dir: str | Path) -> Path:
    """Stage 1 (needs mujoco + rrp, no PIL): raw tile arrays -> tiles.npz (arrays t_<key>, tag strings g_<key>)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for key, label, fn in F3_TILES:
        img, tag = fn()
        data[f"t_{key}"], data[f"g_{key}"] = img, np.array(tag)
        print("tile", key, tag, img.shape, flush=True)
    p = out / "tiles.npz"
    np.savez_compressed(p, **data)
    return p


def fig3_compose(out_dir: str | Path, cols: int = 5) -> Path:
    """Stage 2 (needs PIL): tiles.npz -> fig3_envs.png / fig3_envs.pdf (a strip of `cols` columns)."""
    from PIL import Image
    out = Path(out_dir)
    z = np.load(out / "tiles.npz")
    tiles = [fig3_decorate(z[f"t_{k}"], label, str(z[f"g_{k}"])) for k, label, _ in F3_TILES]
    rows = -(-len(tiles) // cols)
    gap = 8
    sheet = Image.new("RGB", (cols * F3_W + (cols - 1) * gap, rows * F3_H + (rows - 1) * gap), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * (F3_W + gap), (i // cols) * (F3_H + gap)))
    sheet.save(out / "fig3_envs.png")
    pdf = out / "fig3_envs.pdf"
    sheet.save(pdf, "PDF", resolution=sheet.width / 3.4)     # 3.4 in: one column
    return pdf


def fig3_strip(out_dir: str | Path, cols: int = 5) -> Path:
    """Both stages in one process when PIL is importable (the repo venv has none: run `--render-only` there and
    `--compose-only` with a Python that has Pillow)."""
    fig3_render_tiles(out_dir)
    return fig3_compose(out_dir, cols)


def fig3():
    """Render the environment tiles (needs mujoco + rrp and Pillow) into docs/paper/figures/fig3_envs.{png,pdf}."""
    import shutil
    import tempfile
    if FIG_DEPS_ERROR is not None:
        raise SystemExit(f"fig3 needs mujoco on PYTHONPATH (and Pillow): {FIG_DEPS_ERROR}")
    with tempfile.TemporaryDirectory() as td:
        fig3_strip(td)
        (OUT / "figures").mkdir(exist_ok=True)
        for n in ("fig3_envs.png", "fig3_envs.pdf"):
            shutil.copy(Path(td) / n, OUT / "figures" / n)
    print("fig3: wrote", OUT / "figures" / "fig3_envs.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["fig2", "fig3", "table_relations", "table_envs", "all"])
    a = ap.parse_args()
    todo = ["table_relations", "table_envs", "fig2", "fig3"] if a.what == "all" else [a.what]
    for t in todo:
        if t == "table_relations":
            c = relations_table(OUT / "table_relations.tex")
            print("table_relations:", c["totals"])
        elif t == "table_envs":
            (OUT / "table_envs.tex").write_text(build_envs(REPO))
            print("table_envs: wrote", OUT / "table_envs.tex")
        elif t == "fig2":
            fig2()
            print("fig2: wrote", OUT / "figures" / "fig2.pdf")
        elif t == "fig3":
            fig3()


if __name__ == "__main__":
    main()
