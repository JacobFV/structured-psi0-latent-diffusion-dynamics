"""Relation-factor tile grid for the rrp report (appendix figure pages after Table 6).

One annotated tile per implemented relation factor (600x450 scene in the Figure-2 panel style, saved as
`figures/tiles/<factor>.pdf`), three per row, grouped by family and ordered like Table 6. Under each tile a LaTeX caption
generated from the tile record (fig_tiles.tex): factor name + term-source glyphs, the description (what the term encodes,
operator and form, source), the equation in the body's math font, and the ILLUSTRATIVE reason where applicable.
Entry point: `make_figures.py tiles [build [factor ...] | grid | all]`.

Every value drawn is computed by repository code on a real simulator / environment state: the arm and dual-arm
featurizers + collate (`policies.features`, `nets.batch`), the legged relation graph (`nets.legged_latent`), the G1
command-dim graph (`bodies.g1_simple`), the ComputerWorld UI edges, relgen labels (`harness.data.relgen`, simulator
truth, training-only), the task-pipeline packet labels, all pushed through the real factor operators
(`FactorSite.contributions` / `.bias`). Learned coefficients (edge weights w, PaPE a/b, rel_rot B, align b, same-code
gain g, task gate) are SET BY HAND and stated on the tile; nothing is trained attention or a trained probe output.
Probe-sourced terms show the simulator-truth label that trains the probe (badge "gt ... training-only"); the deployed
term is the probe's estimate. Tiles that cannot be computed from shipped code/data are marked ILLUSTRATIVE
(edge.support_of: figure-only task-graph variant; probe.psi0.grasp_pt / grasp_face: label not computable from the
recorded replays).

Families: Figure 2's four colours (geometry cyan, kinematics/UI graphite, interaction/locomotion amber, task violet)
plus structure green. Probe readouts are coloured by WHAT THEY ENCODE: geometric quantities (positions,
displacements, visibility, gaze) cyan; physical interaction (contact, holding, hand distance, lift, grasp, fall) amber;
task / procedure state (focus, subtask, goal, phase, target slot, base command) violet.

Scenes: scripted (privileged) teacher states (PickPlaceTeacher, dual Handover / SupportInsert teachers, legged
scripted teacher + RL tracker), reset states, a G1 keyframe, ComputerWorld desktops, recorded Psi0 teleop replays; the
scene chip on every tile names its source. Heavy deps: MuJoCo (EGL), the ComputerWorld site package (PYTHONPATH), the
Psi0 replay store + its isolated decoder venv (env RRP_PSI_*). Run light CPU under the ops broker.
Tile cache / standalone review renders: $RRP_TILES_WORK (default ~/work/rrp-data/paper/tiles/integrated), never the repo.
"""
from __future__ import annotations

import copy
import json
import math
import os
import pickle
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import make_figures as MF                                    # noqa: E402  (Figure-2 renderer: Cam, Panel, draw_panel, colours)
import matplotlib                                            # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import mujoco                                                # noqa: E402
import torch                                                 # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle   # noqa: E402

REPO = MF.REPO
FIG_DIR = MF.OUT / "figures"
WORK = Path(os.environ.get("RRP_TILES_WORK", Path.home() / "work" / "rrp-data" / "paper" / "tiles" / "integrated"))
LOG = os.environ.get("RRP_TILES_LOG")                        # optional progress log (one line per tile / page)

W, H = MF.F2_W, MF.F2_H                                      # 600 x 450 scene (px; 3.0 x 2.25 in at 200 dpi)
TILE_IN = 3.0                                                # native tile width (in); builders' marks are sized for it
SEED = MF.F2_SEED
IMG_H = H                                                    # BTile scene height (whole 450 px scene)

GEOM, GRAPH, AMBER, VIOLET = MF.GEOM, MF.GRAPH, MF.AMBER, MF.VIOLET
STRUCT = "#4E9A5B"                                           # structure family (no Figure-2 panel; new hue)
FAMILY = {"structure": STRUCT, "geometry": GEOM, "kinematics": GRAPH, "interaction": AMBER, "task": VIOLET,
          "locomotion": AMBER, "ui": GRAPH}                  # probe readouts: colour of what they encode (docstring)
INK = "#1D1F22"
QRING = QCOL = "#E4572E"                                     # query-token ring (Figure 2 convention)
FAINT = "#9AA0A8"
AMBER_TXT = "#B87414"                                        # amber is too light for text / thin lines
MONO = MF.MONO
BADGE = {   # term-source badges (the paper's honesty vocabulary)
    "public": "public (given)",
    "gt": "gt label: sim truth, training-only",
    "gt_probe": "gt label (sim truth, training-only); deployed = estimated-probe",
    "public_label": "label from public task state",
    "illustrative": "illustrative",
}
ARM_TAG = "SCRIPTED TEACHER (privileged)"
DUAL_PAIR = "panda_pg2__ur5e_pg2"
DUAL_SEED = 0
PSI_RUNS = Path(os.environ.get("RRP_PSI_RUNS", Path.home() / "work" / "ext" / "runs" / "psi1z"))
PSI_DATA = Path(os.environ.get("RRP_PSI_DATA", Path.home() / "work" / "ext" / "psi_home" / "data" / "simple"))
PSI_PY = Path(os.environ.get("RRP_PSI_PY", Path.home() / "work" / "ext" / "venvs" / "psi" / "bin" / "python"))


def family_of(name):
    return MF.group_of(name)


def log(msg):
    print(msg, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(msg.rstrip() + "\n")


def ink(c):
    return AMBER_TXT if c == AMBER else c


def caption_of(name, what, opform, badge):
    return f"{name}: {what.rstrip('. ')}. {opform.rstrip('. ')}. Source: {badge}."


# ================================================================================================ tile containers
BUILT = {}          # factor -> dict(kind, tile, caption, status, meta)


class Tile(MF.Panel):
    """A Figure-2 panel (MF.Panel marks: ring / dot / text / line / rect, cbar, scene chip, badge) plus extra mark
    kinds drawn by `draw_extra`: chip, arrow (curved), strip (a row of token cells), grid (knot x assembly cells)."""

    def __init__(self, factor, formula, badge, tag, img, color=None):
        super().__init__(0, factor, factor.replace(".", "_"), color or FAMILY[family_of(factor)], "", formula, badge, tag, img)
        self.extra = []


class BTile:
    """A free-layout tile: the render is cropped (`crop`, render px) and placed at `dest` = (x0, y0, scale) in tile px;
    marks are in tile px (task-token columns, curved edge arrows, boxes)."""

    def __init__(self, factor, family, title, what, form, badge, tag, status, img=None, crop=None, dest=None):
        self.factor, self.family, self.color = factor, family, FAMILY[family]
        self.title, self.what, self.form, self.badge, self.tag = title, what, form, badge, tag
        self.status = status                     # "computed" | "computed (label)" | "illustrative"
        self.img = img
        self.crop = crop or (0, 0, W, H)
        self.dest = dest or (0, 0, 1.0)
        self.marks = []
        self.meta = {}
        self.note = None

    def T(self, uv):                             # render px -> tile px
        uv = np.asarray(uv, float)
        x0, y0, s = self.dest
        return np.array([(uv[0] - self.crop[0]) * s + x0, (uv[1] - self.crop[1]) * s + y0])

    def img_rect(self):
        x0, y0, s = self.dest
        return x0, y0, x0 + (self.crop[2] - self.crop[0]) * s, y0 + (self.crop[3] - self.crop[1]) * s

    def ring(self, p, r=14):
        self.marks.append(dict(k="ring", p=list(map(float, p)), r=r))

    def dot(self, p, face="#FFFFFF", r=5, edge=INK, z=6):
        self.marks.append(dict(k="dot", p=list(map(float, p)), r=r, face=face, edge=edge, z=z))

    def text(self, p, s, color=INK, size=6.0, ha="center", va="center", bg=True, weight="normal", mono=True, z=9):
        self.marks.append(dict(k="text", p=list(map(float, p)), s=s, color=color, size=size, ha=ha, va=va, bg=bg,
                               weight=weight, mono=mono, z=z))

    def arrow(self, a, b, color=VIOLET, lw=1.3, rad=0.0, style="-|>", ls="-", z=5, shrinkA=6, shrinkB=8, alpha=1.0):
        self.marks.append(dict(k="arrow", a=list(map(float, a)), b=list(map(float, b)), color=color, lw=lw, rad=rad,
                               style=style, ls=ls, z=z, sA=shrinkA, sB=shrinkB, alpha=alpha))

    def box(self, x0, y0, x1, y1, label, fc="#FFFFFF", ec=VIOLET, lw=1.0, tc=INK, size=5.6, ls="-", z=4, bold=False):
        self.marks.append(dict(k="box", r=[float(x0), float(y0), float(x1), float(y1)], s=label, fc=fc, ec=ec, lw=lw,
                               tc=tc, size=size, ls=ls, z=z, bold=bold))

    def panel(self, x0, y0, x1, y1, fc=(1, 1, 1, 0.9), ec="#C9CCD1"):
        self.marks.append(dict(k="panel", r=[x0, y0, x1, y1], fc=fc, ec=ec))


def save_tile(P, caption, status, meta, records=None):
    """Register a built Figure-2-style tile (drawn later by `draw_tile`, standalone and on the grid pages)."""
    meta = dict(meta)
    if "status" in meta:
        meta["episode_status"] = meta.pop("status")
    BUILT[P.factor] = dict(kind="panel", tile=P, caption=caption, status=status, meta=meta)
    log(f"[int] tile {P.factor} built ({status})")


def emit_btile(t):
    form = " ".join(t.form.split())
    badge = " ".join(t.badge[1].split())
    BUILT[t.factor] = dict(kind="btile", tile=t, caption=caption_of(t.factor, t.what, form, badge),
                           status=t.status, meta=dict(t.meta, badge=badge, scene=t.tag))
    log(f"[int] tile {t.factor} built ({t.status})")


# ================================================================================================ extra marks
def chip(P, xy, s, fc="#FFFFFF", ec=INK, tc=INK, size=1.0, lw=0.6, alpha=0.92, z=8):
    P.extra.append(dict(kind="chip", x=float(xy[0]), y=float(xy[1]), s=s, fc=fc, ec=ec, tc=tc, size=size, lw=lw,
                        alpha=alpha, z=z))


def arrow(P, a, b, color=INK, lw=1.4, rad=0.0, alpha=1.0, head=True, shrink=6, z=5, ls="-"):
    P.extra.append(dict(kind="arrow", a=[float(a[0]), float(a[1])], b=[float(b[0]), float(b[1])], color=color, lw=lw,
                        rad=rad, alpha=alpha, head=head, shrink=shrink, z=z, ls=ls))


def line_arrow(P, a, b, color=INK, lw=1.6):
    MF.line(P, [a, b], color=color, lw=lw, arrow=True)


def rect(P, box, color, lw=1.6, fill=False, alpha=0.35, z=4):
    x0, y0, x1, y1 = box
    if fill:
        P.marks.append(dict(kind="rect", x0=x0, y0=y0, x1=x1, y1=y1, color=color, lw=lw, fill=True, fc=color, alpha=alpha, z=z - 1))
    P.marks.append(dict(kind="rect", x0=x0, y0=y0, x1=x1, y1=y1, color=color, lw=lw, z=z))


def strip(P, values, groups, rect, cmap, vmin, vmax, query=None, title="", fmt=None, nan_color="#E6E6E3"):
    """A row of token cells (values; NaN = not a key / masked) inside `rect` = (x0, y0, x1, y1) image px; `groups` =
    [(label, start, stop)] brackets under the row; `query` = index outlined in the query colour."""
    P.extra.append(dict(kind="strip", values=[float(v) for v in values], groups=groups, rect=rect, cmap=cmap, vmin=vmin,
                        vmax=vmax, query=query, title=title, fmt=fmt, nan_color=nan_color))


def grid(P, M, rect, row_labels, col_labels, cmap, vmin, vmax, title="", hi_cols=(), fmt=None, cell_text=None):
    P.extra.append(dict(kind="grid", M=np.asarray(M, float).tolist(), rect=rect, rows=row_labels, cols=col_labels,
                        cmap=cmap, vmin=vmin, vmax=vmax, title=title, hi_cols=list(hi_cols), fmt=fmt,
                        cell_text=cell_text))


def draw_extra(ax, P, fs=7.0):
    for mk in getattr(P, "extra", []):
        k = mk["kind"]
        if k == "chip":
            ax.text(mk["x"], mk["y"], mk["s"], color=mk["tc"], fontsize=fs * 0.78 * mk["size"], ha="center", va="center",
                    family=MF.MONO, zorder=mk["z"],
                    bbox=dict(boxstyle="round,pad=0.22", fc=mk["fc"], ec=mk["ec"], lw=mk["lw"], alpha=mk["alpha"]))
        elif k == "arrow":
            st = "-|>" if mk["head"] else "-"
            ax.add_patch(FancyArrowPatch(mk["a"], mk["b"], arrowstyle=st, mutation_scale=fs * 1.25, lw=mk["lw"] * fs * 0.16,
                                         color=mk["color"], alpha=mk["alpha"], zorder=mk["z"], linestyle=mk["ls"],
                                         connectionstyle=f"arc3,rad={mk['rad']}", shrinkA=mk["shrink"], shrinkB=mk["shrink"]))
        elif k == "strip":
            x0, y0, x1, y1 = mk["rect"]
            v = np.array(mk["values"])
            n = len(v)
            cw = (x1 - x0) / n
            ax.add_patch(Rectangle((x0 - 3, y0 - (14 if mk["title"] else 3)), x1 - x0 + 6, y1 - y0 + (14 if mk["title"] else 3) + 15,
                                   fc=(1, 1, 1, 0.86), ec="none", zorder=6))
            if mk["title"]:
                ax.text(x0, y0 - 3, mk["title"], fontsize=fs * 0.62, ha="left", va="bottom", family=MF.MONO, color=INK, zorder=7)
            for i, val in enumerate(v):
                if np.isnan(val):
                    fc = mk["nan_color"]
                else:
                    t = (val - mk["vmin"]) / max(mk["vmax"] - mk["vmin"], 1e-9)
                    fc = mk["cmap"](float(np.clip(t, 0, 1)))
                ax.add_patch(Rectangle((x0 + i * cw, y0), cw, y1 - y0, fc=fc, ec="white", lw=0.35, zorder=7))
                if mk["fmt"] and not np.isnan(val):
                    ax.text(x0 + (i + 0.5) * cw, (y0 + y1) / 2, mk["fmt"].format(val), fontsize=fs * 0.48, ha="center",
                            va="center", color=INK, zorder=8)
            if mk["query"] is not None:
                q = mk["query"]
                ax.add_patch(Rectangle((x0 + q * cw, y0 - 1.5), cw, y1 - y0 + 3, fill=False, ec=QRING, lw=fs * 0.17, zorder=9))
            for lab, a, b in mk["groups"]:
                xa, xb = x0 + a * cw + 1, x0 + b * cw - 1
                ax.plot([xa, xb], [y1 + 3, y1 + 3], color=INK, lw=0.5, zorder=8)
                ax.text((xa + xb) / 2, y1 + 4, lab, fontsize=fs * 0.55, ha="center", va="top", family=MF.MONO, color=INK, zorder=8)
        elif k == "grid":
            x0, y0, x1, y1 = mk["rect"]
            M = np.array(mk["M"])
            nr, nc = M.shape
            cw, ch = (x1 - x0) / nc, (y1 - y0) / nr
            ax.add_patch(Rectangle((x0 - 46, y0 - 26), x1 - x0 + 52, y1 - y0 + 46, fc=(1, 1, 1, 0.88), ec="none", zorder=6))
            if mk["title"]:
                ax.text(x0 - 42, y0 - 22, mk["title"], fontsize=fs * 0.62, ha="left", va="top", family=MF.MONO, color=INK, zorder=7)
            for r in range(nr):
                ax.text(x0 - 3, y0 + (r + 0.5) * ch, mk["rows"][r], fontsize=fs * 0.55, ha="right", va="center", family=MF.MONO, zorder=8)
                for c in range(nc):
                    val = M[r, c]
                    if np.isnan(val):
                        fc = "#E6E6E3"
                    else:
                        t = (val - mk["vmin"]) / max(mk["vmax"] - mk["vmin"], 1e-9)
                        fc = mk["cmap"](float(np.clip(t, 0, 1)))
                    ax.add_patch(Rectangle((x0 + c * cw, y0 + r * ch), cw, ch, fc=fc, ec="white", lw=0.5, zorder=7))
                    if mk["cell_text"] is not None:
                        ax.text(x0 + (c + 0.5) * cw, y0 + (r + 0.5) * ch, mk["cell_text"][r][c], fontsize=fs * 0.5,
                                ha="center", va="center", color="white" if (not np.isnan(val) and val > 0.5) else INK, zorder=8)
            for c in range(nc):
                ax.text(x0 + (c + 0.5) * cw, y1 + 3, mk["cols"][c], fontsize=fs * 0.55, ha="center", va="top", family=MF.MONO,
                        zorder=8, rotation=0)
            for c in mk["hi_cols"]:
                ax.add_patch(Rectangle((x0 + c * cw, y0), cw, y1 - y0, fill=False, ec=QRING, lw=fs * 0.15, zorder=9))



def draw_btile_scene(ax, t, fs=7.0, frame_lw=1.5):
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#FFFFFF")
    if t.img is not None:
        cx0, cy0, cx1, cy1 = t.crop
        x0, y0, x1, y1 = t.img_rect()
        ax.imshow(t.img[cy0:cy1, cx0:cx1], extent=(x0, x1, y1, y0), interpolation="bilinear", zorder=0)
    for mk in t.marks:
        k = mk["k"]
        if k == "panel":
            a, b, c, d = mk["r"]
            ax.add_patch(Rectangle((a, b), c - a, d - b, fc=mk["fc"], ec=mk["ec"], lw=0.5, zorder=1))
        elif k == "ring":
            x, y = mk["p"]
            ax.add_patch(plt.Circle((x, y), mk["r"], fill=False, ec="white", lw=3.2, zorder=7))
            ax.add_patch(plt.Circle((x, y), mk["r"], fill=False, ec=QCOL, lw=1.7, zorder=8))
        elif k == "dot":
            x, y = mk["p"]
            ax.add_patch(plt.Circle((x, y), mk["r"], fc=mk["face"], ec=mk["edge"], lw=0.7, zorder=mk["z"]))
        elif k == "text":
            kw = dict(bbox=dict(boxstyle="round,pad=0.15", fc=(1, 1, 1, 0.85), ec="none")) if mk["bg"] else {}
            ax.text(*mk["p"], mk["s"], color=mk["color"], fontsize=mk["size"], ha=mk["ha"], va=mk["va"],
                    family=MONO if mk["mono"] else "DejaVu Sans", weight=mk["weight"], zorder=mk["z"], **kw)
        elif k == "arrow":
            ax.add_patch(FancyArrowPatch(mk["a"], mk["b"], arrowstyle=mk["style"], mutation_scale=7, lw=mk["lw"],
                                         color=mk["color"], ls=mk["ls"], zorder=mk["z"], alpha=mk["alpha"],
                                         connectionstyle=f"arc3,rad={mk['rad']}", shrinkA=mk["sA"], shrinkB=mk["sB"]))
        elif k == "box":
            a, b, c, d = mk["r"]
            ax.add_patch(FancyBboxPatch((a, b), c - a, d - b, boxstyle="round,pad=0,rounding_size=4", fc=mk["fc"],
                                        ec=mk["ec"], lw=mk["lw"], ls=mk["ls"], zorder=mk["z"]))
            ax.text((a + c) / 2, (b + d) / 2, mk["s"], ha="center", va="center", fontsize=mk["size"], color=mk["tc"],
                    family=MONO, zorder=mk["z"] + 0.5, weight="bold" if mk["bold"] else "normal")
    MF.draw_tag_badge(ax, t.tag, " ".join(t.badge[1].split()), t.color, fs)
    for sp in ax.spines.values():
        sp.set_edgecolor(t.color)
        sp.set_linewidth(frame_lw)


# ================================================================================================ tile drawing
# Tiles are drawn at NATIVE scale (TILE_IN-wide scenes; the marks of every builder are sized for that) and LaTeX scales
# each tile image to \tilew (three per row), so a native size x prints at x * PRINT_SCALE. The image holds the scene,
# its marks and the scene-source chip only; the factor name, term-source glyphs, description, equation and the
# ILLUSTRATIVE note are set by LaTeX under the image (write_tex), generated from the tile records below.
PRINT_W_IN = 8.5 - 2 * 0.57                                  # \textwidth of rrp_report.tex (letter, margin=0.57in)
GRID_COLS = 3
GRID_GAP_IN = 0.16                                           # printed gap between tiles (\tilegap)
TILE_PRINT_IN = (PRINT_W_IN - (GRID_COLS - 1) * GRID_GAP_IN) / GRID_COLS   # ~2.35 in (\tilew)
PRINT_SCALE = TILE_PRINT_IN / TILE_IN                        # native -> printed (~0.78)
TILE_H_IN = TILE_IN * H / W                                  # native scene height
RASTER_DPI = 300                                             # target printed raster resolution (capped at native px)
MIN_PT = 4.5                                                 # printed floor for in-scene labels (raised where collision-free)
TAG_PT = 5.0                                                 # printed scene-source chip
NAME_PT, DESC_PT, EQ_PT = 8.0, 7.0, 7.5                      # printed caption sizes (LaTeX, under each tile)
DESC_MAX_LINES = 4
_PX_PER_PT = W / (TILE_IN * 72.0)                            # scene px per native pt


def npt(pt):
    """Printed points -> native points."""
    return pt / PRINT_SCALE


# term-source classes: glyph + colour (distinct from the family frame colours)
SRC_STYLE = {"public": ("●", "#2B2F36"),                # ● public (given): deployable as shown
             "probe": ("◆", "#1F5FAD"),                 # ◆ deployed value is the estimated-probe output
             "gt": ("▲", "#B3261E")}                    # ▲ simulator-truth label, training-only
SRC_TEX = {"public": r"\tilesrcpub", "gt": r"\tilesrcgt", "probe": r"\tilesrcprobe"}

# equations of the free-layout tiles (interaction / task families; Figure-2 tiles carry theirs in Panel.formula)
FORMULA_B = {
    "ix.contact": r"$\mathrm{contact}(i,j)$",
    "ix.force_flow": r"$\mathrm{flow}^{+}(\mathrm{support})(i,j)$",
    "ix.handover": r"$\mathrm{handover}(i,j)$",
    "ix.held_by": r"$\mathrm{held\_by}(i,j)$",
    "ix.support": r"$\mathrm{support}(i,j)$",
    "task.next_contact": r"$g_{\rm task}\,\mathrm{next}(i,j)$",
    "edge.actor_of": r"$\mathbb{1}[\,\{i,j\}\in E_{\mathrm{actor\_of}}\,]$",
    "edge.consumed_by": r"$\mathbb{1}[\,\{i,j\}\in E_{\mathrm{consumed\_by}}\,]$",
    "edge.destination_of": r"$\mathbb{1}[\,\{i,j\}\in E_{\mathrm{destination\_of}}\,]$",
    "edge.enables": r"$\mathbb{1}[\,j\in\mathrm{requires\_completed}(i)\,]$",
    "edge.maintained": r"$\mathbb{1}[\,j\in\mathrm{requires\_active}(i)\,]$",
    "edge.node_actor_of": r"$\mathbb{1}[\,\mathrm{asm}(i)\in\mathrm{actors}(j)\,]$",
    "edge.output_to": r"$\mathbb{1}[\,\{i,j\}\in E_{\mathrm{output\_to}}\,]$",
    "edge.patient_of": r"$\mathbb{1}[\,\{i,j\}\in E_{\mathrm{patient\_of}}\,]$",
    "edge.pred_arg": r"$\mathbb{1}[\,\{i,j\}\in E_{\mathrm{pred\_arg}}\,]$",
    "edge.produced": r"$\mathbb{1}[\,\{i,j\}\in E_{\mathrm{produced}}\,]$",
    "edge.role_in_event": r"$\mathbb{1}[\,\{i,j\}\in E_{\mathrm{role\_in\_event}}\,]$",
    "edge.role_points_to": r"$\mathbb{1}[\,j=\mathrm{bound}(i)\,]$",
    "edge.support_of": r"$\mathbb{1}[\,\{i,j\}\in E_{\mathrm{support\_of}}\,]$",
    "edge.target_of": r"$\mathbb{1}[\,\{i,j\}\in E_{\mathrm{target\_of}}\,]$",
}
# ILLUSTRATIVE tiles: the reason, set as the last caption line (and removed from the description text)
ILLUS = {
    "edge.support_of": "no shipped arm or dual task graph binds a support role to an entity; align:support is rebound "
                       "to 'left' for this figure only.",
    "probe.psi0.grasp_face": "scene real; the label is not computable here (no contact positions in any recorded "
                             "replay; needs SIMPLE / Isaac Sim).",
    "probe.psi0.grasp_pt": "scene real; the label is not computable here (no contact positions in any recorded "
                           "replay; needs SIMPLE / Isaac Sim).",
}
DESC = {    # condensed descriptions (plain text), only where the full registered text exceeds DESC_MAX_LINES
    "ix.force_flow": "transitive closure of the support graph: all the query carries, directly or via others. Bottom "
                     "cube → middle 1, top 1 (direct edge only to the middle). flow (closure) × bias over support-v1. "
                     "Source: shown: closure of gt support (training-only); deployed: closure of the ix.support "
                     "est-probe.",
    "ix.support": "a supports b: contact + normal within 30° of gravity-up at a's top. Query middle cube → top 1 "
                  "(row); the bottom cube supports it (grey arrow, column). Emits the estimated support-v1 graph. "
                  "bilinear × aug (probe, emits). Source: shown: gt label (training-only); deployed: est-probe.",
    "task.next_contact": "manipulator → graspable score, sharpened by the task gate (g_task = 1.00 at init, learned). "
                         "Dashed: public candidate edges (1/|E|); solid: selected (gt reveal posterior given contact). "
                         "bilinear × aug, gated. Deployed: est-probe × g_task.",
}
_ILLUS_STRIP = [r"\s*ILLUSTRATIVE:.*?\(figure-only\)\.", r"\s*Illustrative:[^.]*\.?\s*$"]


_TAG_SUBS = [("SIMULATOR STATE", "SIM STATE"), ("RECORDED TELEOP REPLAY", "TELEOP REPLAY"), ("(privileged),", "(priv.)"),
             ("(privileged)", "(priv.)"), ("SCRIPTED TEACHER + RL TRACKER", "SCRIPTED + RL TRACKER"),
             ("SCRIPTED TEACHER + CPG", "SCRIPTED + CPG")]


def short_tag(tag):
    for a, b in _TAG_SUBS:
        tag = tag.replace(a, b)
    return " ".join(tag.split())


def badge_text(rec):
    t = rec["tile"]
    return " ".join((t.badge if rec["kind"] == "panel" else t.badge[1]).split())


def badge_chips(rec):
    """[(class, label)] term-source chips of a tile, from its badge string (glyph + colour per class)."""
    f, b = rec["tile"].factor, badge_text(rec).lower()
    if f == "task.next_contact":
        return [("public", "candidates"), ("gt", "selected: gt train-only"), ("probe", "est-probe")]
    if b == "illustrative":                                  # psi0 grasp labels: gt sim truth, training-only (default OFF)
        return [("gt", "gt train-only (OFF)")]
    out = []
    if b.startswith(("public", "given", "label from public", "illustrative: public")):
        out.append(("public", "public label" if b.startswith("label") else "public"))
    if "gt" in b.split() or b.startswith("gt") or "of gt" in b:
        out.append(("gt", "gt train-only"))
    if "probe" in b:
        out.append(("probe", "est-probe" if out and out[-1][0] == "gt" else "default est-probe"))
    assert out, f"unclassified badge {b!r} ({f})"
    return out


def _ext(t, r):
    """Text extent incl. a margin ~ its bbox padding (the bbox patch itself is only positioned at draw time)."""
    bb = t.get_window_extent(r)
    return bb.padded(0.12 * t.get_fontsize() * t.figure.dpi / 72.0)


def _overlaps(boxes):
    s = set()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 0.5 and min(a.y1, b.y1) - max(a.y0, b.y0) > 0.5:
                s.add((i, j))
    return s


def raise_scene_text(ax, fixed, skip):
    """Raise in-scene labels (not the numeric cell values of grids / strips, where colour carries the value) to MIN_PT
    printed, then shrink only the labels that now collide with another label or chip (in 0.2 pt steps, never below
    their original size) and nudge labels that would leave the scene back inside. Returns the per-label printed sizes."""
    r = ax.figure.canvas.get_renderer()
    cell_max = 7.0 * 0.5 + 1e-6
    scene = [t for t in ax.texts if t not in fixed and t not in skip and t.get_text().strip()]
    cand = [t for t in scene if t.get_fontsize() > cell_max]
    orig = {t: t.get_fontsize() for t in cand}
    allt = scene + list(fixed)
    abox = ax.get_window_extent(r)
    d2px = W / abox.width                                       # display px -> scene px

    def inside(b):
        return b.x0 >= abox.x0 - 1 and b.x1 <= abox.x1 + 1 and b.y0 >= abox.y0 - 1 and b.y1 <= abox.y1 + 1

    b0 = [_ext(t, r) for t in allt]
    base, was_in = _overlaps(b0), [inside(b) for b in b0]
    tangled = {allt[k] for pr in base for k in pr}             # labels that already overlap at native size stay as built
    size = {t: (orig[t] if t in tangled else max(orig[t], npt(MIN_PT))) for t in cand}
    step = npt(0.2)
    for _ in range(60):
        for t in cand:
            t.set_fontsize(size[t])
        boxes = [_ext(t, r) for t in allt]
        for i, (t, bb) in enumerate(zip(allt, boxes)):          # nudge back inside (scene px; y axis points down)
            if t in orig and was_in[i] and not inside(bb):
                dx = max(0.0, abox.x0 - bb.x0) - max(0.0, bb.x1 - abox.x1)
                dy = max(0.0, abox.y0 - bb.y0) - max(0.0, bb.y1 - abox.y1)
                if t.get_transform() == ax.transData:
                    x, y = t.get_position()
                    t.set_position((x + dx * d2px, y - dy * d2px))
                else:
                    size[t] = max(orig[t], size[t] - step)
        boxes = [_ext(t, r) for t in allt]
        bad = {k for pr in (_overlaps(boxes) - base) for k in pr}
        bad |= {i for i, bb in enumerate(boxes) if was_in[i] and not inside(bb)}
        shrink = [allt[i] for i in bad if allt[i] in orig and size[allt[i]] > orig[allt[i]] + 1e-9]
        if not shrink:
            break
        for t in shrink:
            size[t] = max(orig[t], size[t] - step)
    return [t.get_fontsize() * PRINT_SCALE for t in cand]


def formula_of(rec):
    t = rec["tile"]
    f = t.formula if rec["kind"] == "panel" else FORMULA_B.get(t.factor, "")
    assert f, f"no equation for {t.factor}"
    return f


def desc_parts(rec):
    """(description, source) of a tile from its registered caption 'name: what. operator and form. Source: src.'"""
    import re
    f, cap = rec["tile"].factor, " ".join(rec["caption"].split())
    pre = f + ": "
    cap = cap[len(pre):] if cap.startswith(pre) else cap
    k = cap.rfind(" Source: ")
    body, src = (cap[:k], cap[k + len(" Source: "):]) if k >= 0 else (cap, "")
    for pat in _ILLUS_STRIP:
        body, src = re.sub(pat, "", body).strip(), re.sub(pat, "", src).strip()
    return body.rstrip(". ") + ".", src.rstrip(". ")


def draw_scene_chip(ax, tag):
    return ax.text(6, 6, tag, color="white", fontsize=npt(TAG_PT), ha="left", va="top", family=MONO, zorder=12,
                   bbox=dict(boxstyle="square,pad=0.2", fc=(0.11, 0.12, 0.13, 0.85), ec="none"))


def draw_tile(fig, rec):
    """The tile scene into the whole figure (TILE_IN x TILE_H_IN native): scene, marks, scene-source chip and the
    ILLUSTRATIVE watermark. Returns (scene_ax, stats)."""
    t = rec["tile"]
    ax = fig.add_axes([0, 0, 1, 1])
    _orig_badge = MF.draw_tag_badge
    MF.draw_tag_badge = lambda *a, **k: None                  # the scene chip is drawn compactly below; the term source
    try:                                                      # is in the LaTeX caption
        if rec["kind"] == "panel":
            MF.draw_panel(ax, t, fs=7.0, frame_lw=1.5 / PRINT_SCALE * 0.45)
            draw_extra(ax, t, fs=7.0)
        else:
            draw_btile_scene(ax, t, fs=7.0, frame_lw=1.5 / PRINT_SCALE * 0.45)
    finally:
        MF.draw_tag_badge = _orig_badge
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fixed = [draw_scene_chip(ax, short_tag(t.tag))]
    skip = []
    if rec["status"] == "illustrative":
        skip.append(ax.text(W / 2, H / 2, "ILLUSTRATIVE", color=(0.75, 0.1, 0.1, 0.35), fontsize=npt(12.0), ha="center",
                            va="center", rotation=20, zorder=11, weight="bold"))
    lab_pt = raise_scene_text(ax, fixed, skip)
    sizes = [tt.get_fontsize() * PRINT_SCALE for tt in ax.texts if tt.get_text().strip() and tt not in skip]
    stats = dict(labels_ge_min=sum(v >= MIN_PT - 1e-6 for v in lab_pt), labels=len(lab_pt),
                 min_scene_text_pt=round(min(sizes), 2) if sizes else None, n_scene_text=len(sizes),
                 n_below_min=sum(v < MIN_PT - 1e-6 for v in sizes), badges=badge_chips(rec))
    return ax, stats


def render_tile(rec, path, dpi=None):
    """One tile image (scene only) at native size; PDF rasters at RASTER_DPI printed (capped at the native scene px)."""
    with plt.rc_context(MF.F2_MATH_RC):
        fig = plt.figure(figsize=(TILE_IN, TILE_H_IN), dpi=200)
        fig.patch.set_facecolor("white")
        _, stats = draw_tile(fig, rec)
        fig.savefig(path, dpi=dpi or min(RASTER_DPI * PRINT_SCALE, W / TILE_IN))
        plt.close(fig)
    return stats


def render_standalone(factor, out_dir, rec=None):
    """One tile at native size as a review PNG."""
    rec = rec or BUILT[factor]
    out_dir.mkdir(parents=True, exist_ok=True)
    return dict(factor=factor, **render_tile(rec, out_dir / f"{factor}.png", dpi=200))


# ##############################################################################################################
# GROUP A renderers: structure, geometry, kinematics/body (arm featurizer, G1 command-dim graph, go2 legged graph)
# ##############################################################################################################

# ------------------------------------------------------------------------------------------------ rendering
def render_scene(m, d, cam, color_fn, min_agree=0.94):
    """make_figures.render_scene, except that a pixel the ray cast misses but the rasterizer's segmentation covers keeps
    its rendered colour and segmentation body (MuJoCo's mesh ray test misses parts of some visual meshes, e.g. the
    Panda forearm, which the Figure-2 close-ups never showed). World points `pts` stay ray-cast only (`hit`)."""
    MF.style_model(m, color_fn)
    mujoco.mj_forward(m, d)
    rgb, sg = MF.render_view(m, d, cam)
    view = MF.View(cam)
    pts, gid, hit = MF.ray_world_points(m, d, view)
    is_geom = sg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    both = hit & is_geom
    agree = float((sg[..., 0][both] == gid[both]).mean())
    assert agree > min_agree, f"ray/segmentation agreement {agree:.3f}"
    rgb = rgb.copy()
    covered = hit | is_geom
    rgb[~covered] = MF.BG
    g_any = np.where(hit, gid, np.where(is_geom, sg[..., 0], -1))
    body = np.where(g_any >= 0, m.geom_bodyid[np.maximum(g_any, 0)], -1)
    return dict(rgb=rgb, view=view, pts=pts, gid=gid, hit=hit, body=body, agree=agree, m=m)


# ------------------------------------------------------------------------------------------------ colour maps
def seq(color):
    return MF.ramp(color)


def div(color):
    return MF.diverging(color, "#9AA0A8")


def binary(color):
    return matplotlib.colors.LinearSegmentedColormap.from_list("bin", ["#F2F2EF", color])


# ================================================================================================ relational helpers
def _rel():
    from rrp.policies.relations import base as RB
    from rrp.policies.relations import ops as RO
    RB._ensure_catalog()
    return RB, RO


def site_of(names, site, carries, heads=1, dim=8):
    RB, RO = _rel()
    specs = RB.resolve(names) if not isinstance(names, tuple) else names
    fs = RO.FactorSite(heads, dim, site, specs, carries)
    return fs, specs


def set_edge_weights(fs, value=1.0):
    with torch.no_grad():
        if fs.w is not None:
            fs.w.fill_(value)
        for k, m in fs.f.items():
            if hasattr(m, "w"):
                m.w.fill_(value)


def set_coeff(mod, bias):
    """A zero-weight `_Coeff` with a hand-set bias: the per-head coefficient is the constant `bias` for every token."""
    with torch.no_grad():
        mod.weight.zero_()
        mod.bias.copy_(torch.as_tensor(np.asarray(bias, np.float32).reshape(-1)))


# ================================================================================================ scene: arm (Panda)
ARM_WIDE = MF.Cam(lookat=[0.30, -0.05, 0.30], dist=1.72, az=125, el=-22, fovy=34.0)


def arm_tokens(s):
    """Real arm ctx/act token sets of the session's current observation: featurizer -> collate -> fields.
    Returns dict(pi, ents, batch, names, kinds, world (drawable world point or None per ctx token), act_names, ...)."""
    from rrp.policies.features.featurizer import cached_featurizer, REL, BANKS
    from rrp.harness.pipelines.relations import arm_featurize
    from rrp.policies.nets.batch import collate_inputs
    feat = cached_featurizer(s)
    obs = s.observe()
    pi, ents = arm_featurize(feat, s.state_view(), obs)
    b = collate_inputs([pi])
    inv = {v: k for k, v in REL.items()}
    R = np.asarray(pi.relations).reshape(-1, 5)
    offs = b.bank_offset
    m, d = s.model, s.data
    names, world, bank_of = [], [], []
    # morph: action nodes, passive joints, assemblies
    for jn in feat.node_joint_names:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        names.append(jn.replace("r0_", "").replace("joint", "j").replace("wrist_", ""))
        world.append(d.xanchor[j].copy())
        bank_of.append("morph")
    for pj in feat.passive:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, pj.name)
        names.append(pj.name.replace("r0_", "").replace("wrist_", "") + "*")
        world.append(d.xanchor[j].copy())
        bank_of.append("morph")
    for a in feat.spec.assemblies:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, a.frame.site)
        names.append(f"asm:{a.id}")
        world.append(d.site_xpos[sid].copy())
        bank_of.append("morph")
    for od in obs.object_descriptors:
        names.append(od.descriptor)
        world.append(None if od.position_estimate is None else np.asarray(od.position_estimate, float))
        bank_of.append("scene")
    # task bank names from the real relations (events, roles, predicates)
    nt = len(pi.tokens["task"])
    tk = pi.token_kind["task"]
    ti = obs.task_input
    ev_names = [e.operator for e in ti.definition.events]
    tnames = [None] * nt
    ev_tok = [i for i in range(nt) if tk[i] == 0]
    for k, i in enumerate(ev_tok):
        tnames[i] = f"ev:{ev_names[k]}"
    TB, MB, SB = BANKS.index("task"), BANKS.index("morph"), BANKS.index("scene")
    short = {}
    for i in range(len(names)):
        short[i] = names[i]
    for i in range(nt):
        if tk[i] == 1:
            ev = [r[3] for r in R if r[0] == TB and r[1] == i and r[2] == TB and inv[r[4]] == "role_in_event"]
            tgt = [(r[2], r[3]) for r in R if r[0] == TB and r[1] == i and inv[r[4]] == "role_points_to"]
            role = "?"
            if ev and tgt:
                for r in R:
                    if r[0] == tgt[0][0] and r[1] == tgt[0][1] and r[2] == TB and r[3] == ev[0] and inv[r[4]] in (
                            "actor_of", "patient_of", "destination_of", "support_of", "target_of"):
                        role = inv[r[4]].replace("_of", "")
            tnames[i] = f"{ev_names[ev_tok.index(ev[0])] if ev else '?'}.{role}"
        elif tk[i] == 2:
            k = sum(1 for j in range(i) if tk[j] == 2)
            pe = obs.predicate_estimates[k] if k < len(obs.predicate_estimates) else None
            tnames[i] = f"p:{pe.predicate}" if pe is not None else "pred"
    for i in range(nt):
        names.append(tnames[i] or "task")
        world.append(None)
        bank_of.append("task")
    ik = pi.token_kind["interact"]
    for i in range(len(pi.tokens["interact"])):
        names.append({0: "receipt", 1: "sensor", 2: "null"}[int(ik[i])])
        world.append(None)
        bank_of.append("interact")
    ch = [c for c in obs.declared_sensor_channels if c.name.startswith("0:")]
    si = [i for i, bk in enumerate(bank_of) if bk == "interact" and names[i] == "sensor"]
    for i, c in zip(si, ch):
        names[i] = "touch" if c.kind == "touch" else "grip_w"
    act_names = [names[i] for i in range(len(feat.node_joint_names))]
    return dict(pi=pi, ents=ents, batch=b, names=names, world=world, bank=bank_of, feat=feat, obs=obs, R=R, inv=inv,
                offs=offs, act_names=act_names, n_act=len(feat.node_joint_names))


def arm_render(s, cam, objs_only_color=None):
    objs = {o.sim_body for o in s.scenario.objects}
    return render_scene(s.model, s.data, cam, objs_only_color or MF.arm_color_fn(objs))


def arm_rc(tok, deploy=False):
    """RelCtx of the arm family exactly as `nets.batch.relation_token_sets` builds it for the flow net (ctx / act sets with
    the derived public fields) + the arm-rel-v1 EdgeSets at ctx>ctx / act>ctx / act>act."""
    RB, _ = _rel()
    from rrp.policies.nets.batch import ctx_geometry_fields, ctx_id_fields, act_assembly_id
    from rrp.policies.relations.catalog import ARM_REL_VOCAB
    b = tok["batch"]
    f = dict(ctx_geometry_fields(b))
    f.update(ctx_id_fields(b))
    ctx = RB.TokenSet("ctx", b.ctx_mask, fields=f)
    act = RB.TokenSet("act", b.node_mask, fields=act_assembly_id(b))
    edges = {"ctx>ctx": RB.EdgeSet(ARM_REL_VOCAB, b.ctx_rel), "act>ctx": RB.EdgeSet(ARM_REL_VOCAB, b.act_rel),
             "act>act": RB.EdgeSet(ARM_REL_VOCAB, b.node_rel)}
    return RB.RelCtx(sets={"ctx": ctx, "act": act}, edges=edges, deploy=deploy)


def chip_row(n, y, x0=20, x1=W - 20):
    if n == 1:
        return [((x0 + x1) / 2, y)]
    return [(x0 + (x1 - x0) * k / (n - 1), y) for k in range(n)]


# ================================================================================================ scene: G1 keyframe
G1_CAM = MF.Cam(lookat=[0.0, 0.02, 0.97], dist=1.45, az=200, el=-6, fovy=34.0)


def g1_scene(cam=G1_CAM, color=None):
    from rrp.bodies import g1_simple as G
    from rrp.envs.mujoco import humanoid_scenes as HS
    m, sp, meta = HS.task_model("g1", "h_steps", {"h_frac": 0.02})
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    dim_body, dim_joint = {}, {}
    for i, (nm, jn, *_r) in enumerate(G._DIMS):
        if jn is None:
            continue
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "r0_" + jn)
        if j >= 0:
            dim_body[i] = int(m.jnt_bodyid[j])
            dim_joint[i] = j
    base = color or (lambda bn, t, g: None if bn == "world" else np.array([0.70, 0.72, 0.75, 1.0]))
    sc = render_scene(m, d, cam, base)
    return dict(G=G, m=m, d=d, sc=sc, dim_body=dim_body, dim_joint=dim_joint)


def g1_body_assembly(g):
    """{body id: G1 command-space assembly name} for the rendered bodies (arm links by their dims; hands = links below
    the wrist yaw link (none in this model: the wrist-yaw link carries the hand); torso = waist/torso links; base = pelvis
    and legs). Used only to colour bodies by assembly."""
    G, m = g["G"], g["m"]
    out = {}
    for i, b in g["dim_body"].items():
        out[b] = G.ASSEMBLIES[G.DIM_ASM[i]]
    for b in range(1, m.nbody):
        if b in out:
            continue
        n = m.body(b).name
        out[b] = "torso" if ("torso" in n or "waist" in n) else "base"
    return out


def g1_paint(g, values, cmap, vmin, vmax, alpha=0.9):
    """values: {dim index: value} painted on the dim's link body."""
    bv = {g["dim_body"][i]: v for i, v in values.items() if i in g["dim_body"]}
    return MF.paint_bodies(g["sc"], bv, cmap, vmin, vmax, alpha)


def g1_dim_uv(g, i):
    return MF.uv_of(g["sc"], g["d"].xanchor[g["dim_joint"][i]])[0]


def g1_strip(P, g, row, cmap, vmin, vmax, q, y=386, title="all 36 command-dim tokens (key row of the query)"):
    G = g["G"]
    order = []
    groups = []
    for a in G.ASSEMBLIES:
        idx = [i for i in range(len(G.DIM_NAMES)) if G.ASSEMBLIES[G.DIM_ASM[i]] == a]
        groups.append((a.replace("left_", "L").replace("right_", "R"), len(order), len(order) + len(idx)))
        order += idx
    vals = [row[i] for i in order]
    strip(P, vals, groups, (24, y, W - 24, y + 16), cmap, vmin, vmax, query=order.index(q), title=title)
    return order


# ================================================================================================ scene: go2 (legged)
GO2_CAM = MF.Cam(lookat=[0.12, 0.0, 0.12], dist=1.15, az=145, el=-28, fovy=34.0)
GO2_TOP = MF.Cam(lookat=[-0.06, 0.0, 0.14], dist=1.30, az=160, el=-62, fovy=34.0)
GO2_SCAN = MF.Cam(lookat=[0.24, 0.0, 0.06], dist=1.60, az=160, el=-60, fovy=34.0)


def go2_scene(cam=GO2_CAM, K=None):
    """go2 on mujoco/legged foothold_steps at reset (no controller), its public morphology (LeggedMorph -> static batch),
    the public terrain scan (TerrainScan sensor model: noise + dropout, as the declared channel `0:terrain_scan`) and the
    legged relation graph `legged_graph` (ctx = [glob | joints | limbs | feet | cells])."""
    from rrp.envs.base import make_env
    from rrp.policies.features.legged import LeggedMorph
    from rrp.policies.legged import static_batch
    from rrp.policies.nets import legged_latent as LL
    from rrp.envs.mujoco.legged_core import TerrainScan, scan_world_xy, yaw_of
    env = make_env("mujoco/legged", task="foothold_steps", body="go2", seed=SEED)
    m, d = env.model, env.data
    morph = LeggedMorph(m, env.binding, env.scenario.robots[0].robot_spec.spec_hash)
    b = static_batch(morph, "cpu")
    ts = TerrainScan(env.binding)
    ts.reset(d, SEED)
    exact = ts.exact(d)
    b["terrain"] = torch.as_tensor(ts.values, dtype=torch.float32)[None]
    b["terrain_valid"] = torch.as_tensor(ts.valid)[None]
    K = K or len(LL.KNOT_TIMES)
    tok, rc = LL.legged_graph(b, K, deploy=True)
    q = d.qpos[env.binding.qa:env.binding.qa + 7]
    wx, wy = scan_world_xy(float(q[0]), float(q[1]), yaw_of(q[3:7]))
    gz = float(q[2]) - (ts.h0 - exact)              # ground height under each cell from the exact (noise-free) reading
    cells = np.stack([wx, wy, gz], -1)
    floor_col = np.array([0.86, 0.86, 0.85, 1.0])

    def color(bn, t, g):
        if t == mujoco.mjtGeom.mjGEOM_PLANE:
            return floor_col
        if bn.startswith("foothold"):
            return np.array([0.80, 0.80, 0.79, 1.0])
        if bn == "world":
            return np.array([0.84, 0.84, 0.83, 1.0])
        return np.array([0.66, 0.68, 0.71, 1.0])
    sc = render_scene(m, d, cam, color)
    acts = list(env.binding.pol_act) + list(env.binding.held_act)
    jids = [int(m.actuator_trnid[a, 0]) for a in acts]
    jnames = [m.joint(j).name.replace("r0_", "").replace("_joint", "") for j in jids]
    feet = [d.site_xpos[sid].copy() for sid in env.binding.foot_sids] if len(getattr(env.binding, "foot_sids", [])) else \
        [d.xpos[fb].copy() for fb in env.binding.foot_bids]
    asm_names = [n.replace("r0_", "") for n in ("FL", "FR", "RL", "RR")][:morph.nf] + ["body"]
    hips = [d.xanchor[jids[list(morph.node_asm).index(a)]].copy() for a in range(morph.nf)]
    return dict(env=env, m=m, d=d, morph=morph, b=b, tok=tok, rc=rc, K=K, sc=sc, cells=cells, scan=ts, jids=jids,
                jnames=jnames, feet=feet, hips=hips, asm_names=asm_names, LL=LL)


def go2_bodies_of_asm(g, a):
    """Rendered link bodies of leg assembly a (the bodies of its joints) or the trunk for the body assembly."""
    m, morph = g["m"], g["morph"]
    if a >= morph.nf:
        return [m.body("r0_base").id]
    return sorted({int(m.jnt_bodyid[g["jids"][k]]) for k in range(morph.N) if morph.node_asm[k] == a})


def go2_ctx_index(g, part, i):
    """Flat ctx index of the i-th token of a part ('glob', 'joint', 'limb', 'foot', 'cell')."""
    sl = g["tok"].sl(part)
    return range(*sl.indices(g["tok"].T))[i]


# ================================================================================================ explore (camera tuning)


# ================================================================================================ tiles
TILES = {}


def tile(name):
    def deco(fn):
        TILES[name] = fn
        return fn
    return deco


# ------------------------------------------------------------------------------------------------ arm drawing helpers
ARM_TICK = 29                     # Figure 2's approach state (PickPlaceTeacher, phase "descend"/approach)
SIDE_X = 528                      # right-hand column of non-spatial (task / interact) token chips
ACT_X = 52                        # left-hand column of action-node (act set) query chips


def arm_setup(tick=ARM_TICK, cam=None, setup=None):
    s, T = MF.arm_at_tick(tick, setup=setup)
    tok = arm_tokens(s)
    sc = arm_render(s, cam or ARM_WIDE)
    uv = [None if w is None else MF.uv_of(sc, w)[0] for w in tok["world"]]
    return s, T, tok, sc, uv


def side_chips(tok, y0=96, y1=338):
    """uv of every non-spatial ctx token (task, interact) as a chip in the right-hand column."""
    idx = [i for i, w in enumerate(tok["world"]) if w is None]
    pos = chip_row(len(idx), 0) if len(idx) == 1 else [(SIDE_X, y0 + (y1 - y0) * k / max(len(idx) - 1, 1)) for k in range(len(idx))]
    return {i: np.array(p, float) for i, p in zip(idx, pos)}


def act_chips(tok, y0=110, y1=330):
    n = tok["n_act"]
    return {i: np.array((ACT_X, y0 + (y1 - y0) * i / max(n - 1, 1)), float) for i in range(n)}


def draw_ctx_dots(P, tok, uv, values=None, cmap=None, vmin=0.0, vmax=1.0, skip=(), r=4.2):
    """Small dots for every spatial ctx token; coloured by `values` when given (NaN/None -> white)."""
    for i, p in enumerate(uv):
        if p is None or i in skip or not MF.in_frame(p, 3):
            continue
        face = "#FFFFFF"
        if values is not None and values[i] is not None and not np.isnan(values[i]) and abs(values[i]) > 1e-9:
            face = matplotlib.colors.to_hex(cmap(float(np.clip((values[i] - vmin) / max(vmax - vmin, 1e-9), 0, 1))))
        MF.dot(P, p, face, r=r)


def arm_meta(s, T, tick, extra=None):
    m = dict(env="mujoco/arm", body="panda_pg2", task="pick_place(n_distractors=2)", seed=SEED, tick=tick, phase=T.phase,
             scene_source="scripted privileged teacher (PickPlaceTeacher) state, replayed from reset (Figure 2 scene)")
    m.update(extra or {})
    return m




# ================================================================================================ structure tiles
@tile("edge.same_node")
def tile_edge_same_node():
    s, T, tok, sc, uv = arm_setup()
    rc = arm_rc(tok)
    fs, _ = site_of(["edge.same_node"], "act>ctx", ("edges:arm-rel-v1",))
    set_edge_weights(fs)
    V = fs.contributions(rc)["edge.same_node"][0, 0].numpy()            # [N act nodes, C ctx tokens]
    q = 3                                                                # act node of joint 4
    P = Tile("edge.same_node", r"$\mathbb{1}[\,j=\mathrm{morph}(i)\,]$", BADGE["public"], ARM_TAG, sc["rgb"])
    draw_ctx_dots(P, tok, uv, values=list(V[q]), cmap=binary(STRUCT))
    ac = act_chips(tok)
    for i, p in ac.items():
        chip(P, p, "act:" + tok["act_names"][i], fc="#FFFFFF" if i != q else "#FCE9E3", ec=QRING if i == q else INK, size=0.95)
        for j in np.nonzero(V[i] > 0)[0]:
            if uv[j] is not None:
                arrow(P, p + np.array([34, 0]), uv[j], color=STRUCT if i == q else FAINT, lw=1.5 if i == q else 0.7,
                      alpha=1.0 if i == q else 0.8, rad=-0.15)
    kq = int(np.nonzero(V[q] > 0)[0][0])
    MF.ring(P, uv[kq], color=STRUCT, r=12)
    MF.text(P, uv[kq] + np.array([0, -22]), f"{tok['names'][kq]}: {V[q, kq]:.0f}", size=1.0)
    cap = caption_of("edge.same_node", "an action-node query attends to its OWN morphology token (one per actuated joint)",
                     "edge x bias, w*A[i,j] at act>ctx (arm-rel-v1, w=1 shown)", "public morphology (featurizer)")
    save_tile(P, cap, "computed", arm_meta(s, T, ARM_TICK, dict(
        query=f"act node {tok['act_names'][q]}", site="act>ctx", row={tok["names"][j]: float(V[q, j]) for j in range(V.shape[1])},
        n_edges=int((V > 0).sum()), code_path="arm_featurize -> collate_inputs -> act_rel; FactorSite(['edge.same_node'], act>ctx).contributions, w=1")))


@tile("edge.node_in_assembly")
def tile_edge_node_in_assembly():
    s, T, tok, sc, uv = arm_setup()
    rc = arm_rc(tok)
    fs, _ = site_of(["edge.node_in_assembly"], "act>ctx", ("edges:arm-rel-v1",))
    set_edge_weights(fs)
    V = fs.contributions(rc)["edge.node_in_assembly"][0, 0].numpy()
    fc_, _ = site_of(["edge.node_in_assembly"], "ctx>ctx", ("edges:arm-rel-v1",))
    set_edge_weights(fc_)
    Vc = fc_.contributions(rc)["edge.node_in_assembly"][0, 0].numpy()   # ctx>ctx: sensor tokens <-> gripper assembly
    q = 7                                                                # the gripper slide node
    P = Tile("edge.node_in_assembly", r"$\mathbb{1}[\,j=\mathrm{asm}(i)\,]$", BADGE["public"], ARM_TAG, sc["rgb"])
    draw_ctx_dots(P, tok, uv, values=list(V[q]), cmap=binary(STRUCT))
    ac = act_chips(tok)
    for i, p in ac.items():
        chip(P, p, "act:" + tok["act_names"][i], fc="#FCE9E3" if i == q else "#FFFFFF", ec=QRING if i == q else INK, size=0.95)
        for j in np.nonzero(V[i] > 0)[0]:
            arrow(P, p + np.array([34, 0]), uv[j], color=STRUCT if i == q else FAINT, lw=1.6 if i == q else 0.6,
                  alpha=1.0 if i == q else 0.75, rad=-0.12)
    sch = side_chips(tok)
    names = tok["names"]
    for i, p in sch.items():
        on = any(Vc[i, j] > 0 for j in range(Vc.shape[1]) if names[j].startswith("asm:"))
        chip(P, p, names[i], fc="#FFFFFF", ec=STRUCT if on else FAINT, tc=INK if on else FAINT, size=0.85)
        for j in range(Vc.shape[1]):
            if Vc[i, j] > 0 and names[j].startswith("asm:"):
                arrow(P, p - np.array([40, 0]), uv[j], color=STRUCT, lw=0.8, alpha=0.7, rad=0.2, ls="--")
    for j, n in enumerate(names):
        if n.startswith("asm:") and uv[j] is not None:
            MF.ring(P, uv[j], color=STRUCT, r=10)
            MF.text(P, uv[j] + np.array([46, -14 if n == "asm:arm" else 14]), n, size=0.95)
    cap = caption_of("edge.node_in_assembly", "a joint / sensor token attends to the assembly (arm, gripper) it belongs to",
                     "edge x bias at act>ctx and ctx>ctx (solid: act nodes, dashed: touch / grip sensors; w=1)",
                     "public morphology (featurizer)")
    save_tile(P, cap, "computed", arm_meta(s, T, ARM_TICK, dict(
        query=f"act node {tok['act_names'][q]}", site="act>ctx (+ ctx>ctx sensor rows)",
        act_to_asm={tok["act_names"][i]: [names[j] for j in np.nonzero(V[i] > 0)[0]] for i in range(V.shape[0])},
        sensor_to_asm={names[i]: [names[j] for j in np.nonzero(Vc[i] > 0)[0]] for i in sch},
        code_path="FactorSite(['edge.node_in_assembly']) at act>ctx and ctx>ctx .contributions over the collated arm-rel-v1 edges, w=1")))


@tile("msg.incidence")
def tile_msg_incidence():
    s, T, tok, sc, uv = arm_setup()
    b = tok["batch"]
    ptr = b.pointers[0].numpy()
    ptr = ptr[ptr[:, 0] >= 0]
    names = tok["names"]
    sch = side_chips(tok)
    q = names.index("grasp.patient")
    P = Tile("msg.incidence", r"$h_i \leftarrow h_i + W\,h_{\mathrm{ptr}(i)}$", BADGE["public"], ARM_TAG, sc["rgb"])
    draw_ctx_dots(P, tok, uv)
    srcs = sorted(set(ptr[:, 0].tolist()))
    for i, p in sch.items():
        isq = i == q
        chip(P, p, names[i], fc="#FCE9E3" if isq else "#FFFFFF", ec=QRING if isq else (STRUCT if i in srcs else FAINT),
             tc=INK if (i in srcs) else FAINT, size=0.85)
    for src, dst in ptr:
        a = uv[dst] if uv[dst] is not None else sch[dst] - np.array([40, 0])
        bpt = sch[src] - np.array([42, 0])
        isq = src == q
        arrow(P, a, bpt, color=STRUCT if isq else "#7FAE88", lw=1.7 if isq else 0.8, alpha=1.0 if isq else 0.85,
              rad=0.18 if uv[dst] is not None else 0.5)
    for j in sorted(set(ptr[:, 1].tolist())):
        if uv[j] is not None:
            MF.ring(P, uv[j], color=STRUCT, r=10)
            lab = names[j].replace("green target zone", "target zone")
            off = {"target zone": (-62, -4), "asm:gripper": (-62, -26), "red cube": (62, 22)}.get(lab, (-58, 16))
            MF.text(P, uv[j] + np.array(off), lab, size=0.9)
    cap = caption_of("msg.incidence", "role and predicate tokens receive a message from the entity they point to "
                     "(arrows: entity -> role)", "message form, not a logit term; W learned; control `serialized` = same "
                     "facts as hashed text", "public task graph + tracker slots")
    save_tile(P, cap, "computed", arm_meta(s, T, ARM_TICK, dict(
        query="grasp.patient (role token)", pointers=[[names[a], names[c]] for a, c in ptr],
        note="incidence structure is the real featurizer output (PolicyInput.pointers); the message map W (FlowNet.ptr) is learned and not shown",
        code_path="arm_featurize -> collate_inputs -> Batch.pointers; FlowNet: h.scatter_add(src, ptr(h[dst]))")))


@tile("id.slot_handle")
def tile_id_slot_handle():
    s, T, tok, sc, uv = arm_setup(cam=MF.Cam(lookat=[0.43, -0.05, 0.03], dist=0.92, az=135, el=-46, fovy=34.0))
    obs = tok["obs"]
    names = tok["names"]
    P = Tile("id.slot_handle", r"$x_k \leftarrow x_k + E[\mathrm{slot}_k]$", BADGE["public"], ARM_TAG, sc["rgb"])
    so = tok["offs"]["scene"]
    cm = matplotlib.colormaps["tab10"]
    rows = []
    for k, od in enumerate(obs.object_descriptors):
        j = so + k
        if uv[j] is None:
            continue
        col = matplotlib.colors.to_hex(cm(k % 10))
        std = np.sqrt(np.asarray(od.position_cov_diag, float)) if od.position_cov_diag else np.zeros(3)
        rpx = max(9.0, float(np.linalg.norm(MF.uv_of(sc, np.asarray(od.position_estimate) + np.array([3 * std[0], 0, 0]))[0] - uv[j])))
        MF.dot(P, uv[j], "#FFFFFF", r=5.5, edge=INK)
        chip(P, uv[j] + np.array([0, -30]), f"slot {od.slot}", fc=STRUCT, ec=STRUCT, tc="white", size=1.05)
        MF.text(P, uv[j] + np.array([0, 24]), od.descriptor.replace("green target zone", "target zone"), size=0.9)
        rows.append(dict(slot=int(od.slot), descriptor=od.descriptor, position_estimate=list(map(float, od.position_estimate)),
                         visible=bool(od.visible), std=std.tolist(), radius_px=rpx))
    chip(P, (W / 2, 76), "learned embedding row E[slot] added to each scene token", fc="#FFFFFF", ec=STRUCT, size=0.85)
    cap = caption_of("id.slot_handle", "each scene token carries its public tracker slot address as a learned embedding, "
                     "so a slot keeps its identity across steps", "embed form (unary, no attention term); E learned",
                     "public tracker slot ids")
    save_tile(P, cap, "computed", arm_meta(s, T, ARM_TICK, dict(
        slots=rows, note="slot ids / descriptors / estimates are the real tracker output (PolicyObservation.object_descriptors); E is learned",
        code_path="FlowNet.forward: x_scene += slot_emb.weight[:T_scene] (id.slot_handle on)")))


@tile("id.same_body")
def tile_id_same_body():
    s, T, tok, sc, uv = arm_setup()
    rc = arm_rc(tok)
    fs, _ = site_of(["id.same_body"], "ctx>ctx", ("entity_id",))
    with torch.no_grad():
        fs.f["id__same_body"].g.fill_(1.0)
    C = tok["batch"].ctx_mask.shape[1]
    x = torch.zeros(1, C, 8)
    V = fs.contributions(rc, x, x)["id.same_body"][0, 0].numpy()        # [C, C] <e(id_i), e(id_j)>
    names = tok["names"]
    q = names.index("red cube")
    ent = rc.sets["ctx"].field("entity_id")[0, :, 0].long().tolist()
    cm = div(STRUCT)
    P = Tile("id.same_body", r"$\langle e(\mathrm{id}_i),e(\mathrm{id}_j)\rangle\approx\mathbb{1}[\mathrm{id}_i=\mathrm{id}_j]$",
             BADGE["public"], ARM_TAG, sc["rgb"])
    draw_ctx_dots(P, tok, uv, values=list(V[q]), cmap=cm, vmin=-1, vmax=1, skip=(q,))
    MF.ring(P, uv[q], r=13)
    MF.text(P, uv[q] + np.array([-92, 4]), "query: red cube", size=0.9)
    sch = {i: p - np.array([16, 0]) for i, p in side_chips(tok).items()}
    for i, p in sch.items():
        v = V[q, i]
        same = ent[i] == ent[q]
        fc = matplotlib.colors.to_hex(cm(float(np.clip((v + 1) / 2, 0, 1))))
        chip(P, p, f"{names[i]} {v:+.2f}", fc=fc, ec=STRUCT if same else FAINT, tc="white" if v > 0.6 else INK, size=0.76)
        if same:
            arrow(P, uv[q], p - np.array([48, 0]), color=STRUCT, lw=1.4, rad=-0.2)
    for j in range(C):
        if uv[j] is not None and j != q and names[j] in ("blue cube", "yellow cube", "asm:gripper"):
            MF.text(P, uv[j] + np.array([0, 16]), f"{names[j].replace(' cube', '').replace('asm:', '')} {V[q, j]:+.2f}", size=0.85)
    cap = caption_of("id.same_body", "tokens denoting the SAME entity (a tracker slot and the task roles bound to it) "
                     "attend to each other", "same x aug: fixed random 16-d unit codes for 64 ids (1 on equal ids, small "
                     "cross-talk otherwise), g=1", "public ids (featurizer pointers)")
    save_tile(P, cap, "computed", arm_meta(s, T, ARM_TICK, dict(
        query="red cube (scene slot 0)", entity_id=dict(zip(names, ent)), row={names[j]: float(V[q, j]) for j in range(C)},
        code_path="nets.batch.ctx_id_fields -> TokenSet ctx.entity_id; FactorSite(['id.same_body'], ctx>ctx).contributions (aug), g=1")))


# ================================================================================================ G1 (Psi0 command dims)
G1_TAG = "SIMULATOR STATE (G1 keyframe)"


def g1_rc():
    RB, _ = _rel()
    from rrp.bodies import g1_simple as G
    n = G.ACTION_DIM
    rel = torch.from_numpy(G.relation_matrix()).permute(1, 2, 0).unsqueeze(0)          # [1,T,T,R] g1-dim-rel-v1
    return RB.RelCtx(sets={"dims": RB.TokenSet("dims", torch.ones(1, n, dtype=torch.bool))},
                     edges={"dims>dims": RB.EdgeSet(G.RELATIONS, rel)})


def g1_dims_value(name, rc=None, params=None):
    spec = [{"name": name, **({"params": params} if params else {})}]
    fs, _ = site_of(spec, "dims>dims", ("edges:g1-dim-rel-v1",))
    set_edge_weights(fs)
    from rrp.bodies import g1_simple as G
    n = G.ACTION_DIM
    return fs.contributions(rc or g1_rc(), torch.zeros(1, n, 8), torch.zeros(1, n, 8))[name][0, 0].numpy()


def g1_meta(extra=None):
    m = dict(env="mujoco/humanoid (rrp.envs.mujoco.humanoid_scenes.task_model('g1','h_steps'))",
             body="g1 (36 command-dim tokens, bodies.g1_simple, vocab g1-dim-rel-v1)", task="keyframe 0 (static pose, no policy)",
             seed=SEED, tick=0, scene_source="simulator keyframe (no controller, no motion)",
             not_drawn="finger and base dims have no link in the rendered model: they are tokens in the strip only")
    m.update(extra or {})
    return m


def g1_graph_tile(name, qname, formula, what, opform, edge_color=GRAPH, extra_lines=None, value_note=None):
    g = g1_scene()
    G = g["G"]
    V = g1_dims_value(name)
    q = G.DIM_NAMES.index(qname)
    row = V[q]
    color = FAMILY[family_of(name)]
    img = g1_paint(g, {i: row[i] for i in range(len(row)) if row[i] > 0}, binary(color), 0, 1, alpha=0.92)
    P = Tile(name, formula, BADGE["public"], G1_TAG, img)
    for i in g["dim_joint"]:
        p = g1_dim_uv(g, i)
        MF.dot(P, p, matplotlib.colors.to_hex(color) if row[i] > 0 else "#FFFFFF", r=4.2)
    if extra_lines:
        for a, b_ in extra_lines(V):
            if a in g["dim_joint"] and b_ in g["dim_joint"]:
                MF.line(P, [g1_dim_uv(g, a), g1_dim_uv(g, b_)], color=FAINT, lw=0.8)
    pq = g1_dim_uv(g, q) if q in g["dim_joint"] else None
    if pq is not None:
        MF.ring(P, pq, r=12)
        MF.text(P, pq + np.array([-62 if pq[0] > W / 2 else 62, -16]), f"query {qname}", size=0.9)
        for j in np.nonzero(row > 0)[0]:
            if j in g["dim_joint"]:
                arrow(P, pq, g1_dim_uv(g, j), color=color, lw=1.5, rad=0.25, shrink=7)
                pj = g1_dim_uv(g, j)
                if (row > 0).sum() <= 3:
                    MF.text(P, pj + np.array([0, 15]), G.DIM_NAMES[j], size=0.78)
    g1_strip(P, g, row, binary(color), 0, 1, q)
    cap = caption_of(name, what, opform, "public morphology (bodies.g1_simple)")
    keys = [G.DIM_NAMES[j] for j in np.nonzero(row > 0)[0]]
    save_tile(P, cap, "computed", g1_meta(dict(
        query=qname, site="dims>dims", keys=keys, n_edges_total=int((V > 0).sum()), note=value_note,
        code_path=f"FactorSite(['{name}'], dims>dims).contributions over RelCtx edges = bodies.g1_simple.relation_matrix(); weight 1")))


@tile("edge.same_assembly")
def tile_edge_same_assembly():
    g1_graph_tile("edge.same_assembly", "l_elbow", r"$\mathbb{1}[\,\mathrm{asm}(i)=\mathrm{asm}(j),\ i\neq j\,]$",
                  "a command dim attends to the other dims of its own assembly (here the left arm)",
                  "edge x bias at dims>dims (g1-dim-rel-v1 channel same_assembly; w=1)")


@tile("edge.kin_child")
def tile_edge_kin_child():
    g1_graph_tile("edge.kin_child", "waist_pitch", r"$\mathbb{1}[\,\mathrm{parent}(j)=i\,]$",
                  "a command dim attends to its kinematic children (waist pitch -> both shoulder pitches)",
                  "edge x bias at dims>dims (channel kin_child; w=1)")


@tile("edge.mirror")
def tile_edge_mirror():
    from rrp.bodies import g1_simple as G
    g1_graph_tile("edge.mirror", "l_elbow", r"$\mathbb{1}[\,j=\mathrm{mirror}(i)\,]$",
                  "a command dim attends to its left/right homologue (grey lines: every mirror pair of the body)",
                  "edge x bias at dims>dims (channel mirror; w=1)",
                  extra_lines=lambda V: [(i, j) for i in range(V.shape[0]) for j in np.nonzero(V[i] > 0)[0] if i < j])


@tile("kin.sibling")
def tile_kin_sibling():
    g1_graph_tile("kin.sibling", "l_shoulder_pitch", r"$\mathbb{1}[\,d_{\mathrm{tree}}(i,j)=2\,]$",
                  "dims at undirected tree distance 2: the sibling (r_shoulder_pitch) and, by the same op, grandparent "
                  "and grandchild", "hop x bias over the symmetrized kin_parent graph (hops=2; w=1)",
                  value_note="hop op lights grandparent / grandchild pairs at the same undirected distance (catalog doc)")


def g1_knot_rc(K, M):
    RB, _ = _rel()
    from rrp.bodies import g1_simple as G
    DA = G.ACTION_DIM
    knot_asm = torch.from_numpy(np.tile(np.arange(M, dtype=np.int64), K))
    return RB.RelCtx(sets={
        "dims": RB.TokenSet("dims", torch.ones(1, DA, dtype=torch.bool), fields={"assembly_id": torch.from_numpy(G.DIM_ASM)[None, :, None]}),
        "knots": RB.TokenSet("knots", torch.ones(1, K * M, dtype=torch.bool), fields={"assembly_id": knot_asm[None, :, None]})})


def g1_knot_tile(name, qname, spec, formula, what, opform, mask_form):
    from rrp.bodies import g1_simple as G
    from rrp.policies.psi0.nets import K
    M = len(G.ASSEMBLIES)
    fs, specs = site_of([spec], "dims>knots", ("assembly_id",))
    rc = g1_knot_rc(K, M)
    DA = G.ACTION_DIM
    if mask_form:
        Bv = fs.bias(rc)[0, 0].numpy()                                         # [DA, K*M] 0 / -inf
        V = np.isfinite(Bv).astype(float)
    else:
        with torch.no_grad():
            fs.f[name.replace(".", "__")].g.fill_(1.0)
        V = fs.contributions(rc, torch.zeros(1, DA, 8), torch.zeros(1, K * M, 8))[name][0, 0].numpy()
    q = G.DIM_NAMES.index(qname)
    row = V[q].reshape(K, M)                                                   # k-major knots
    color = FAMILY[family_of(name)]
    asm_val = {G.ASSEMBLIES[a]: float(row[0, a]) for a in range(M)}
    g = g1_scene()
    bas = g1_body_assembly(g)
    if mask_form:
        bv = {b: 1.0 for b, a in bas.items() if asm_val[a] > 0.5}
        img = MF.paint_bodies(g["sc"], bv, binary(color), 0, 1, 0.9)
    else:
        cm = div(color)
        bv = {b: asm_val[a] for b, a in bas.items()}
        img = MF.paint_bodies(g["sc"], bv, cm, -1, 1, 0.9)
    P = Tile(name, formula, BADGE["public"], G1_TAG, img)
    pq = g1_dim_uv(g, q)
    MF.ring(P, pq, r=12)
    MF.text(P, pq + np.array([62 if pq[0] < W / 2 else -62, -18]), f"query dim {qname}", size=0.9)
    labs = {"left_hand": "LH", "right_hand": "RH", "left_arm": "LA", "right_arm": "RA", "torso": "T", "base": "B"}
    cols = [labs[a] for a in G.ASSEMBLIES]
    if mask_form:
        txt = [["ok" if row[k, a] > 0.5 else "-inf" for a in range(M)] for k in range(K)]
        grid(P, row, (60, 240, 60 + 22 * M, 240 + 15 * K), [f"k{k}" for k in range(K)], cols, binary(color), 0, 1,
             title="packet knots (k x assembly)", cell_text=txt)
    else:
        txt = [[f"{row[k, a]:+.1f}" for a in range(M)] for k in range(K)]
        grid(P, row, (60, 240, 60 + 22 * M, 240 + 15 * K), [f"k{k}" for k in range(K)], cols, div(color), -1, 1,
             title="packet knots (k x assembly)", cell_text=txt)
    cap = caption_of(name, what, opform, "public morphology (bodies.g1_simple)")
    save_tile(P, cap, "computed", g1_meta(dict(
        query=qname, query_assembly=G.ASSEMBLIES[G.DIM_ASM[q]], site="dims>knots (Psi0 system-0 Realizer)", K=K,
        per_assembly=asm_val, n_allowed=int((V[q] > 0.5).sum()),
        code_path=f"FactorSite([{spec!r}], dims>knots).{'bias' if mask_form else 'contributions'} over dims/knots assembly_id "
                  "(exactly psi0.nets.Realizer.forward's RelCtx)")))


@tile("route.assembly_reads")
def tile_route_assembly_reads():
    from rrp.bodies import g1_simple as G
    g1_knot_tile("route.assembly_reads", "l_elbow", {"name": "route.assembly_reads", "params": {"reads": G.reads_table().tolist()}},
                 r"$0\ \mathrm{if}\ \mathrm{READS}[\mathrm{asm}(i),\mathrm{asm}(j)]\ \mathrm{else}\ -\infty$",
                 "system 0 (Psi0 Realizer): a command dim reads packet knots of its own assembly and its kinematic "
                 "neighbours only (left arm: arm, hand, torso)", "same x mask with params.reads = bodies.g1_simple.READS",
                 mask_form=True)


@tile("id.same_assembly")
def tile_id_same_assembly():
    g1_knot_tile("id.same_assembly", "l_elbow", {"name": "id.same_assembly"},
                 r"$\langle e(a_i),e(a_j)\rangle\approx\mathbb{1}[a_i=a_j]$",
                 "soft same-assembly term: a command dim prefers knots of its own assembly (left arm); other "
                 "assemblies stay readable", "same x aug, random 16-d codes for 64 ids (small cross-talk), g=1, "
                 "dims>knots", mask_form=False)


# ================================================================================================ go2 (legged)
GO2_TAG = "RESET STATE (no controller)"


def go2_meta(g, extra=None):
    m = dict(env="mujoco/legged", body="go2", task="foothold_steps", seed=SEED, tick=0,
             scene_source="reset state (no controller running); morphology = LeggedMorph (public), terrain = TerrainScan "
                          "sensor model (public declared channel 0:terrain_scan)",
             ctx_layout="[glob | joints | limbs | feet | terrain cells] (nets.legged_latent.legged_graph), padded to MAX_N / MAX_M")
    m.update(extra or {})
    return m


def go2_edge(g, name, site="ctx>ctx"):
    fs, _ = site_of([name], site, ("edges:legged-rel-v1",))
    set_edge_weights(fs)
    T = g["tok"].T
    return fs.contributions(g["rc"], torch.zeros(1, T, 8), torch.zeros(1, T, 8))[name][0, 0].numpy()


def go2_joint_uv(g, k):
    return MF.uv_of(g["sc"], g["d"].xanchor[g["jids"][k]])[0]


@tile("route.own_assembly")
def tile_route_own_assembly():
    RB, _ = _rel()
    g = go2_scene()
    b, K, morph = g["b"], g["K"], g["morph"]
    M = b["asm_mask"].shape[1]
    fs, _ = site_of([{"name": "route.own_assembly", "params": {"also_key_field": "body"}}], "act>knots", ("assembly_id",))
    knot_asm = torch.arange(M).repeat(K)[None]
    rc = RB.RelCtx(sets={
        "act": RB.TokenSet("act", b["node_mask"], fields={"assembly_id": b["node_asm"]}),
        "knots": RB.TokenSet("knots", b["asm_mask"].repeat(1, K), fields={"assembly_id": knot_asm, "body": knot_asm == b["body_asm"][:, None]})})
    Bv = fs.bias(rc)[0, 0].numpy()                                             # [N, K*M]
    q = g["jnames"].index("FL_calf")
    Mr = morph.M
    row = np.isfinite(Bv[q]).reshape(K, M)[:, :Mr].astype(float)
    allowed = [a for a in range(Mr) if row[0, a] > 0.5]
    bv = {bd: 1.0 for a in allowed for bd in go2_bodies_of_asm(g, a)}
    color = FAMILY["structure"]
    img = MF.paint_bodies(g["sc"], bv, binary(color), 0, 1, 0.9)
    P = Tile("route.own_assembly", r"$0\ \mathrm{if}\ a_j\in\{a_i,\mathrm{body}\}\ \mathrm{else}\ -\infty$", BADGE["public"], GO2_TAG, img)
    pq = go2_joint_uv(g, q)
    MF.ring(P, pq, r=12)
    MF.text(P, pq + np.array([0, 22]), "query node FL_calf", size=0.9)
    txt = [["ok" if row[k, a] > 0.5 else "-inf" for a in range(Mr)] for k in range(K)]
    grid(P, row, (440, 300, 440 + 26 * Mr, 300 + 15 * K), [f"k{k}" for k in range(K)], g["asm_names"], binary(color), 0, 1,
         title="packet knots (k x assembly)", cell_text=txt)
    cap = caption_of("route.own_assembly", "system 0 (legged Realizer): a joint node reads only the packet knots of its own "
                     "limb plus the body assembly", "same x mask on assembly_id, params.also_key_field = body (preset legged-s0)",
                     "public morphology (LeggedMorph)")
    save_tile(P, cap, "computed", go2_meta(g, dict(
        query="node FL_calf", site="act>knots", K=K, allowed=[g["asm_names"][a] for a in allowed],
        code_path="FactorSite([route.own_assembly also_key_field=body], act>knots).bias over the RelCtx of LeggedRealizer._route_bias")))


@tile("edge.kin_parent")
def tile_edge_kin_parent():
    g = go2_scene()
    V = go2_edge(g, "edge.kin_parent")
    J = range(*g["tok"].sl("joint").indices(g["tok"].T))
    jidx = list(J)[:g["morph"].N]
    q = g["jnames"].index("FL_calf")
    qi = jidx[q]
    P = Tile("edge.kin_parent", r"$\mathbb{1}[\,j=\mathrm{parent}(i)\,]$", BADGE["public"], GO2_TAG, g["sc"]["rgb"])
    for k, ci in enumerate(jidx):
        p = go2_joint_uv(g, k)
        for kk, cj in enumerate(jidx):
            if V[ci, cj] > 0:
                isq = k == q
                arrow(P, p, go2_joint_uv(g, kk), color=GRAPH if isq else FAINT, lw=1.8 if isq else 0.8, rad=0.0, shrink=5)
        MF.dot(P, p, matplotlib.colors.to_hex(GRAPH) if V[qi, ci] > 0 else "#FFFFFF", r=4.2)
    pq = go2_joint_uv(g, q)
    MF.ring(P, pq, r=11)
    MF.text(P, pq + np.array([-40, 20]), "query FL_calf", size=0.9)
    par = [g["jnames"][kk] for kk, cj in enumerate(jidx) if V[qi, cj] > 0]
    for kk, cj in enumerate(jidx):
        if V[qi, cj] > 0:
            MF.text(P, go2_joint_uv(g, kk) + np.array([-36, -14]), g["jnames"][kk], size=0.85)
    cap = caption_of("edge.kin_parent", "a joint token attends to its kinematic parent joint (grey: every parent edge of "
                     "the four legs)", "edge x bias at ctx>ctx (legged-rel-v1, from the per-assembly depth chain; w=1)",
                     "public morphology (LeggedMorph)")
    save_tile(P, cap, "computed", go2_meta(g, dict(query="joint FL_calf", parents=par, site="ctx>ctx",
                                               n_edges=int((V > 0).sum()),
                                               code_path="legged_graph -> EdgeSet ctx>ctx; FactorSite(['edge.kin_parent']).contributions, w=1")))


def go2_limb_uv(g, a):
    return MF.uv_of(g["sc"], g["hips"][a])[0]


def go2_foot_uv(g, a):
    return MF.uv_of(g["sc"], g["feet"][a])[0]


@tile("edge.limb_adjacent")
def tile_edge_limb_adjacent():
    g = go2_scene(GO2_TOP)
    V = go2_edge(g, "edge.limb_adjacent")
    L = list(range(*g["tok"].sl("limb").indices(g["tok"].T)))
    nf = g["morph"].nf
    q = 0
    P = Tile("edge.limb_adjacent", r"$\mathbb{1}[\,\mathrm{kind}(i)=\mathrm{kind}(j)\neq\mathrm{body},\ i\neq j\,]$", BADGE["public"], GO2_TAG, g["sc"]["rgb"])
    for a in range(nf):
        pa = go2_limb_uv(g, a)
        chip(P, pa + np.array([0, -22]), "limb:" + g["asm_names"][a], fc="#FCE9E3" if a == q else "#FFFFFF",
             ec=QRING if a == q else INK, size=0.85)
        MF.dot(P, pa, matplotlib.colors.to_hex(GRAPH) if V[L[q], L[a]] > 0 else "#FFFFFF", r=5)
    for a in range(nf):
        if V[L[q], L[a]] > 0:
            arrow(P, go2_limb_uv(g, q), go2_limb_uv(g, a), color=GRAPH, lw=1.7, rad=0.3, shrink=7)
    MF.ring(P, go2_limb_uv(g, q), r=11)
    body_row = float(V[L[q], L[nf]])
    MF.text(P, (W / 2, 400), f"body assembly token: {body_row:.0f} (not a limb)", size=0.9)
    cap = caption_of("edge.limb_adjacent", "a limb token attends to the other limbs of the same kind on the trunk (all "
                     "four legs; tokens drawn at their hips)", "edge x bias at ctx>ctx (legged-rel-v1, symmetric; w=1)",
                     "public morphology (LeggedMorph)")
    save_tile(P, cap, "computed", go2_meta(g, dict(query="limb FL", row={g["asm_names"][a]: float(V[L[q], L[a]]) for a in range(nf + 1)},
                                               drawn_at="limb tokens drawn at their hip joint anchor (their pos3d field is the default foot position)",
                                               code_path="legged_graph ctx>ctx; FactorSite(['edge.limb_adjacent']).contributions, w=1")))


@tile("edge.foot_of")
def tile_edge_foot_of():
    g = go2_scene()
    V = go2_edge(g, "edge.foot_of")
    T = g["tok"].T
    L = list(range(*g["tok"].sl("limb").indices(T)))
    Fo = list(range(*g["tok"].sl("foot").indices(T)))
    nf = g["morph"].nf
    q = 1
    P = Tile("edge.foot_of", r"$\mathbb{1}[\,j=\mathrm{foot}(i)\,]$", BADGE["public"], GO2_TAG, g["sc"]["rgb"])
    for a in range(nf):
        pl, pf = go2_limb_uv(g, a), go2_foot_uv(g, a)
        for bb in range(nf):
            if V[L[a], Fo[bb]] > 0:
                arrow(P, pl, go2_foot_uv(g, bb), color=GRAPH if a == q else FAINT, lw=1.8 if a == q else 0.9, rad=0.18, shrink=6)
        MF.dot(P, pl, "#FFFFFF", r=4.5)
        MF.dot(P, pf, matplotlib.colors.to_hex(GRAPH) if V[L[q], Fo[a]] > 0 else "#FFFFFF", r=4.5)
    MF.ring(P, go2_limb_uv(g, q), r=11)
    MF.text(P, go2_limb_uv(g, q) + np.array([0, -24]), "query limb:FR", size=0.9)
    MF.text(P, go2_foot_uv(g, q) + np.array([0, 18]), "foot:FR", size=0.9)
    cap = caption_of("edge.foot_of", "a limb token attends to its own foot token (feet exist only on foot-carrying "
                     "assemblies; limbs drawn at hips, feet at foot sites)", "edge x bias at ctx>ctx (legged-rel-v1; w=1)",
                     "public morphology (LeggedMorph)")
    save_tile(P, cap, "computed", go2_meta(g, dict(query="limb FR", n_edges=int((V > 0).sum()),
                                               code_path="legged_graph ctx>ctx; FactorSite(['edge.foot_of']).contributions, w=1")))


@tile("edge.over_cell")
def tile_edge_over_cell():
    g = go2_scene(GO2_SCAN)
    V = go2_edge(g, "edge.over_cell")
    T = g["tok"].T
    Fo = list(range(*g["tok"].sl("foot").indices(T)))
    Ce = list(range(*g["tok"].sl("cell").indices(T)))
    q = 0
    row = V[Fo[q], Ce]
    valid = g["scan"].valid
    P = Tile("edge.over_cell", r"$\mathbb{1}[\,\mathrm{cell}\ j\in\mathrm{window}(\mathrm{foot}\ i)\,]$", BADGE["public"], GO2_TAG, g["sc"]["rgb"])
    sc = g["sc"]
    for c, P3 in enumerate(g["cells"]):
        p = MF.uv_of(sc, P3)[0]
        if not MF.in_frame(p, 4):
            continue
        if row[c] > 0:
            MF.dot(P, p, matplotlib.colors.to_hex(GRAPH), r=4.6, edge="white")
        else:
            MF.dot(P, p, "#FFFFFF" if valid[c] else "#D9D9D6", r=2.8, edge="#6B7078")
    pf = go2_foot_uv(g, q)
    MF.ring(P, pf, r=11)
    MF.text(P, pf + np.array([-30, -22]), "query foot:FL", size=0.9)
    other = [int(V[Fo[a], Ce].sum()) for a in range(g["morph"].nf)]
    MF.text(P, (300, 412), "white dots: 11x7 public terrain-scan cells (grey: dropout)", size=0.85)
    cap = caption_of("edge.over_cell", "a foot token attends to the terrain-scan cells of its landing window (0.05 m behind "
                     "to 0.35 m ahead, +-0.15 m aside of its default stance, body frame)",
                     "edge x bias at ctx>ctx (legged-rel-v1; w=1)", "public morphology + public terrain scan")
    save_tile(P, cap, "computed", go2_meta(g, dict(query="foot FL", n_cells_in_window=int(row.sum()), cells_per_foot=other,
                                               scan_valid=int(valid.sum()), scan_cells=int(len(valid)),
                                               code_path="legged_graph (scan_xy, OVER_CELL_DX/DY, default stance foot_xy) ctx>ctx; FactorSite(['edge.over_cell']).contributions, w=1")))


# ================================================================================================ arm: kin.ancestor
@tile("kin.ancestor")
def tile_kin_ancestor():
    s, T, tok, sc, uv = arm_setup()
    rc = arm_rc(tok)
    fs, _ = site_of(["kin.ancestor"], "ctx>ctx", ("edges:arm-rel-v1",))
    set_edge_weights(fs)
    V = fs.contributions(rc)["kin.ancestor"][0, 0].numpy()
    names = tok["names"]
    q = names.index("j7")
    P = Tile("kin.ancestor", r"$\mathbb{1}[\,j\ \mathrm{ancestor\ of}\ i\,]$", BADGE["public"], ARM_TAG, sc["rgb"])
    A = rc.edges["ctx>ctx"].data[0, :, :, 16].numpy()                   # kin_parent channel (arm-rel-v1 index 16)
    for i in range(len(names)):
        for j in range(len(names)):
            if A[i, j] and uv[i] is not None and uv[j] is not None:
                arrow(P, uv[i], uv[j], color=FAINT, lw=0.8, shrink=5)
    draw_ctx_dots(P, tok, uv, values=list(V[q]), cmap=binary(GRAPH))
    MF.ring(P, uv[q], r=11)
    MF.text(P, uv[q] + np.array([60, 0]), "query j7", size=0.9)
    anc = [names[j] for j in np.nonzero(V[q] > 0)[0]]
    groups = {}
    for j in np.nonzero(V[q] > 0)[0]:
        key = tuple(np.round(uv[j]).astype(int))
        groups.setdefault(key, []).append(names[j])
    for key, ns in groups.items():
        MF.text(P, np.array(key, float) + np.array([-26, -14]), ",".join(ns), size=0.8)
    cap = caption_of("kin.ancestor", "a joint token attends to every joint up its kinematic chain (transitive closure of "
                     "kin_parent; grey arrows: the parent edges)", "ancestor (closure) x bias at ctx>ctx (arm-rel-v1; w=1)",
                     "public morphology (featurizer)")
    save_tile(P, cap, "computed", arm_meta(s, T, ARM_TICK, dict(query="morph joint token j7", ancestors=anc,
                                                                code_path="FactorSite(['kin.ancestor'], ctx>ctx).contributions -> ClosureOp over arm-rel-v1 kin_parent, w=1")))


# ================================================================================================ geometry tiles
def to_base(feat, P):
    """World -> robot base frame, exactly `Featurizer._to_base` (vectorized): the frame the arm ctx pos3d field uses."""
    c, s_ = math.cos(-feat.base_yaw), math.sin(-feat.base_yaw)
    d = np.asarray(P, float) - feat.base
    return np.stack([c * d[..., 0] - s_ * d[..., 1], s_ * d[..., 0] + c * d[..., 1], d[..., 2]], -1)


def non_robot(s, sc):
    rob = [b for b in range(s.model.nbody) if s.model.body(b).name.startswith(("r0_", "r1_"))]
    return sc["hit"] & ~np.isin(sc["body"], rob)


def heat_img(sc, mask, vals, cmap, vmin, vmax, alpha=0.80):
    t = np.clip((vals - vmin) / (vmax - vmin), 0, 1)
    return MF.blend(sc["rgb"], cmap(t)[..., :3], mask, alpha)


@tile("geo.pos3d")
def tile_geo_pos3d():
    s, T, tok, sc, uv = arm_setup(cam=MF.ARM_CAM)
    rc = arm_rc(tok)
    names = tok["names"]
    pos = rc.sets["ctx"].field("pos3d")[0].numpy().astype(np.float64)
    pv = rc.sets["ctx"].field("pos3d.valid")[0].numpy()
    q = names.index("asm:gripper")
    a, bb = np.array([30.0, 30.0, 30.0]), np.array([0.0, 0.0, -8.0])
    mask = non_robot(s, sc)
    pix = to_base(tok["feat"], sc["pts"][mask])
    vals = np.zeros((H, W))
    vals[mask] = MF.pape_values(pos[q:q + 1], pix, a, bb)[0][0]
    keys = [j for j in range(len(names)) if pv[j] and names[j] in ("red cube", "blue cube", "yellow cube", "green target zone")]
    tv, err = MF.pape_values(pos[q:q + 1], pos[keys], a, bb)
    cm = seq(GEOM)
    vmin, vmax = -2.5, 0.6
    img = heat_img(sc, mask, vals, cm, vmin, vmax)
    P = Tile("geo.pos3d", r"$-(r_j-r_i)^{\top}M(r_j-r_i)+\tilde b^{\top}(r_j-r_i)$",
             "given (FK + tracker); illustrative coeffs", ARM_TAG, img)
    P.cbar = (f"{vmax:+.1f}", f"{vmin:+.1f}", cm)
    if uv[q] is not None:
        MF.ring(P, uv[q], r=13)
        MF.text(P, uv[q] + np.array([-100, 6]), "query: gripper", size=0.9)
    for k, j in enumerate(keys):
        if uv[j] is not None and MF.in_frame(uv[j], 8):
            MF.dot(P, uv[j], "#FFFFFF", r=4)
            MF.text(P, uv[j] + np.array([0, 16]), f"{names[j].replace('green target zone', 'target')} {tv[0, k]:+.2f}", size=0.9)
    cap = caption_of("geo.pos3d", "PaPE relative-position term: tokens near and below the gripper score high; heat = "
                     "the term at every surface point", "sqdiff+diff x aug, a=30, b=(0,0,-8), W_p=I (learned per head)",
                     "public FK + tracker pos3d (given)")
    save_tile(P, cap, "computed", arm_meta(s, T, ARM_TICK, dict(
        query="asm:gripper (TCP frame token)", coefficients=dict(a=a.tolist(), b=bb.tolist(), g=1.0, W_p="I"),
        token_values={names[j]: float(tv[0, k]) for k, j in enumerate(keys)}, closed_form_max_abs_err=err,
        pixel=dict(min=float(vals[mask].min()), max=float(vals[mask].max())),
        code_path="nets.batch.ctx_geometry_fields pos3d (base frame) -> make_figures.pape_values (FactorSite(['geo.pos3d']).contributions, asserted == closed form); pixels: mj_multiRay world points -> base frame")))


def fixed_cam(m, d, name):
    """A make_figures.Cam that reproduces a declared MuJoCo camera (no roll): lookat = pos + fwd, az/el from fwd."""
    from rrp.envs.mujoco.sensors import camera_frame
    cpos, cmat, fovy = camera_frame(m, d, name)
    fwd = -cmat[:, 2]
    az = math.degrees(math.atan2(fwd[1], fwd[0]))
    el = math.degrees(math.asin(np.clip(fwd[2], -1, 1)))
    return MF.Cam(lookat=cpos + fwd * 1.0, dist=1.0, az=az, el=el, fovy=math.degrees(fovy))


@tile("geo.depth3d")
def tile_geo_depth3d():
    from rrp.envs.mujoco.sensors import project_points
    from rrp.policies.nets.batch import attach_cam_uvd
    s, T = MF.arm_at_tick(ARM_TICK)
    tok = arm_tokens(s)
    sc = arm_render(s, MF.ARM_CAM)
    uv = [None if w is None else MF.uv_of(sc, w)[0] for w in tok["world"]]
    rc = arm_rc(tok)
    names = tok["names"]
    pos, pv = rc.sets["ctx"].field("pos3d"), rc.sets["ctx"].field("pos3d.valid")
    # pos3d is base frame; the camera projects world points: base -> world (inverse of _to_base)
    feat = tok["feat"]
    c, s_ = math.cos(feat.base_yaw), math.sin(feat.base_yaw)
    pb = pos[0].numpy().astype(np.float64)
    pw = np.stack([c * pb[:, 0] - s_ * pb[:, 1], s_ * pb[:, 0] + c * pb[:, 1], pb[:, 2]], -1) + feat.base
    cu = attach_cam_uvd(torch.as_tensor(pw[None], dtype=torch.float32), pv, [(s.model, s.data, "front")])
    uvd = cu["cam_uvd"][0].numpy().astype(np.float64)
    ok = cu["cam_uvd.valid"][0].numpy()
    q = names.index("red cube")
    a, bb = np.array([0.0, 0.0, 60.0]), np.zeros(3)
    mask = non_robot(s, sc)
    puvd = project_points(s.model, s.data, "front", sc["pts"][mask]).astype(np.float64)
    vals = np.zeros((H, W))
    vals[mask] = MF.pape_values(uvd[q:q + 1], puvd, a, bb)[0][0]
    keys = [j for j in range(len(names)) if ok[j] and names[j] in ("blue cube", "yellow cube")]
    tv, err = MF.pape_values(uvd[q:q + 1], uvd[keys], a, bb)
    cm = seq(GEOM)
    vmin, vmax = -0.6, 0.0
    img = heat_img(sc, mask, vals, cm, vmin, vmax)
    P = Tile("geo.depth3d", r"$-(r_j-r_i)^{\top}M(r_j-r_i),\ r=(u,v,\mathrm{depth})$",
             "given (projected tracker pos3d); default = probe", ARM_TAG, img)
    P.cbar = ("0", f"{vmin:.1f}", cm)
    MF.ring(P, uv[q], r=13)
    MF.text(P, uv[q] + np.array([-40, 40]), f"query: red cube, depth {uvd[q, 2]:.2f} m", size=0.9)
    for k, j in enumerate(keys):
        if uv[j] is not None and MF.in_frame(uv[j], 8):
            MF.dot(P, uv[j], "#FFFFFF", r=4)
            MF.text(P, uv[j] + np.array([0, 18]), f"{tv[0, k]:+.2f} (depth {uvd[j, 2]:.2f})", size=0.85)
    from rrp.envs.mujoco.sensors import camera_frame
    cpos, cmat, _ = camera_frame(s.model, s.data, "front")
    MF.text(P, (300, 412), "heat: every surface point projected into the declared front camera", size=0.85)
    cap = caption_of("geo.depth3d", "PaPE over (u, v, depth) in the declared front camera; weight on depth only, so "
                     "tokens at the query's depth score high", "sqdiff+diff x aug on cam_uvd, a=(0,0,60), b=0 (learned)",
                     "projected tracker pos3d (given); default source = probe")
    save_tile(P, cap, "computed", arm_meta(s, T, ARM_TICK, dict(
        camera="cam_uvd w.r.t. the declared 'front' camera (looks along -x toward the robot); tile rendered from Figure 2's arm view", query="red cube scene slot",
        coefficients=dict(a=a.tolist(), b=bb.tolist(), g=1.0, W_p="I"), token_values={names[j]: float(tv[0, k]) for k, j in enumerate(keys)},
        token_cam_uvd={names[j]: uvd[j].tolist() for j in range(len(names)) if ok[j]}, closed_form_max_abs_err=err,
        code_path="ctx pos3d (base frame) -> world -> nets.batch.attach_cam_uvd(front) -> PaPE kernel (make_figures.pape_values = FactorSite contributions); pixels: project_points of ray-cast surface points")))


def entity_tokens(s):
    """(ids, gt labels pos3d / orient / contact_normal) for the StateView entities (privileged, labels only)."""
    from rrp.harness.data import relgen as RG
    sv = s.state_view()
    ids, pos, kinds = MF.entity_table(sv)
    lab = {n: RG.LABELS[n].fn(sv, RG.TokenIndex(sets={"ctx": ids})) for n in ("pos3d", "orient", "contact_normal")}
    return sv, ids, pos, lab


def label_set(ids, lab, names):
    RB, _ = _rel()
    n = len(ids)
    labels = {}
    for k in names:
        labels[k] = torch.as_tensor(np.asarray(lab[k].value, np.float32))[None]
        labels[k + ".valid"] = torch.as_tensor(np.asarray(lab[k].valid))[None]
    return RB.TokenSet("ctx", torch.ones(1, n, dtype=torch.bool), labels=labels)


@tile("geo.orient")
def tile_geo_orient():
    RB, _ = _rel()
    s, T = MF.arm_at_tick(ARM_TICK)
    sc = arm_render(s, MF.ARM_CAM)
    sv, ids, pos, lab = entity_tokens(s)
    ts = label_set(ids, lab, ["orient"])
    rc = RB.RelCtx(sets={"ctx": ts})
    fs, _ = site_of([{"name": "geo.orient", "source": "gt"}], "ctx>ctx", ("orient",))
    set_coeff(fs.f["geo__orient"].B, np.eye(3).reshape(-1))
    n = len(ids)
    V = fs.contributions(rc, torch.zeros(1, n, 8), torch.zeros(1, n, 8))["geo.orient"][0, 0].numpy()
    q = ids.index("cube")
    R = np.asarray(lab["orient"].value).reshape(n, 3, 3)
    chk = np.array([np.trace(R[q].T @ R[j]) for j in range(n)])
    assert np.allclose(chk, V[q], atol=1e-4), (chk, V[q])
    eb = MF.entity_bodies(s, s.model)
    cm = div(GEOM)
    vals = {b: V[q, ids.index(e)] for e, bs in eb.items() if e in ids for b in bs}
    img = MF.paint_bodies(sc, vals, cm, -1, 3, 0.85)
    P = Tile("geo.orient", r"$\langle R_iB,R_j\rangle_F=\mathrm{tr}(R_i^{\top}R_j),\ B=I$", BADGE["gt_probe"], ARM_TAG, img)
    P.cbar = ("+3", "-1", cm)
    for j, e in enumerate(ids):
        p = MF.uv_of(sc, pos[j])[0]
        if not MF.in_frame(p, 10):
            continue
        for ax_i, col in enumerate(("#C0392B", "#27AE60", "#2C3E8F")):
            tip = MF.uv_of(sc, pos[j] + 0.035 * R[j][:, ax_i])[0]
            MF.line(P, [p, tip], color=col, lw=1.1)
        if j != q:
            ang = math.degrees(math.acos(np.clip((V[q, j] - 1) / 2, -1, 1)))
            nm = {"target": "target", "obj:distractor0": "blue", "obj:distractor1": "yellow", "gripper": "gripper"}.get(e, e)
            MF.text(P, p + np.array([0, 24]), f"{nm} {V[q, j]:+.2f} ({ang:.0f} deg)", size=0.85)
    MF.ring(P, MF.uv_of(sc, pos[q])[0], r=14)
    MF.text(P, MF.uv_of(sc, pos[q])[0] + np.array([0, -26]), "query: red cube", size=0.9)
    cap = caption_of("geo.orient", "relative rotation between token frames: tr(R_i^T R_j) = 1 + 2 cos(angle) (axes: "
                     "x red, y green, z blue)", "rel_rot x aug, B = I (learned per head)",
                     "gt orient label (sim truth, training-only); deployed = probe estimate (object frames are not tracked)")
    save_tile(P, cap, "computed", arm_meta(s, T, ARM_TICK, dict(
        query="cube", source="gt (relgen LABELS['orient'], privileged); arm ctx fills `orient` publicly only on assembly tokens",
        row={e: float(V[q, j]) for j, e in enumerate(ids)}, check="== trace(R_i^T R_j) (asserted)",
        code_path="FactorSite([{'name':'geo.orient','source':'gt'}], ctx>ctx).contributions -> RelRotOp, B bias = I, weights 0")))


@tile("geo.normal_align")
def tile_geo_normal_align():
    RB, _ = _rel()
    tick = 58
    s, T = MF.arm_at_tick(tick)
    cam = MF.Cam(lookat=[0.49, -0.08, 0.07], dist=0.62, az=150, el=-14, fovy=34.0)
    sc = arm_render(s, cam)
    sv, ids, pos, lab = entity_tokens(s)
    ts = label_set(ids, lab, ["contact_normal"])
    rc = RB.RelCtx(sets={"ctx": ts})
    fs, _ = site_of([{"name": "geo.normal_align", "source": "gt"}], "ctx>ctx", ("normal",))
    bq = np.array([0.0, 0.0, 1.0])
    set_coeff(fs.f["geo__normal_align"].b, bq)
    n = len(ids)
    V = fs.contributions(rc, torch.zeros(1, n, 8), torch.zeros(1, n, 8))["geo.normal_align"][0, 0].numpy()
    q = ids.index("gripper")
    N = np.asarray(lab["contact_normal"].value)
    vn = np.asarray(lab["contact_normal"].valid)
    eb = MF.entity_bodies(s, s.model)
    cm = div(GEOM)
    vals = {b: V[q, ids.index(e)] for e, bs in eb.items() if e in ids and vn[ids.index(e)] for b in bs}
    img = MF.paint_bodies(sc, vals, cm, -1, 1, 0.85)
    P = Tile("geo.normal_align", r"$\langle b_i,n_j\rangle,\ b=(0,0,1)$", BADGE["gt_probe"], ARM_TAG + f", t={tick}", img)
    P.cbar = ("+1", "-1", cm)
    cpts = {}
    for c in sv.contacts():
        cpts.setdefault(c.a, []).append(np.asarray(c.pos))
        cpts.setdefault(c.b, []).append(np.asarray(c.pos))
    for j, e in enumerate(ids):
        if not vn[j]:
            continue
        base = np.mean(cpts[e], 0) if e in cpts else pos[j]
        p0, p1 = MF.uv_of(sc, base)[0], MF.uv_of(sc, base + 0.04 * N[j])[0]
        if MF.in_frame(p0, 6):
            arrow(P, p0, p1, color=INK, lw=1.3, shrink=0)
            lbl = e.replace("obj:distractor0", "blue cube").replace("obj:distractor1", "yellow cube").replace("cube", "cube") if e != "cube" else "red cube"
            off = {"cube": (-80, -12), "gripper": (80, 14)}.get(e, (0, 22))
            MF.text(P, p0 + np.array(off), f"{lbl} {V[q, j]:+.2f}", size=0.85)
    cap = caption_of("geo.normal_align", "query direction b vs each token's contact normal (arrows: outward normal, "
                     "mean over the token's contacts)", "align x aug, world frame, b=(0,0,1) (learned per head)",
                     "gt contact_normal (sim truth, training-only); deployed = probe")
    save_tile(P, cap, "computed", arm_meta(s, T, tick, dict(
        query="gripper (query side reads only b in the world frame)", b=bq.tolist(),
        normals={e: (N[j].tolist() if vn[j] else None) for j, e in enumerate(ids)}, row={e: float(V[q, j]) for j, e in enumerate(ids)},
        note="the arm family fills no public `normal` field: resolve() accepts this factor only with source probe or gt",
        code_path="relgen LABELS['contact_normal'] -> TokenSet.labels; FactorSite([{'name':'geo.normal_align','source':'gt'}]).contributions -> AlignOp")))


STACK_CAM = MF.Cam(lookat=[0.48, -0.10, 0.07], dist=0.36, az=165, el=-18, fovy=34.0)


def stack_setup(s):
    for k, (b, z) in enumerate((("distractor0", 0.022), ("distractor1", 0.0665), ("cube", 0.111))):
        s.teleport_object(b, [0.48, -0.10, z + 0.002 * k])


@tile("geo.above")
def tile_geo_above():
    RB, _ = _rel()
    s, T, tok, sc, uv = arm_setup(tick=10, cam=STACK_CAM, setup=stack_setup)
    rc = arm_rc(tok)
    names = tok["names"]
    pos = rc.sets["ctx"].field("pos3d")
    q = names.index("yellow cube")
    mask = non_robot(s, sc)
    pix = to_base(tok["feat"], sc["pts"][mask])
    fs, _ = site_of(["geo.above"], "ctx>pix", ("pos3d",))
    set_edge_weights(fs)
    rcp = RB.RelCtx(sets={"ctx": RB.TokenSet("ctx", torch.ones(1, 1, dtype=torch.bool), fields={"pos3d": pos[:, q:q + 1]}),
                          "pix": RB.TokenSet("pix", torch.ones(1, len(pix), dtype=torch.bool),
                                             fields={"pos3d": torch.as_tensor(pix, dtype=torch.float32)[None]})})
    pv = fs.contributions(rcp)["geo.above"][0, 0, 0].numpy()
    fs2, _ = site_of(["geo.above"], "ctx>ctx", ("pos3d",))
    set_edge_weights(fs2)
    V = fs2.contributions(rc)["geo.above"][0, 0].numpy()
    vals = np.zeros((H, W))
    vals[mask] = pv
    cm = MF.diverging(GEOM, "#7C838D")
    img = heat_img(sc, mask, vals, cm, -1, 1, alpha=0.7)
    P = Tile("geo.above", r"$\mathrm{sign}((r_j-r_i)\cdot\hat z)\ \mathrm{if}\ |\cdot|>1\,\mathrm{cm}$", BADGE["public"],
             "TELEPORTED STACK (figure-only)", img)
    P.cbar = ("+1", "-1", cm)
    MF.ring(P, uv[q], r=14)
    MF.text(P, uv[q] + np.array([-70, 0]), "query", size=0.9)
    for j, n in enumerate(names):
        if uv[j] is not None and j != q and n in ("red cube", "blue cube") and MF.in_frame(uv[j], 8):
            MF.dot(P, uv[j], "#FFFFFF", r=4)
            MF.text(P, uv[j] + np.array([52, 0]), f"{n} {V[q, j]:+.0f}", size=0.9)
    cap = caption_of("geo.above", "above / below along gravity: +1 for tokens (and surface points) higher than the "
                     "query by > 1 cm, -1 lower, 0 in the dead zone (query: middle cube of a stack)",
                     "order x bias on pos3d, axis +z, margin 0.01 m (w=1)", "tracker pos3d (given)")
    save_tile(P, cap, "computed", arm_meta(s, T, 10, dict(
        task="pick_place cubes teleported into a 3-stack (Session.teleport_object, logged intervention) then 10 ticks of settling",
        scene_source="figure-only intervention scene (Figure 2 panel 5's stack)", query="yellow cube (middle)",
        row={names[j]: float(V[q, j]) for j in range(len(names)) if not np.isnan(V[q, j])},
        code_path="FactorSite(['geo.above']) ctx>ctx and ctx>pix .contributions -> OrderOp over pos3d (base frame), w=1")))


# ##############################################################################################################
# GROUP B renderers: interaction (ix.*) and task / procedure (task-graph edges, task.next_contact)
# ##############################################################################################################
# ============================================================================================== scene helpers
def dual_color_fn(objs):
    def f(bn, typ, g):
        if typ == mujoco.mjtGeom.mjGEOM_PLANE:
            return np.array([0.80, 0.81, 0.82, 1.0])
        if bn in objs:
            return np.array([0.93, 0.93, 0.92, 1.0])
        if bn.startswith(("r0_", "r1_")):
            return np.array([0.62, 0.64, 0.67, 1.0])
        if bn == "world":
            return np.array([0.78, 0.79, 0.80, 1.0])
        return np.array([0.72, 0.73, 0.74, 1.0])
    return f


def dual_at(task, tick, scene=None):
    from rrp.envs.mujoco.dual import make_dual_env
    from rrp.policies.teachers.dual import TEACHERS
    s = make_dual_env(task=task, body=DUAL_PAIR, seed=DUAL_SEED, scene=scene)
    T = TEACHERS[task](s)
    for _ in range(tick):
        s.step(T.act())
    return s, T


def dual_scene(s, cam):
    objs = {o.sim_body for o in s.scenario.objects}
    sc = MF.render_scene(s.model, s.data, cam, dual_color_fn(objs), min_agree=0.95)
    return sc


def manip_entity_of_asm(s):
    """assembly id (per robot) -> task manipulator entity id ('left' / 'right' / 'gripper'), from the public bindings."""
    out = {}
    for i, mr in enumerate(s.scenario.robots):
        for ent, asm in mr.manipulator_bindings.items():
            out[(i, asm)] = ent
    return out


# ============================================================================================== featurized token tables
def featurize(s, multi: bool):
    """Real public featurizer -> PolicyInput -> Batch, plus a name / anchor table for every ctx token."""
    from rrp.policies.features.featurizer import BANKS
    from rrp.policies.nets.batch import collate_inputs
    obs = s.observe()
    if multi:
        from rrp.policies.features.multi import multi_featurizer
        F = multi_featurizer(s)
        subs = F.subs
    else:
        from rrp.policies.features.featurizer import cached_featurizer
        F = cached_featurizer(s)
        subs = [F]
    pi = F(obs)
    batch = collate_inputs([pi])
    ti = obs.task_input
    m2e = manip_entity_of_asm(s)
    # ---- names (mirrors Featurizer.__call__ / MultiFeaturizer.__call__ bank layouts)
    morph = []
    for i, f in enumerate(subs):
        morph += [dict(name=f"r{i}:{n}", kind="node", robot=i, joint=n) for n in f.node_joint_names]
    for i, f in enumerate(subs):
        morph += [dict(name=f"r{i}:{j.name}", kind="passive", robot=i) for j in f.passive]
        morph += [dict(name=m2e.get((i, a.id), f"r{i}:{a.id}"), kind="assembly", robot=i, asm=a.id,
                       ent=m2e.get((i, a.id))) for a in f.spec.assemblies]
    scene_desc = {o.descriptor: (o.task_entity or o.sim_body) for o in s.scenario.objects}
    scene = [dict(name=scene_desc.get(od.descriptor, od.descriptor), kind="entity", desc=od.descriptor, slot=od.slot)
             for od in obs.object_descriptors]
    task = []
    evs = list(ti.definition.events) if ti else []
    st = {v.event_id: v.status for v in ti.events} if ti else {}
    for e in evs:
        task.append(dict(name=e.id, kind="event", op=e.operator, status=st.get(e.id, "?")))
    for e in evs:
        for sl in e.roles:
            b = sl.binding
            tgt = getattr(getattr(b, "entity", None), "id", None) or f"{getattr(b, 'event_id', '?')}.{getattr(b, 'output_name', '')}"
            task.append(dict(name=f"{e.id}:{sl.role}", kind="role", event=e.id, role=sl.role, bound=tgt))
    for pe in obs.predicate_estimates:
        task.append(dict(name=f"{pe.predicate}({','.join(pe.args)})", kind="pred", value=pe.value))
    inter = [dict(name=f"{r.event_id}.{r.output_name}", kind="receipt", event=r.event_id, valid=bool(r.valid))
             for r in (ti.receipts if ti else [])]
    for i, f in enumerate(subs):
        chans = [ch for ch in obs.declared_sensor_channels if ch.name.startswith(f"{f.robot_index}:")]
        inter += [dict(name=ch.name, kind="sensor", robot=i) for ch in chans]
    banks = dict(morph=morph, scene=scene, task=task, interact=inter)
    names = []
    for b in BANKS:
        n_tok = int(batch.bank_mask[b][0].sum())
        assert len(banks[b]) == n_tok or (n_tok == 1 and len(banks[b]) == 0), (b, len(banks[b]), n_tok)
        T_b = batch.bank_mask[b].shape[1]
        rows = banks[b] + [dict(name=f"<pad:{b}>", kind="pad")] * (T_b - len(banks[b]))
        for k, r in enumerate(rows):
            names.append(dict(r, bank=b, idx=batch.bank_offset[b] + k))
    assert len(names) == batch.ctx_mask.shape[1]
    # ---- 3-D anchors (public token features: assembly FK position, tracker slot estimate; joint anchors for nodes)
    tok = batch.bank_tokens
    for r in names:
        b, li = r["bank"], r["idx"] - batch.bank_offset[r["bank"]]
        if r["kind"] == "assembly":
            r["pos"] = tok["morph"][0, li, 10:13].numpy().astype(float)
        elif r["kind"] == "entity":
            r["pos"] = tok["scene"][0, li, 0:3].numpy().astype(float)
        elif r["kind"] == "node":
            j = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_JOINT, f"r{r['robot']}_" + r["joint"].split(":")[-1])
            if j < 0:
                j = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_JOINT, r["joint"])
            r["pos"] = s.data.xanchor[j].copy() if j >= 0 else None
    return pi, batch, names, obs


def edge_contrib(batch, factor, site="ctx>ctx"):
    """FactorSite([factor]).contributions with the typed-edge weight w = 1 -> [Q,K] (the REAL EdgeOp path)."""
    from rrp.policies.relations.base import EdgeSet, RelCtx, TokenSet, resolve
    from rrp.policies.relations.catalog import ARM_REL_VOCAB
    from rrp.policies.relations.ops import FactorSite
    from rrp.policies.features.featurizer import REL
    q, k = site.split(">")
    sets = {"ctx": TokenSet("ctx", batch.ctx_mask)}
    if q == "act":
        sets["act"] = TokenSet("act", batch.node_mask)
    rel = batch.ctx_rel if site == "ctx>ctx" else batch.act_rel
    rc = RelCtx(sets=sets, edges={site: EdgeSet(ARM_REL_VOCAB, rel)})
    fs = FactorSite(1, 4, site, resolve([factor]), ("edges:arm-rel-v1",))
    assert [x.name for x in fs.specs] == [factor], fs.specs
    with torch.no_grad():
        fs.w.fill_(1.0)
    v = fs.contributions(rc)[factor][0, 0].numpy()
    ref = rel[0, ..., REL[factor.split(".", 1)[1]]].numpy().astype(float)
    assert np.array_equal(v, ref), f"{factor}: FactorSite != featurizer relation channel"
    return v


# ============================================================================================== task-graph tiles (edge.*)
DUAL_CAM = None


def handover_cam(s):
    bar = s.data.xpos[s.model.body("bar").id]
    return MF.Cam(lookat=[0.34, -0.06, 0.08], dist=1.40, az=180, el=-52, fovy=34.0)


def short_ent(n):
    return {"left": "L grip", "right": "R grip", "target": "target", "bar": "bar", "gripper": "gripper", "cube": "cube",
            "peg": "peg", "fixture": "fixture", "hole": "hole"}.get(n, n)


class TaskScene:
    """One featurized dual state (+ render) shared by the task-edge tiles of one task."""

    def __init__(self, task, tick, scene=None, cam_fn=handover_cam, label=None):
        self.s, self.T = dual_at(task, tick, scene)
        self.task, self.tick = task, tick
        self.pi, self.batch, self.names, self.obs = featurize(self.s, multi=True)
        self.cam = cam_fn(self.s)
        self.sc = dual_scene(self.s, self.cam)
        self.phase = str(self.T.phase)
        self.idx = {r["name"]: r["idx"] for r in self.names}
        self.label = label

    def find(self, **kw):
        return [r["idx"] for r in self.names if all(r.get(k) == v for k, v in kw.items())]


PX0, PX1 = 372, 596            # task-token column (tile px)
COL_X0, COL_X1 = PX0 + 6, PX0 + 172


def task_tile(ts: TaskScene, factor, qidx, *, what, form="edge × bias", badge=None, extra=(), status="computed",
              site="ctx>ctx", show_roles_of=None, note=None, tag=None, keys_row=None):
    """Scene (left, scaled) + task-token column (right). Edges = nonzero entries of the query row of the REAL
    FactorSite contribution (site ctx>ctx; act>ctx tiles pass their own row afterwards)."""
    V = edge_contrib(ts.batch, factor, site) if keys_row is None else None
    row = V[qidx] if keys_row is None else keys_row
    keys = [int(k) for k in np.nonzero(row)[0]]
    names = ts.names
    t = BTile(factor, "task", factor, what, form,
             badge or ("public", "public (given): supplied task graph + runtime"),
             tag or "SCRIPTED TEACHER (privileged)", status, img=ts.sc["rgb"],
             crop=(100, 20, 520, 450), dest=(0, 36, 0.95))
    t.note = note
    pnl = len(t.marks)
    t.panel(PX0, 66, PX1, IMG_H - 6)                # bottom trimmed to the column's content below
    t.text((PX0 + 6, 72), "task / interact tokens", color="#555", size=5.2, ha="left", va="top", bg=False)
    pos = {}
    ev = [r for r in names if r["kind"] == "event"]
    want = set(extra) | set(keys) | ({qidx} if qidx is not None else set())
    if show_roles_of:
        want |= {r["idx"] for r in names if r["kind"] == "role" and r["event"] == show_roles_of}
    xr = [r for r in names if r["idx"] in want and r["kind"] in ("role", "pred", "receipt", "sensor")]
    y, bh = 88, 18
    for r in ev:
        pos[r["idx"]] = (COL_X0, y, COL_X1, y + bh)
        y += bh + 5
    if xr:
        y += 6
    for r in xr:
        pos[r["idx"]] = (COL_X0, y, COL_X1, y + bh)
        y += bh + 5
    assert y < IMG_H, "column overflow"
    t.marks[pnl]["r"][3] = float(y + 2)
    STC = {"succeeded": "#DDE9DF", "active": "#FFF1C9", "pending": "#F2F2EF"}
    for r in ev + xr:
        a, b, c, d = pos[r["idx"]]
        if r["kind"] == "event":
            lab, fc = f"{r['name']} [{r['status'][:4]}]", STC.get(r["status"], "#FFF")
        elif r["kind"] == "role":
            lab = f"{r['role']}→{short_ent(r['bound'])}" if show_roles_of else f"{r['event']}:{r['role']}→{short_ent(r['bound'])}"
            fc = "#F4F1FE"
        elif r["kind"] == "pred":
            lab, fc = r["name"], "#EEF4F8"
        elif r["kind"] == "receipt":
            lab, fc = f"rcpt {r['name']}" + ("" if r["valid"] else " (stale)"), "#F8EFE6"
        else:
            lab, fc = r["name"], "#EEE"
        hot = r["idx"] in keys
        t.box(a, b, c, d, lab, fc=fc, ec=VIOLET if hot else "#A9ADB4", lw=1.5 if hot else 0.6,
              size=4.5 if len(lab) > 26 else (4.9 if len(lab) > 22 else 5.2), bold=hot)
    # ---- scene tokens (entity slots + manipulator assemblies bound to task entities)
    placed = []
    for r in names:
        if r["kind"] in ("assembly", "entity") and r.get("pos") is not None:
            uv = t.T(MF.uv_of(ts.sc, r["pos"])[0])
            pos[r["idx"]] = ("pt", uv)
            hot = r["idx"] in keys
            if r["kind"] == "assembly" and not r.get("ent"):
                t.dot(uv, face="#E6E7E9", r=2.5, edge="#8A8F98")          # unbound assembly token (arm base)
                continue
            t.dot(uv, face=VIOLET if hot else "#FFFFFF", r=4.5)
            lab = short_ent(r["name"])
            wl = 4.5 * len(lab) + 3                     # half width (px) of a 5.4 pt mono label
            base = [np.array([0, -12]), np.array([0, 13])]
            if r["kind"] != "entity":
                base = base[::-1]
            cands = base + [np.array([wl + 8, 0]), np.array([-wl - 8, 0]), np.array([0, 24]), np.array([0, -24])]
            for off in cands:
                c = uv + off
                if all(abs(c[0] - p_[0]) > wl + w_ or abs(c[1] - p_[1]) > 14 for p_, w_ in placed):
                    break
            placed.append((c, wl))
            t.text(c, lab, size=5.4, color=INK)

    def anchor(i, toward=None):
        p = pos[i]
        if p[0] == "pt":
            return np.asarray(p[1])
        a, b, c, d = p
        return np.array([a, (b + d) / 2])

    return t, V, row, keys, pos, anchor


def finish_edges(t, q, keys, pos, row, color=VIOLET):
    """Arrows query -> keys. Scene point -> box: straight to the box's left edge. Box -> box (same column): arc on the
    right side of the column, label at the arc's apex."""
    def is_pt(i):
        return pos[i][0] == "pt"
    for k in keys:
        if is_pt(q) and not is_pt(k):
            a = np.asarray(pos[q][1]); bb = pos[k]; b = np.array([bb[0], (bb[1] + bb[3]) / 2])
            t.arrow(a, b, color=color, lw=1.2, rad=0.0, shrinkA=6, shrinkB=1)
            lp = a + (b - a) * 0.80
            t.text(lp + np.array([0, -6]), f"{row[k]:.0f}", size=5.4, color=color, weight="bold")
        elif not is_pt(q) and is_pt(k):
            qa = pos[q]; a = np.array([qa[0], (qa[1] + qa[3]) / 2]); b = np.asarray(pos[k][1])
            t.arrow(a, b, color=color, lw=1.2, rad=0.0, shrinkA=1, shrinkB=6)
            lp = a + (b - a) * 0.5
            t.text(lp + np.array([0, -8]), f"{row[k]:.0f}", size=5.4, color=color, weight="bold")
        elif not is_pt(q) and not is_pt(k):
            qa, kb = pos[q], pos[k]
            a = np.array([qa[2], (qa[1] + qa[3]) / 2]); b = np.array([kb[2], (kb[1] + kb[3]) / 2])
            dy = b[1] - a[1]
            mag = min(1.2, 70.0 / max(abs(dy), 1.0))
            rad = -mag if dy > 0 else mag
            t.arrow(a, b, color=color, lw=1.2, rad=rad, shrinkA=1, shrinkB=1)
            t.text((b[0] + 9, b[1] + (5 if dy > 0 else -5)), f"{row[k]:.0f}", size=5.2, color=color, weight="bold")
        else:
            a, b = np.asarray(pos[q][1]), np.asarray(pos[k][1])
            t.arrow(a, b, color=color, lw=1.2, rad=0.2, shrinkA=6, shrinkB=6)
            t.text((a + b) / 2 + np.array([0, -6]), f"{row[k]:.0f}", size=5.4, color=color, weight="bold")


def mark_query(t, q, pos):
    if pos[q][0] == "pt":
        t.ring(pos[q][1], r=10)
    else:
        a, b, c, d = pos[q]
        t.marks.append(dict(k="box", r=[a - 2.5, b - 2.5, c + 2.5, d + 2.5], s="", fc="none", ec=QCOL, lw=1.8, tc=INK,
                            size=5, ls="-", z=6, bold=False))


def build_task_tiles(only=None):
    tiles = []
    want = lambda f: only is None or f in only
    ts = TaskScene("handover", 75)
    N = ts.names
    I = ts.idx
    left = ts.find(kind="assembly", ent="left")[0]
    right = ts.find(kind="assembly", ent="right")[0]
    bar = ts.find(kind="entity", name="bar")[0]
    common_meta = dict(env="mujoco/dual", body=DUAL_PAIR, task="handover", seed=DUAL_SEED, tick=ts.tick, phase=ts.phase,
                       scene_source="scripted privileged HandoverTeacher, replayed from reset",
                       code_path="MultiFeaturizer(session.observe()) -> collate_inputs -> RelCtx{ctx>ctx: EdgeSet(arm-rel-v1, ctx_rel)} -> FactorSite([edge.X]).contributions (w=1); asserted == ctx_rel channel")

    def mk(factor, q, what, **kw):
        t, V, row, keys, pos, anchor = task_tile(ts, factor, q, what=what, **kw)
        mark_query(t, q, pos)
        finish_edges(t, q, keys, pos, row)
        t.meta = dict(common_meta, query=N[q]["name"], keys={N[k]["name"]: float(row[k]) for k in keys},
                      n_nonzero=len(keys), n_tokens=int(ts.batch.ctx_mask.sum()))
        return t

    if want("edge.actor_of"):
        tiles.append(mk("edge.actor_of", left, "manipulator entity ↔ every event it is (cooperating) actor of; L grip acts in take, offer, release. Typed edge channel of the arm-rel-v1 graph, one learned weight per head."))
    if want("edge.patient_of"):
        tiles.append(mk("edge.patient_of", bar, "patient entity ↔ the events that act on it: the bar is the patient of all five handover events (take, offer, receive, release, place)."))
    if want("edge.target_of"):
        tiles.append(mk("edge.target_of", left, "target entity (and every role without its own channel, here receive:source) ↔ event: the left gripper is the source of 'receive'."))
    if want("edge.destination_of"):
        tiles.append(mk("edge.destination_of", right, "destination entity ↔ event: the right gripper is where 'offer' and 'release' deliver the bar (target zone is place's destination)."))
    if want("edge.enables"):
        q = I["release"]
        tiles.append(mk("edge.enables", q, "event → event it requires COMPLETED (requires_completed): release needs receive and offer done. Public task-graph precedence."))
    if want("edge.maintained"):
        q = I["receive"]
        tiles.append(mk("edge.maintained", q, "event → event it requires ACTIVE (requires_active): receive is only valid while offer is held."))
    if want("edge.output_to"):
        q = I["receive"]
        tiles.append(mk("edge.output_to", q, "event ↔ event consuming its output: receive's reference role binds offer's 'anchor' output (event_output binding)."))
    if want("edge.produced"):
        q = ts.find(kind="receipt")[0]
        tiles.append(mk("edge.produced", q, "runtime receipt ↔ the event that produced it: the 'anchor' receipt (interact bank) was emitted by offer.",
                        extra=[q]))
    if want("edge.consumed_by"):
        q = ts.find(kind="receipt")[0]
        tiles.append(mk("edge.consumed_by", q, "runtime receipt ↔ the event whose output-binding role consumes it: offer.anchor is consumed by receive.",
                        extra=[q]))
    if want("edge.pred_arg"):
        q = I["held_by(bar,left)"]
        tiles.append(mk("edge.pred_arg", q, "predicate-estimate token ↔ its argument entities: held_by(bar, left) points at the bar slot and the left-gripper assembly.",
                        extra=[q]))
    if want("edge.role_in_event"):
        q = I["receive"]
        tiles.append(mk("edge.role_in_event", q, "role-slot token ↔ its event: receive owns four ordered role slots (actor, patient, source, reference); multiplicity is explicit.",
                        show_roles_of="receive"))
    if want("edge.role_points_to"):
        q = I["receive:source"]
        tiles.append(mk("edge.role_points_to", q, "role-slot token → the entity token it is bound to: receive:source → left-gripper assembly (an output binding points to no entity).",
                        extra=[q]))
    if want("edge.node_actor_of"):
        tiles.append(node_actor_tile(ts, common_meta))
    if want("edge.support_of"):
        tiles.append(support_of_tile())
    return tiles


def node_actor_tile(ts, common_meta):
    """act>ctx: action-node query (a left-arm joint) -> the events its manipulator acts in (2-hop via bindings)."""
    factor = "edge.node_actor_of"
    V = edge_contrib(ts.batch, factor, site="act>ctx")                       # [N, C], asserted == act_rel channel
    nodes = [r for r in ts.names if r["kind"] == "node"]
    # query: the left-arm (r0) action node whose joint anchor projects inside the scene crop, nearest the wrist
    def in_crop(r):
        if r["pos"] is None:
            return False
        uv = MF.uv_of(ts.sc, r["pos"])[0]
        return 110 < uv[0] < 510 and 70 < uv[1] < 410
    cand = [i for i, r in enumerate(nodes) if r["robot"] == 0 and in_crop(r) and V[i].any()]
    arm = [i for i in cand if "joint" in nodes[i]["joint"]]
    qn = (arm or cand)[-1]
    jn = nodes[qn]["joint"].split(":")[-1].replace("r0_", "")
    row = V[qn]
    t, _, _, keys, pos, anchor = task_tile(
        ts, factor, None, keys_row=row, form="edge × bias  (act>ctx)",
        what="action node (one joint-command token) → every event its manipulator acts in: a 2-hop composition "
             f"node → assembly → actor binding. Left-arm {jn} → " + ", ".join(ts.names[k]["name"] for k in np.nonzero(row)[0]) + ".")
    for r in nodes:
        if r["pos"] is not None:
            t.dot(t.T(MF.uv_of(ts.sc, r["pos"])[0]), face="#D9D4FB" if r["robot"] == 0 else "#EEEEEE", r=2.4,
                  edge="#6B7078", z=5)
    qa = t.T(MF.uv_of(ts.sc, nodes[qn]["pos"])[0])
    pos["q"] = ("pt", qa)
    t.ring(qa, r=8)
    t.text(clampt(qa + np.array([-12 - 4.5 * len(jn), -10]), jn, 5.2), jn, size=5.2)
    finish_edges(t, "q", keys, pos, row)
    t.meta = dict(common_meta, site="act>ctx", query=nodes[qn]["name"], keys={ts.names[k]["name"]: float(row[k]) for k in keys},
                  code_path="FactorSite('act>ctx', [edge.node_actor_of]).contributions over EdgeSet(arm-rel-v1, batch.act_rel); asserted == act_rel channel")
    return t


def support_of_tile():
    """No shipped arm/dual task graph binds a `support` role to an ENTITY (support_and_insert binds align:support to the
    support EVENT's output -> output_to). Figure-only graph variant: align:support -> entity `left`. The edge is then
    computed by the real featurizer + FactorSite; marked illustrative because the graph is edited."""
    from rrp.envs.mujoco.scenario import load_task
    tg = copy.deepcopy(load_task("support_and_insert"))
    for e in tg["events"]:
        for r in e["roles"]:
            if r["role"] == "support":
                r["binding"] = {"kind": "entity", "entity": {"id": "left", "version": 0}}

    def cam(s):
        fx = s.data.xpos[s.model.body("fixture").id] if mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, "fixture") >= 0 else np.array([0.4, 0, 0])
        return MF.Cam(lookat=[0.34, -0.06, 0.08], dist=1.40, az=180, el=-52, fovy=34.0)
    ts = TaskScene("support_insert", 60, scene={"task": tg}, cam_fn=cam)
    left = ts.find(kind="assembly", ent="left")[0]
    t, V, row, keys, pos, anchor = task_tile(
        ts, "edge.support_of", left, status="illustrative",
        what="support-role entity ↔ event: the left arm braces the fixture during align. ILLUSTRATIVE: no shipped arm/dual "
             "graph binds a support role to an entity; align:support rebound to 'left' (figure-only).",
        badge=("illustrative", "illustrative: public edge on a figure-only task-graph variant"),
        tag="SCRIPTED TEACHER (privileged)")
    mark_query(t, left, pos)
    finish_edges(t, left, keys, pos, row)
    t.meta = dict(env="mujoco/dual", body=DUAL_PAIR, task="support_insert (task graph EDITED: align:support -> entity left)",
                  seed=DUAL_SEED, tick=ts.tick, phase=ts.phase, query="left", keys={ts.names[k]["name"]: float(row[k]) for k in keys},
                  why_illustrative="no shipped arm/dual task graph binds role 'support' to an entity; only humanoid graphs do (crate), whose net family uses legged-rel-v1, not arm-rel-v1")
    return t


# ============================================================================================== interaction tiles (ix.*)
PROBE_BADGE = ("probe", "shown: gt label (training-only); deployed: estimated-probe")


def ix_label_tile(factor, s, sc, sv, ids, pos, label, qname, what, tag=ARM_TAG, contacts_of=None, meta=None,
                  skip=("target",), num_off=(0, -16)):
    Mx = MF.lab(sv, ids, label)[..., 0]
    qi = ids.index(qname)
    eb = MF.entity_bodies(s, s.model)
    cm = MF.ramp(AMBER)
    vals = {b: Mx[qi, ids.index(e)] for e, bs in eb.items() if e in ids and e != qname for b in bs}
    img = MF.paint_bodies(sc, vals, cm, 0, 1)
    t = BTile(factor, "interaction", factor, what, "bilinear × aug  (probe)", PROBE_BADGE, tag, "computed (label)", img=img)
    if contacts_of:
        for c in sv.contacts():
            if {c.a, c.b} <= set(contacts_of) or (c.a in contacts_of and c.b in contacts_of):
                uv = MF.uv_of(sc, c.pos)[0]
                t.dot(t.T(uv), face="#FFFFFF", r=2.8, edge="#7A4A00", z=6)
    return t, Mx, qi


def clampt(p, s, size=5.6):
    hw = 0.6 * size * 200 / 72 * len(max(s.split("\n"), key=len)) / 2 + 4
    return np.array([min(max(p[0], hw + 4), W - hw - 4), min(max(p[1], 34), IMG_H - 12)])


def label_entities(t, sc, ids, pos, Mx, qi, off=(0, -20), names=None):
    """Ring the query token, label every other entity token with its value; list off-frame ones in a corner note."""
    names = names or {}
    q = ids[qi]
    t.ring(t.T(MF.uv_of(sc, pos[qi])[0]), r=13)
    qp = t.T(MF.uv_of(sc, pos[qi])[0])
    t.text(clampt(qp + np.array([0, 24]), f"{names.get(q, q)} (query)"), f"{names.get(q, q)} (query)", size=5.6,
           weight="bold")
    x0, y0, x1, y1 = t.img_rect()
    offf = []
    for k, e in enumerate(ids):
        if k == qi:
            continue
        nm = names.get(e, e.replace("obj:", ""))
        p = t.T(MF.uv_of(sc, pos[k])[0])
        lab = f"{nm} {Mx[qi, k]:.0f}"
        if x0 + 4 < p[0] < x1 - 4 and y0 + 26 < p[1] < y1 - 4:
            t.dot(p, face="#FFFFFF", r=3.0, z=6)
            t.text(clampt(p + np.array(off), lab), lab, size=5.6)
        else:
            offf.append(lab)
    if offf:
        t.text((W - 10, IMG_H - 10), "off-frame: " + ", ".join(offf), size=5.2, ha="right", va="bottom", color="#3A3F47")


def entity_anchor(s, sv, sc, ids, pos, e):
    if e in ("gripper",):
        return MF.palm_uv(s, sc)
    return MF.uv_of(sc, pos[ids.index(e)])[0]


def build_ix_tiles(only=None):
    tiles = []
    want = lambda f: only is None or f in only
    meta0 = dict(env="mujoco/arm", body="panda_pg2", task="pick_place(n_distractors=2)", seed=SEED,
                 scene_source="scripted privileged PickPlaceTeacher, replayed from reset")
    if want("ix.contact"):
        s, T = MF.arm_at_tick(40)
        c = s.data.xpos[s.model.body("cube").id]
        cam = MF.Cam(lookat=[c[0], c[1] - 0.03, c[2] + 0.03], dist=0.42, az=125, el=-30, fovy=34.0)
        sc, sv, ids, pos, kinds = MF.arm_scene(s, cam)
        t, Mx, qi = ix_label_tile("ix.contact", s, sc, sv, ids, pos, "contact_pairs", "gripper",
                                  "two entities in geometric contact (sim contact truth): query gripper → cube 1, "
                                  "distractors / target 0. White dots: the gripper–cube contact points.",
                                  contacts_of=("gripper", "cube"))
        label_entities(t, sc, ids, pos, Mx, qi)
        t.meta = dict(meta0, tick=40, phase=T.phase, query="gripper", label="contact_pairs",
                      values={e: float(Mx[qi, k]) for k, e in enumerate(ids)})
        tiles.append(t)
    if want("ix.held_by"):
        s, T = MF.arm_at_tick(58)
        c = s.data.xpos[s.model.body("cube").id]
        cam = MF.Cam(lookat=[c[0], c[1] - 0.04, c[2] - 0.04], dist=0.55, az=125, el=-22, fovy=34.0)
        sc, sv, ids, pos, kinds = MF.arm_scene(s, cam)
        t, Mx, qi = ix_label_tile("ix.held_by", s, sc, sv, ids, pos, "held_pairs", "cube",
                                  "object held by a manipulator assembly (≥ 2 contacts with its hand bodies): lifted "
                                  "cube → gripper 1; target and distractors 0.")
        label_entities(t, sc, ids, pos, Mx, qi)
        t.meta = dict(meta0, tick=58, phase=T.phase, query="cube", label="held_pairs", cube_z=float(c[2]),
                      values={e: float(Mx[qi, k]) for k, e in enumerate(ids)})
        tiles.append(t)
    if want("ix.handover"):
        tiles.append(handover_tile())
    if want("ix.support") or want("ix.force_flow"):
        tiles += stack_tiles(only)
    if want("task.next_contact"):
        tiles.append(next_contact_tile())
    return tiles


def handover_tile():
    s, T = dual_at("handover", 162)
    bar = s.data.xpos[s.model.body("bar").id]
    cam = MF.Cam(lookat=[bar[0], bar[1], bar[2] - 0.02], dist=0.55, az=165, el=-25, fovy=34.0)
    sc = dual_scene(s, cam)
    sv = s.state_view()
    ids, pos, kinds = MF.entity_table(sv)
    Mx = MF.lab(sv, ids, "handover_pairs")[..., 0]
    qi = ids.index("left")
    eb = MF.entity_bodies(s, s.model)
    vals = {b: Mx[qi, ids.index(e)] for e, bs in eb.items() if e in ids and e != "left" for b in bs}
    img = MF.paint_bodies(sc, vals, MF.ramp(AMBER), 0, 1)
    t = BTile("ix.handover", "interaction", "ix.handover",
             "two manipulator assemblies in contact with the same object at once (a handover in progress): query L grip "
             "→ R grip 1, bar / target 0. Dots: bar contacts of each hand. Dual env only.",
             "bilinear × aug  (probe)", PROBE_BADGE, ARM_TAG, "computed (label)", img=img)
    for c in sv.contacts():
        if "bar" in (c.a, c.b) and ({"left", "right"} & {c.a, c.b}):
            other = c.a if c.b == "bar" else c.b
            t.dot(t.T(MF.uv_of(sc, c.pos)[0]), face="#FFFFFF" if other == "left" else AMBER, r=2.8, edge="#7A4A00", z=6)
    pa = {e: t.T(MF.uv_of(sc, pos[ids.index(e)])[0]) for e in ids}
    t.ring(pa["left"], r=13)
    t.arrow(pa["left"], pa["right"], color=AMBER_TXT, lw=1.4, rad=-0.3, shrinkA=14, shrinkB=12)
    for e in ids:
        if e == "left":
            continue
        p = pa[e]
        if 8 < p[0] < W - 8 and 20 < p[1] < IMG_H - 8:
            lab = f"{short_ent(e)} {Mx[qi, ids.index(e)]:.0f}"
            t.text(clampt(p + np.array([0, 34 if e == "bar" else 20]), lab), lab, size=5.6)
    t.text(pa["left"] + np.array([0, 20]), "L grip (query)", size=5.4)
    t.meta = dict(env="mujoco/dual", body=DUAL_PAIR, task="handover", seed=DUAL_SEED, tick=162, phase=str(T.phase),
                  query="left", label="handover_pairs", values={e: float(Mx[qi, k]) for k, e in enumerate(ids)},
                  scene_source="scripted privileged HandoverTeacher")
    return t


def stack_tiles(only=None):
    from rrp.policies.relations.base import EdgeSet, RelCtx, resolve
    from rrp.policies.relations.ops import FactorSite

    def setup(s):
        for k, (b, z) in enumerate((("distractor0", 0.022), ("distractor1", 0.0665), ("cube", 0.111))):
            s.teleport_object(b, [0.48, -0.10, z + 0.002 * k])
    s, T = MF.arm_at_tick(10, setup=setup)
    cam = MF.Cam(lookat=[0.48, -0.10, 0.065], dist=0.36, az=165, el=-20, fovy=34.0)
    sc, sv, ids, pos, kinds = MF.arm_scene(s, cam)
    idx = {e: k for k, e in enumerate(ids)}
    A = MF.lab(sv, ids, "support_pairs")[..., 0]
    C = MF.lab(sv, ids, "support_closure")[..., 0]
    stack = sorted([e for e in ids if e in ("cube", "obj:distractor0", "obj:distractor1")], key=lambda e: pos[idx[e]][2])
    eb = MF.entity_bodies(s, s.model)
    nm = {"obj:distractor0": "blue (bottom)", "obj:distractor1": "yellow (middle)", "cube": "red (top)"}
    out = []
    TAG = "TELEPORTED STACK (figure-only)"
    meta0 = dict(env="mujoco/arm", body="panda_pg2", task="pick_place cubes teleported into a 3-stack (Session.teleport_object; logged intervention), 10 settle ticks under the scripted teacher",
                 seed=SEED, tick=10, stack_bottom_to_top=stack)
    if only is None or "ix.support" in only:
        q = stack[1]
        vals = {b: A[idx[q], idx[e]] for e, bs in eb.items() if e in idx and e not in (q, "gripper", "target") for b in bs}
        img = MF.paint_bodies(sc, vals, MF.ramp(AMBER), 0, 1)
        t = BTile("ix.support", "interaction", "ix.support",
                 "a supports b: contact + contact normal within 30° of gravity-up at a's top. Query middle cube → top 1 "
                 "(row); it is itself supported by the bottom cube (grey arrow, column). Emits the estimated support-v1 graph.",
                 "bilinear × aug  (probe, emits)", PROBE_BADGE, TAG, "computed (label)", img=img)
        pq = t.T(MF.uv_of(sc, pos[idx[q]])[0])
        t.ring(pq, r=14)
        for c in sv.contacts():
            if {c.a, c.b} in ({q, stack[2]}, {q, stack[0]}):
                p0 = t.T(MF.uv_of(sc, c.pos)[0])
                t.dot(p0, face="#FFFFFF", r=2.4, edge="#7A4A00", z=6)
        for e in stack:
            p = t.T(MF.uv_of(sc, pos[idx[e]])[0])
            t.text(p + np.array([44, 0]), f"{nm[e]}  {A[idx[q], idx[e]]:.0f}" if e != q else f"{nm[e]}  (query)", size=5.6, ha="left")
        p_top = t.T(MF.uv_of(sc, pos[idx[stack[2]]])[0])
        p_bot = t.T(MF.uv_of(sc, pos[idx[stack[0]]])[0])
        t.arrow(pq + np.array([-46, 0]), p_top + np.array([-46, 0]), color=AMBER_TXT, lw=1.5, rad=-0.4, shrinkA=2, shrinkB=2)
        t.arrow(p_bot + np.array([-46, 0]), pq + np.array([-46, 0]), color="#8A8F98", lw=1.1, rad=-0.4, ls="--", shrinkA=2, shrinkB=2)
        t.text(pq + np.array([-80, -28]), "supports", size=5.2, color=AMBER_TXT, bg=True)
        t.meta = dict(meta0, query=q, label="support_pairs", row={e: float(A[idx[q], k]) for k, e in enumerate(ids)},
                      column={e: float(A[k, idx[q]]) for k, e in enumerate(ids)})
        out.append(t)
    if only is None or "ix.force_flow" in only:
        n = len(ids)
        rc = RelCtx(sets={"ctx": MF.tset("ctx", pos)},
                    edges={"ctx>ctx#support-v1@gt": EdgeSet(("support",), torch.as_tensor(A, dtype=torch.float32)[None, :, :, None], prov="privileged")})
        site = FactorSite(1, 3, "ctx>ctx", resolve([{"name": "ix.force_flow", "source": "gt"}]), ("edges:support-v1",))
        with torch.no_grad():
            site.f["ix__force_flow"].w.fill_(1.0)
        flow = site.contributions(rc, torch.zeros(1, n, 3), torch.zeros(1, n, 3))["ix.force_flow"][0, 0].numpy()
        assert np.array_equal(flow > 0.5, C > 0.5)
        q = stack[0]
        vals = {b: flow[idx[q], idx[e]] for e, bs in eb.items() if e in idx and e not in (q, "gripper", "target") for b in bs}
        img = MF.paint_bodies(sc, vals, MF.ramp(AMBER), 0, 1)
        t = BTile("ix.force_flow", "interaction", "ix.force_flow",
                 "transitive closure of the support graph: everything the query carries, directly or through others. "
                 "Bottom cube → middle 1 and top 1 (direct edge only to the middle). Op: flow (closure) over support-v1.",
                 "flow × bias", ("probe", "shown: closure of gt support (training-only); deployed: closure of the ix.support estimated-probe"),
                 TAG, "computed", img=img)
        pq = t.T(MF.uv_of(sc, pos[idx[q]])[0])
        t.ring(pq, r=14)
        for e in stack:
            p = t.T(MF.uv_of(sc, pos[idx[e]])[0])
            t.text(p + np.array([44, 0]), f"{nm[e]}  {flow[idx[q], idx[e]]:.0f}" if e != q else f"{nm[e]}  (query)", size=5.6, ha="left")
        pm = t.T(MF.uv_of(sc, pos[idx[stack[1]]])[0])
        p_top = t.T(MF.uv_of(sc, pos[idx[stack[2]]])[0])
        t.arrow(pq + np.array([-46, 0]), pm + np.array([-46, 0]), color="#8A8F98", lw=1.0, rad=-0.4, ls="--", shrinkA=2, shrinkB=2)
        t.arrow(pm + np.array([-46, 0]), p_top + np.array([-46, 0]), color="#8A8F98", lw=1.0, rad=-0.4, ls="--", shrinkA=2, shrinkB=2)
        t.arrow(pq + np.array([-62, 0]), p_top + np.array([-62, 0]), color=AMBER_TXT, lw=1.5, rad=-0.45, shrinkA=2, shrinkB=2)
        t.text(pq + np.array([-140, -40]), "closure\n(dashed: direct\nsupport edges)", size=5.0, color=AMBER_TXT)
        t.meta = dict(meta0, query=q, flow_row={e: float(flow[idx[q], k]) for k, e in enumerate(ids)},
                      code_path="FactorSite([{ix.force_flow, source: gt}]).contributions -> ClosureOp over rc.edges['ctx>ctx#support-v1@gt'] (relgen support_pairs); asserted == relgen support_closure",
                      closure_check="flow == support_closure for all pairs")
        out.append(t)
    return out


def next_contact_tile():
    """Task-gated: gate text (active event from the public runtime), PUBLIC candidate edges (R18
    candidate_interaction_edges, graspable row of the gripper's sensor token) vs the selected edge (relgen
    next_contact label + reveal posterior with the default contact-truth evidence)."""
    from rrp.harness.data import relgen as RG
    import rrp.harness.data.relgen.transforms  # noqa: F401
    from rrp.harness.data.relgen import TRANSFORMS
    from rrp.harness.data.relgen import task as TK
    from rrp.policies.nets.batch import candidate_interaction_edges
    s, T = MF.arm_at_tick(40)
    cam = MF.Cam(lookat=[0.45, -0.03, 0.03], dist=0.72, az=125, el=-34, fovy=34.0)
    sc, sv, ids, pos, kinds = MF.arm_scene(s, cam)
    pi, batch, names, obs = featurize(s, multi=False)
    E = candidate_interaction_edges([pi], batch)
    G = E.data[0, ..., E.channel("graspable")].numpy()
    sens = [r for r in names if r["kind"] == "sensor"]
    qs = next(r for r in sens if "touch" in r["name"]) if any("touch" in r["name"] for r in sens) else sens[0]
    cand = {names[k]["name"]: float(G[qs["idx"], k]) for k in np.nonzero(G[qs["idx"]])[0]}
    # selected: privileged label + reveal posterior over the candidate entity ids
    ent_of = {r["name"]: r["name"] for r in names if r["kind"] == "entity"}
    cands_ent = [c for c in cand]
    sv_ids = set(ids)
    map_id = {c: (c if c in sv_ids else ("obj:" + c if "obj:" + c in sv_ids else c)) for c in cands_ent}
    smp = TK.next_contact_sample([map_id[c] for c in cands_ent], sv)
    [o] = TRANSFORMS["reveal"].fn(smp, np.random.default_rng(0), {"schedule": [0]})
    lk = next(iter(o["labels"]))
    post = {c: float(v) for c, v in zip(cands_ent, np.asarray(o["labels"][lk]["value"]).reshape(-1))}
    nowv = np.asarray(RG.LABELS["next_contact"].fn(sv, RG.TokenIndex(sets={"ctx": ids})).value)[:, 0]
    now = {e: float(nowv[k]) for k, e in enumerate(ids)}
    ti = obs.task_input
    active = [e for e in ti.definition.events if {v.event_id: v.status for v in ti.events}.get(e.id) == "active"]
    gate_txt = "; ".join(f"{e.operator}(" + ", ".join(f"{sl.role}={getattr(getattr(sl.binding, 'entity', None), 'id', '?')}" for sl in e.roles) + ")" for e in active) or "(no active event)"
    eb = MF.entity_bodies(s, s.model)
    vals = {b: post.get(e.replace("obj:", ""), post.get(e, 0.0)) for e, bs in eb.items() if e in ids and e not in ("gripper",) and (e in post or e.replace("obj:", "") in post) for b in bs}
    img = MF.paint_bodies(sc, vals, MF.ramp(VIOLET), 0, 1)
    t = BTile("task.next_contact", "task", "task.next_contact",
             "manipulator → graspable score, sharpened by the task gate (g_task = 1.00 at init, learned). Dashed: "
             "PUBLIC candidate edges (1/|E|); solid: selected (gt reveal posterior given contact).",
             "bilinear × aug, gated", ("probe", "candidates: public; selected: gt (training-only); deployed: estimated-probe × g_task"),
             ARM_TAG, "computed (label)", img=img)
    pg = t.T(MF.palm_uv(s, sc))                    # the touch-sensor token has no pose: anchored at the palm it rides on
    t.ring(pg, r=13)
    ql = "gripper touch-sensor token (query)"
    t.text(clampt(pg + np.array([70, -24]), ql, 5.4), ql, size=5.4, weight="bold")
    offf, placed = [], []
    for c, pval in cand.items():
        e = map_id[c]
        if e not in ids:
            continue
        pe = t.T(MF.uv_of(sc, pos[ids.index(e)])[0])
        sel = post.get(c, 0.0)
        if not (4 < pe[0] < W - 4 and 26 < pe[1] < IMG_H - 4):
            offf.append(f"{c}: cand {pval:.2f} · sel {sel:.2f}")
            continue
        t.arrow(pg, pe, color="#5D626A", lw=1.0, ls="--", rad=0.22, shrinkA=13, shrinkB=7, style="-")
        if sel > 0.5:
            t.arrow(pg, pe, color=VIOLET, lw=2.2, rad=0.0, shrinkA=13, shrinkB=7)
        lab = f"{c}\ncand {pval:.2f} · sel {sel:.2f}"
        x0_, y0_, x1_, y1_ = t.img_rect()
        if not (x0_ + 4 < pe[0] < x1_ - 4 and y0_ + 26 < pe[1] < y1_ - 4):
            offf.append(f"{c}: cand {pval:.2f} · sel {sel:.2f}")
            continue
        t.dot(pe, face="#FFFFFF", r=3.0, z=6)
        hw = 0.6 * 5.2 * 200 / 72 * 21 / 2 + 6
        for off in ([0, 22], [hw + 10, 6], [-hw - 10, 6], [0, -24]):
            lp = clampt(pe + np.array(off), lab, 5.2)
            if np.linalg.norm(lp - pg) >= 110 and all(abs(lp[0] - q_[0]) > 2 * hw or abs(lp[1] - q_[1]) > 28 for q_ in placed):
                break
        placed.append(lp)
        t.text(lp, lab, size=5.2)
    if offf:
        t.text((W - 10, IMG_H - 10), "off-frame: " + "; ".join(offf), size=5.2, ha="right", va="bottom", color="#3A3F47")
    # gate box (top-left, under the scene chip)
    gx0, gy0 = 8, 30
    t.marks.append(dict(k="box", r=[gx0, gy0, gx0 + 318, gy0 + 38], s="", fc=(1, 1, 1, 0.94), ec=VIOLET, lw=1.0, tc=INK,
                        size=5, ls="-", z=8, bold=False))
    t.text((gx0 + 6, gy0 + 10), "g_task ← task text (active event):", size=5.0, ha="left", bg=False, color="#555")
    t.text((gx0 + 6, gy0 + 27), gate_txt[:50], size=5.4, ha="left", bg=False, color=VIOLET, weight="bold")
    t.meta = dict(env="mujoco/arm", body="panda_pg2", task="pick_place(n_distractors=2)", seed=SEED, tick=40, phase=T.phase,
                  query=qs["name"], candidates=cand, posterior=post, next_contact_label_now=now, gate_text=gate_txt,
                  gate_value_at_init=1.0,
                  code_path="nets.batch.candidate_interaction_edges (public, graspable row of the sensor token); relgen.task.next_contact_sample(default contact-truth evidence) -> TRANSFORMS['reveal'](schedule=[0]); gate = FactorSite._gate = 2*sigmoid(u·task + b) = 1 at init")
    return t

# ##############################################################################################################
# GROUP C renderers: locomotion (leg.*), UI (ui.*) and probe readouts (probe.*)
# ##############################################################################################################
ARM_H = 16                                                    # pack horizon of the arm latent recipes (horizon: 16)


def arm_episode():
    """One scripted privileged PickPlaceTeacher episode (Figure-2 scene: panda_pg2, pick_place, 2 distractors, seed
    3000001) recorded with the REAL collection hook (_TeacherTrace -> privileged_labels per tick) plus a qpos
    snapshot per tick, then turned into training samples by chunks.episode_samples (H=16), exactly as pack_dataset."""
    from rrp.harness import hooks as HK
    from rrp.harness.data.collect import _TeacherTrace, run_teacher_rollout
    from rrp.harness.data.chunks import episode_samples
    from rrp.policies.features.featurizer import featurizer_for
    from rrp.policies.teachers.arm import PickPlaceTeacher
    s = MF.arm_session(seed=SEED)
    feat = featurizer_for(s)
    T = PickPlaceTeacher(s, robot=0)
    tr = _TeacherTrace(s, T, feat, exec_noise=0.0, noise_seed=0, dart_safety=None, dart_descent_sigma=0.0)
    snaps = []
    rec = HK.Recorder(on_act=lambda i, env, act: snaps.append(env.data.qpos.copy()))
    ep = run_teacher_rollout(s, T, version="fig_tiles", hooks=[tr, rec], max_steps=600, settle=1)
    a = feat.aspace
    pub = dict(meta=dict(episode_id="tiles"), inputs=tr.inputs, actions=tr.actions, q0=tr.q0s, statuses=tr.statuses,
               action_space=dict(node_group=a.node_group, node_col=a.node_col, lower=a.lower, upper=a.upper,
                                 is_gripper=a.is_gripper, open_value=a.open_value, closed_value=a.closed_value,
                                 delta_scale=a.delta_scale))
    prv = dict(labels=tr.labels, phases=tr.phases, manipulators=list(s.manip_map), slots=[o.sim_body for o in s.detectables])
    samples = episode_samples(pub, prv, ARM_H)
    return s, T, ep, snaps, samples, prv


def arm_labels_at(sample, prv):
    from rrp.harness.data.chunks import collate_samples
    from rrp.harness.data.packed import _focus
    from rrp.policies.features.derived import active_operator, OPERATORS
    from rrp.policies.nets.latent_batch import goal_effect_from_batch
    L = dict(sample.labels)
    S = len(L["held"])
    L["focus"] = _focus(sample.pi, S)
    L["subtask"] = int(active_operator(sample.pi))
    L["subtask_name"] = OPERATORS[L["subtask"]]
    L["goal_effect"] = goal_effect_from_batch(collate_samples([sample])[0])[0].numpy()[:S]
    return L, OPERATORS


def arm_render_at(s, snaps, t, cam):
    d = s.data
    d.qpos[:] = snaps[t]
    d.qvel[:] = 0
    mujoco.mj_forward(s.model, d)
    sc, sv, ids, pos, kinds = MF.arm_scene(s, cam)
    return sc


def slot_world(s, slot):
    return s.data.xpos[s.model.body(slot).id].copy()


def slot_bodies(s, slot):
    """Body ids rendered for a detectable slot (its body and children)."""
    m = s.model
    b0 = m.body(slot).id
    return [b for b in range(m.nbody) if b == b0 or m.body_rootid[b] == b0 and b != 0 and m.body_parentid[b] == b0]


def tcp_world(s):
    ri, asm = s.manip_map["gripper"]
    sid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_SITE, s.robots[ri].tcp_sites[asm])
    return s.data.site_xpos[sid].copy()


SLOT_SHORT = {"cube": "cube", "target_zone": "target", "distractor0": "dist0", "distractor1": "dist1"}


def build_arm(records):
    from rrp.policies.features.derived import OPERATORS
    s, T, ep, snaps, samples, prv = arm_episode()
    slots = prv["slots"]
    phases = prv["phases"]
    tag = "SCRIPTED TEACHER (privileged)"
    cam_wide = MF.Cam(lookat=[0.47, -0.04, 0.14], dist=0.95, az=165, el=-30, fovy=34.0)
    base_meta = dict(env="mujoco/arm", body="panda_pg2", task="pick_place(n_distractors=2)", seed=SEED,
                     episode_steps=int(ep.steps), outcome=ep.outcome, slots=slots, horizon_H=ARM_H,
                     scene_source="scripted privileged PickPlaceTeacher episode via harness.data.collect._TeacherTrace (the real collection hook); state replayed per tick from qpos snapshots",
                     label_code="collect.privileged_labels -> chunks.episode_samples(H=16) -> packed._focus / derived.active_operator / latent_batch.goal_effect_from_batch")

    def first(pred):
        return next(t for t in range(len(samples)) if pred(t))

    t_occ = first(lambda t: not samples[t].labels["visible"][0])                       # cube occluded by the closing gripper
    t_lift = first(lambda t: phases[t] == "lift") + 8
    t_tr = first(lambda t: phases[t] == "transport") + 2
    t_pre = 6
    paint = lambda sc, vals, cm: MF.paint_bodies(sc, vals, cm, 0, 1)

    def slot_paint(sc, vals, cm):
        bv = {}
        for k, sl in enumerate(slots):
            for b in slot_bodies(s, sl):
                bv[b] = vals[k]
        return paint(sc, bv, cm)

    def labels_text(P, sc, fmt):
        for k, sl in enumerate(slots):
            uv = MF.uv_of(sc, slot_world(s, sl))[0]
            if MF.in_frame(uv, 20):
                MF.text(P, uv + np.array([0, -24]), fmt(k, sl), size=1.0)

    # ---- visible (per entity, bce): camera visibility of each slot in the 'front' camera
    t = t_occ
    L, _ = arm_labels_at(samples[t], prv)
    cam_close = MF.Cam(lookat=[0.47, -0.04, 0.10], dist=0.62, az=160, el=-32, fovy=34.0)
    sc = arm_render_at(s, snaps, t, cam_close)
    v = L["visible"].astype(float)
    P = MF.Panel(0, "probe.arm.visible", "x", GEOM, "", r"$\mathrm{visible}(e)\in\{0,1\}$", BADGE["gt"], f"{tag} t={t}",
                 slot_paint(sc, v, MF.ramp(GEOM)))
    labels_text(P, sc, lambda k, sl: f"{SLOT_SHORT[sl]} {int(v[k])}")
    records_meta = dict(base_meta, tick=t, phase=phases[t], value={sl: int(v[k]) for k, sl in enumerate(slots)},
                        note="label = envs.mujoco.sensors.camera_visibility(model, data, 'front', slot) (privileged); colour = label")
    save_tile(P, caption_of("probe.arm.visible", "readout of whether each entity is visible in the front camera (cube occluded by the closing gripper here: 0)",
                            "readout (inert) on the packet, address entity, bce", "gt sim truth, training-only target; deployed = probe estimate"),
              "computed", records_meta, records)

    # ---- looking_at (per entity, mse, /30): angle (deg) between front-camera axis and the entity
    t = t_pre
    L, _ = arm_labels_at(samples[t], prv)
    sc = arm_render_at(s, snaps, t, cam_wide)
    g = L["gaze"]
    P = MF.Panel(0, "probe.arm.looking_at", "x", GEOM, "", r"$\angle(\mathrm{cam\ axis},\ p_e-c)$ [deg]", BADGE["gt"],
                 f"{tag} t={t}", slot_paint(sc, 1 - np.clip(g / 30.0, 0, 1), MF.ramp(GEOM)))
    P.cbar = ("0°", "30°", MF.ramp(GEOM))
    labels_text(P, sc, lambda k, sl: f"{SLOT_SHORT[sl]} {g[k]:.1f}°")
    save_tile(P, caption_of("probe.arm.looking_at", "readout of each entity's angle off the front-camera optical axis (gaze label, degrees, scaled 1/30)",
                            "readout (inert), address entity, mse", "gt sim truth (camera pose + body pose), training-only"),
              "computed", dict(base_meta, tick=t, phase=phases[t], value={sl: float(g[k]) for k, sl in enumerate(slots)},
                               note="collect.privileged_labels slot_gaze_angle; front camera is the scene camera, not the render view"), records)

    # ---- focused_on (per entity, bce): slots bound as patient/target/destination of an ACTIVE task event (public)
    t = t_tr
    L, _ = arm_labels_at(samples[t], prv)
    sc = arm_render_at(s, snaps, t, cam_wide)
    f = L["focus"].astype(float)
    P = MF.Panel(0, "probe.arm.focused_on", "x", VIOLET, "", r"$\mathrm{focus}(e)=\exists$ active event $\ni e$", BADGE["public_label"],
                 f"{tag} t={t}", slot_paint(sc, f, MF.ramp(VIOLET)))
    labels_text(P, sc, lambda k, sl: f"{SLOT_SHORT[sl]} {int(f[k])}")
    save_tile(P, caption_of("probe.arm.focused_on", f"readout of which entities a currently ACTIVE task event binds (active op '{L['subtask_name']}': cube + target)",
                            "readout (inert), address entity, bce", "public label (task runtime status + bindings)"),
              "computed", dict(base_meta, tick=t, phase=phases[t], value={sl: int(f[k]) for k, sl in enumerate(slots)},
                               note="packed._focus(pi): task event tokens with status active -> scene slots via relations 4/5/6 (patient/target/destination)"), records)

    # ---- held_by / acting_on (entity x asm, bce): close view, lift phase
    cam_h = MF.Cam(lookat=[0.48, -0.04, 0.12], dist=0.58, az=118, el=-20, fovy=34.0)
    for name, key, what, form in (
            ("probe.arm.held_by", "held", "readout of whether entity e is held by assembly a (here gripper; cube 1, others 0)", r"$\mathrm{held}(e,a)$"),
            ("probe.arm.acting_on", "contact", "readout of whether assembly a's hand links touch entity e (contact truth)", r"$\mathrm{contact}(e,a)$")):
        t = t_lift
        L, _ = arm_labels_at(samples[t], prv)
        sc = arm_render_at(s, snaps, t, cam_h)
        v = L[key].astype(float)
        P = MF.Panel(0, name, "x", AMBER, "", form, BADGE["gt"], f"{tag} t={t}", slot_paint(sc, v, MF.ramp(AMBER)))
        MF.ring(P, MF.palm_uv(s, sc), r=16)
        MF.text(P, MF.palm_uv(s, sc) + np.array([0, -30]), "a = gripper", size=0.95)
        labels_text(P, sc, lambda k, sl: f"{SLOT_SHORT[sl]} {int(v[k])}")
        save_tile(P, caption_of(name, what, "readout (inert), address entity x assembly, bce",
                                "gt sim truth (Session.truth held_by / contacts), training-only"),
                  "computed", dict(base_meta, tick=t, phase=phases[t], query_assembly="gripper",
                                   value={sl: int(v[k]) for k, sl in enumerate(slots)}), records)

    # ---- rel_pos (entity x asm, gauss 6 = mu,logvar of 3): p_e - p_tcp
    t = first(lambda t: phases[t] == "descend")
    L, _ = arm_labels_at(samples[t], prv)
    sc = arm_render_at(s, snaps, t, cam_wide)
    P = MF.Panel(0, "probe.arm.rel_pos", "x", GEOM, "", r"$p_e - p_{\rm tcp}(a)$ [m]", BADGE["gt"], f"{tag} t={t}", sc["rgb"])
    tcp = tcp_world(s)
    uvt = MF.uv_of(sc, tcp)[0]
    MF.ring(P, uvt, r=12)
    MF.text(P, uvt + np.array([0, -26]), "tcp(gripper)", size=0.9)
    vals = {}
    for k, sl in enumerate(slots):
        r_ = L["rel_tcp"][k]
        pe = tcp + r_                          # rel_tcp = p - tcp in world frame (collect.privileged_labels)
        uve = MF.uv_of(sc, pe)[0]
        line_arrow(P, uvt, uve, color=GEOM, lw=1.4)
        MF.dot(P, uve, "#FFFFFF", r=4)
        MF.text(P, uve + np.array([0, 16]), f"{SLOT_SHORT[sl]} ({r_[0]:+.2f},{r_[1]:+.2f},{r_[2]:+.2f})", size=0.75)
        vals[sl] = r_.tolist()
    save_tile(P, caption_of("probe.arm.rel_pos", "readout of entity position relative to the assembly's TCP, (x,y,z) m, world frame",
                            "readout (inert), address entity x assembly, gauss (mu, logvar)", "gt sim truth, training-only"),
              "computed", dict(base_meta, tick=t, phase=phases[t], query_assembly="gripper", value=vals,
                               note="arrows drawn from the TCP site to tcp + label (the label's own vector)"), records)

    # ---- observed_effect (entity, gauss): realized displacement over the chunk, slot_pos[t+H-1] - slot_pos[t] (base frame)
    t = t_lift - 2
    L, _ = arm_labels_at(samples[t], prv)
    t_end = min(t + ARM_H - 1, len(snaps) - 1)
    d = s.data
    d.qpos[:] = snaps[t_end]
    mujoco.mj_forward(s.model, d)
    p_end = {sl: slot_world(s, sl) for sl in slots}
    sc = arm_render_at(s, snaps, t, cam_h)
    p_now = {sl: slot_world(s, sl) for sl in slots}
    P = MF.Panel(0, "probe.arm.observed_effect", "x", GEOM, "", r"$p_e(t{+}H{-}1)-p_e(t)$, H=16", BADGE["gt"], f"{tag} t={t}", sc["rgb"])
    vals = {}
    for k, sl in enumerate(slots):
        fd = L["future_disp"][k]
        a_, b_ = MF.uv_of(sc, p_now[sl])[0], MF.uv_of(sc, p_end[sl])[0]
        if np.linalg.norm(fd) > 0.01:
            line_arrow(P, a_, b_, color=GEOM, lw=2.0)
            MF.dot(P, b_, "#FFFFFF", r=3.5)
        if MF.in_frame(a_, 20):
            MF.text(P, a_ + np.array([40, 22]), f"{SLOT_SHORT[sl]} |d|={np.linalg.norm(fd)*100:.1f} cm", size=0.8)
        vals[sl] = fd.tolist()
    save_tile(P, caption_of("probe.arm.observed_effect", "readout of each entity's realized displacement over the 16-tick chunk (future_disp; cube lifted, others 0)",
                            "readout (inert), address entity, gauss", "gt sim truth (future state), training-only"),
              "computed", dict(base_meta, tick=t, tick_end=t_end, phase=phases[t], value_base_frame=vals,
                               note="arrow = world positions at t and t+H-1 (same rigid displacement as the base-frame label)"), records)

    # ---- subtask (asm, ce 12): operator of the first ACTIVE event
    t = t_tr
    L, OPS = arm_labels_at(samples[t], prv)
    sc = arm_render_at(s, snaps, t, cam_wide)
    P = MF.Panel(0, "probe.arm.subtask", "x", VIOLET, "", r"$\arg\,\mathrm{op}(\mathrm{active\ event})$", BADGE["public_label"],
                 f"{tag} t={t}", sc["rgb"])
    MF.ring(P, MF.palm_uv(s, sc), r=16)
    ops_shown = OPS[:12]
    for k, op in enumerate(ops_shown):
        on = (k == L["subtask"])
        MF.text(P, (16, 70 + 24 * k), f"{k:>2} {op}", color=(VIOLET if on else "#8A8F98"), size=(1.05 if on else 0.85), ha="left")
    MF.text(P, MF.palm_uv(s, sc) + np.array([0, -32]), f"subtask = {L['subtask']} '{L['subtask_name']}'", size=1.0)
    seq = [OPS[int(arm_labels_at(samples[q], prv)[0]['subtask'])] for q in range(0, len(samples), 1)]
    runs = [seq[0]] + [b for a, b in zip(seq, seq[1:]) if a != b]
    save_tile(P, caption_of("probe.arm.subtask", f"readout of the active task operator for the assembly (12-way; this episode: {' -> '.join(runs)})",
                            "readout (inert), address assembly, ce", "public label (task runtime: first ACTIVE event's operator)"),
              "computed", dict(base_meta, tick=t, phase=phases[t], value=int(L["subtask"]), value_name=L["subtask_name"], episode_sequence=runs,
                               note="derived.active_operator(pi); OPERATORS[:12] shown (the head is 12-way)"), records)

    # ---- goal_effect (entity, gauss): pos[destination] - pos[patient] for open events (public spec + estimates)
    t = t_pre
    L, _ = arm_labels_at(samples[t], prv)
    sc = arm_render_at(s, snaps, t, cam_wide)
    P = MF.Panel(0, "probe.arm.goal_effect", "x", VIOLET, "", r"$\hat p_{\rm dst}-\hat p_{\rm patient}$ (open events)", BADGE["public_label"],
                 f"{tag} t={t}", sc["rgb"])
    ge = L["goal_effect"]
    vals = {}
    for k, sl in enumerate(slots):
        a_ = MF.uv_of(sc, slot_world(s, sl))[0]
        if np.linalg.norm(ge[k]) > 1e-4:
            b_ = MF.uv_of(sc, slot_world(s, "target_zone"))[0]
            line_arrow(P, a_, b_, color=VIOLET, lw=2.2)
            MF.text(P, (a_ + b_) / 2 + np.array([0, -18]), f"({ge[k][0]:+.2f},{ge[k][1]:+.2f},{ge[k][2]:+.2f}) m", size=0.85)
        else:
            MF.text(P, a_ + np.array([0, -22]), f"{SLOT_SHORT[sl]} 0", size=0.85)
        vals[sl] = ge[k].tolist()
    save_tile(P, caption_of("probe.arm.goal_effect", "readout of the displacement the task asks of each entity: destination minus patient for open events (cube -> target; others 0)",
                            "readout (inert), address entity, gauss", "public label (task spec + public pose estimates)"),
              "computed", dict(base_meta, tick=t, phase=phases[t], value_base_frame=vals,
                               note="latent_batch.goal_effect_from_batch on collate_samples([sample]); arrow drawn cube -> target_zone (sim poses), numbers = label (base frame, public estimates)"), records)


# ------------------------------------------------------------------------------------------------ ComputerWorld (UI + pointer)
def cw_env(task):
    from rrp.envs.computerworld import make_env
    return make_env(task=task, seed=SEED)


def cw_table(env):
    from rrp.envs.computerworld import scene_widgets
    return env.slots.assign(scene_widgets(env.scene()))


class Crop:
    """4:3 crop of the 960x640 desktop frame resized to the 600x450 tile; maps frame px -> tile px."""

    def __init__(self, img, x0, y0, w):
        from PIL import Image
        h = int(round(w * 0.75))
        x0 = int(min(max(x0, 0), img.shape[1] - w))
        y0 = int(min(max(y0, 0), img.shape[0] - h))
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.s = W / w
        self.img = np.asarray(Image.fromarray(img[y0:y0 + h, x0:x0 + w]).resize((W, H), Image.LANCZOS))

    def pt(self, u, v):
        return np.array([(u - self.x0) * self.s, (v - self.y0) * self.s])

    def box(self, b):
        a = self.pt(b[0], b[1])
        c = self.pt(b[2], b[3])
        return (a[0], a[1], c[0], c[1])

    def visible(self, b):
        x0, y0, x1, y1 = self.box(b)
        return x1 > 2 and y1 > 2 and x0 < W - 2 and y0 < H - 2


def ui_contrib(name, table, extra_specs=None):
    """[n,n] value of one ui.* factor from the REAL FactorSite on the public widget table (weights set to 1)."""
    from rrp.envs.computerworld import UI_REL_VOCAB, ui_edges, ui_public_fields
    from rrp.policies.relations.base import EdgeSet, RelCtx, TokenSet, resolve
    from rrp.policies.relations.ops import FactorSite
    E = ui_edges(table)
    fld = ui_public_fields(table)
    n = len(table)
    f = {"zlayer": torch.as_tensor(np.asarray(fld["zlayer"], np.float32)).view(1, n, 1),
         "parent_id": torch.as_tensor(np.asarray(fld["parent_id"], np.float32)).view(1, n, 1)}
    ts = TokenSet("ui", torch.ones(1, n, dtype=torch.bool), fields=f)
    rc = RelCtx(sets={"ui": ts}, edges={"ui>ui": EdgeSet(UI_REL_VOCAB, torch.as_tensor(E, dtype=torch.bool)[None])})
    site = FactorSite(1, 4, "ui>ui", resolve([name]), ("edges:ui-rel-v1", "zlayer", "parent_id"))
    assert [x.name for x in site.specs] == [name], site.specs
    with torch.no_grad():
        if site.w is not None:
            site.w.fill_(1.0)
        for mod in site.f.values():
            if mod is not None and hasattr(mod, "w"):
                mod.w.fill_(1.0)
    v = site.contributions(rc, torch.zeros(1, n, 4), torch.zeros(1, n, 4))[name][0, 0].numpy()
    return v, E, fld


def short(w, k=18):
    s_ = w.get("label") or w.get("role")
    return s_ if len(s_) <= k else s_[:k - 1] + "…"


def build_ui(records):
    from rrp.envs.computerworld import UI_REL_VOCAB
    tag = "RESET STATE (no controller)"
    base = dict(env="computerworld (cw-site)", seed=SEED, tick=0, scene_source="rendered ComputerWorld desktop at reset; no teacher, no policy",
                code_path="FactorSite([factor]).contributions on RelCtx(ui_public_fields, ui_edges) of the public widget table; weights = 1")
    env = cw_env("cw/fill_form")
    table = cw_table(env)
    img = np.asarray(env.render())[..., :3]
    idx = {w["key"]: i for i, w in enumerate(table)}
    form = Crop(img, 100, 140, 440)

    # ---- ui.label_for: query = the 'Email' label
    v, E, fld = ui_contrib("ui.label_for", table)
    qi = next(i for i, w in enumerate(table) if w["role"] == "label" and w["label"] == "Email")
    P = MF.Panel(0, "ui.label_for", "x", GRAPH, "", r"$\mathrm{label\_for}(i,j)$", BADGE["public"], tag, form.img)
    rect(P, form.box(table[qi]["box"]), "#E4572E", lw=2.4, z=6)
    MF.text(P, form.pt(table[qi]["box"][0], table[qi]["box"][3]) + np.array([150, 60]), f"query i: label '{table[qi]['label']}' (red)", size=0.9)
    hits = np.nonzero(v[qi])[0]
    for j in hits:
        rect(P, form.box(table[j]["box"]), GRAPH, fill=True, alpha=0.40)
        b = form.box(table[j]["box"])
        MF.text(P, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), "1", size=1.3)
        line_arrow(P, form.pt(table[qi]["box"][2], (table[qi]["box"][1] + table[qi]["box"][3]) / 2), (b[0], (b[1] + b[3]) / 2), color=GRAPH, lw=1.6)
    MF.text(P, (300, 420), f"row i: 1 at {len(hits)} widget, 0 at the other {len(table) - 1 - len(hits)}", size=0.9)
    save_tile(P, caption_of("ui.label_for", "public UI edge: a role=label widget -> the next widget after it in the same window (scene order); here 'Email' -> its textbox",
                            "op edge, form bias (edges:ui-rel-v1 channel label_for)", "public (given widget tree)"),
              "computed", dict(base, task="cw/fill_form", query=dict(slot=qi, **{k: table[qi][k] for k in ("role", "label", "box")}),
                               partners=[dict(slot=int(j), label=table[j]["label"], role=table[j]["role"]) for j in hits],
                               row_stats=dict(min=float(v[qi].min()), max=float(v[qi].max()), nnz=int((v[qi] != 0).sum()))), records)

    # ---- ui.focus_next: query = the Name textbox
    v, E, fld = ui_contrib("ui.focus_next", table)
    qi = next(i for i, w in enumerate(table) if w["role"] == "textbox" and w["label"] == "Name")
    P = MF.Panel(0, "ui.focus_next", "x", GRAPH, "", r"$\mathrm{focus\_next}(i,j)$", BADGE["public"], tag, form.img)
    rect(P, form.box(table[qi]["box"]), "#E4572E", lw=2.4, z=6)
    MF.text(P, form.pt(table[qi]["box"][2], table[qi]["box"][1]) + np.array([110, 8]), "query i: Name textbox", size=0.9)
    hits = np.nonzero(v[qi])[0]
    for j in hits:
        b = form.box(table[j]["box"])
        rect(P, b, GRAPH, fill=True, alpha=0.40)
        MF.text(P, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), "1", size=1.3)
        qb = form.box(table[qi]["box"])
        line_arrow(P, (qb[2] + 4, (qb[1] + qb[3]) / 2), (b[2] + 4, (b[1] + b[3]) / 2), color="#E4572E", lw=1.8)
    # the public tab order around the query (context only, thin)
    order, j = [qi], qi
    for _ in range(3):
        nx = np.nonzero(E[j, :, UI_REL_VOCAB.index("focus_next")])[0]
        if not len(nx):
            break
        j = int(nx[0])
        order.append(j)
    MF.text(P, (300, 400), "tab order: " + " -> ".join(short(table[k], 12) for k in order), size=0.85)
    MF.text(P, (300, 425), "(cyclic over focusable, enabled, boxed widgets)", size=0.75)
    save_tile(P, caption_of("ui.focus_next", "public UI edge: i, j consecutive in the tab order (focusable, enabled widgets in scene order, cyclic); Name -> Email",
                            "op edge, form bias (edges:ui-rel-v1 channel focus_next)", "public (given widget tree)"),
              "computed", dict(base, task="cw/fill_form", query=dict(slot=qi, label=table[qi]["label"], role=table[qi]["role"]),
                               partners=[int(j) for j in hits], tab_order_from_query=[dict(slot=k, label=table[k]["label"]) for k in order]), records)

    # ---- ui.same_window: query = the Name textbox; every widget coloured by 1[parent_id equal]
    v, E, fld = ui_contrib("ui.same_window", table)
    full = Crop(img, 60, 0, 853)
    P = MF.Panel(0, "ui.same_window", "x", GRAPH, "", r"$\mathbb{1}[\mathrm{parent}_i=\mathrm{parent}_j]$", BADGE["public"], tag, full.img)
    for j, w in enumerate(table):
        if w["box"] is None or j == qi or not full.visible(w["box"]):
            continue
        b = full.box(w["box"])
        big = (b[2] - b[0]) * (b[3] - b[1]) > 0.25 * W * H
        if v[qi, j] > 0:
            rect(P, b, GRAPH, lw=1.2 if big else 1.4, fill=not big, alpha=0.35, z=3)
        else:
            rect(P, b, "#E8E2D0", lw=0.9, z=2)
    rect(P, full.box(table[qi]["box"]), "#E4572E", lw=2.4, z=6)
    MF.text(P, (300, 330), f"query i: Name textbox (window {int(fld['parent_id'][qi])})", size=0.9)
    MF.text(P, (300, 352), f"dark = 1 ({int((v[qi] > 0).sum())} widgets of window 0)", size=0.8)
    MF.text(P, (300, 372), "light outline = 0 (shell items, parent -1)", size=0.8)
    save_tile(P, caption_of("ui.same_window", "1 when two widgets belong to the same window (parent_id equality; shell items such as the tab strip have parent -1 and never match)",
                            "op same over field parent_id, form bias", "public (given widget tree)"),
              "computed", dict(base, task="cw/fill_form", query=dict(slot=qi, label=table[qi]["label"]),
                               n_same=int((v[qi] > 0).sum()), n_widgets=len(table),
                               same=[int(j) for j in np.nonzero(v[qi])[0]]), records)

    # ---- ui.above (drag_window scene, layered): query = calculator key '5' (zlayer 2)
    env2 = cw_env("cw/drag_window")
    t2 = cw_table(env2)
    img2 = np.asarray(env2.render())[..., :3]
    v, E, fld = ui_contrib("ui.above", t2)
    qi = next(i for i, w in enumerate(t2) if w["label"] == "5")
    full2 = Crop(img2, 60, 0, 853)
    P = MF.Panel(0, "ui.above", "x", GRAPH, "", r"$\mathrm{sign}(z_j-z_i)$", BADGE["public"], tag, full2.img)
    for j, w in enumerate(t2):
        if w["box"] is None or j == qi or not full2.visible(w["box"]):
            continue
        b = full2.box(w["box"])
        big = (b[2] - b[0]) * (b[3] - b[1]) > 0.2 * W * H
        val = v[qi, j]
        col = GRAPH if val > 0 else ("#C9CDD3" if val < 0 else None)
        if col is None:
            continue
        rect(P, b, col, lw=1.4 if val > 0 else 1.0, fill=(not big) and val > 0, alpha=0.45, z=3 if val > 0 else 2)
        if not big and (b[2] - b[0]) > 30 and (b[3] - b[1]) > 16:
            MF.text(P, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), f"{val:+.0f}", size=0.8)
    bt = full2.box(t2[idx2]["box"]) if (idx2 := next(i for i, w in enumerate(t2) if w["key"].startswith("window:0:drag"))) is not None else None
    MF.text(P, ((bt[0] + bt[2]) / 2, (bt[1] + bt[3]) / 2), f"title bar {v[qi, idx2]:+.0f} (layer {int(fld['zlayer'][idx2])})", size=0.85)
    rect(P, full2.box(t2[qi]["box"]), "#E4572E", lw=2.4, z=6)
    MF.text(P, full2.pt(*t2[qi]["box"][:2]) + np.array([40, -10]), f"query i: key '5' (layer {int(fld['zlayer'][qi])})", size=0.9)
    zs = {int(z): int(((fld["zlayer"] == z)).sum()) for z in np.unique(fld["zlayer"])}
    save_tile(P, caption_of("ui.above", "signed render order: +1 if widget j is drawn above i (higher dense z-layer), -1 below, 0 same layer; dark = +1 (resize handles, shell), light = -1 (window frame, desktop icon)",
                            "op order over field zlayer, form bias (antisymmetric)", "public (given widget tree)"),
              "computed", dict(base, task="cw/drag_window", query=dict(slot=qi, label=t2[qi]["label"], zlayer=float(fld["zlayer"][qi])),
                               counts=dict(plus=int((v[qi] > 0).sum()), minus=int((v[qi] < 0).sum()), zero=int((v[qi] == 0).sum()) - 1),
                               widgets_per_layer=zs), records)

    # ---- ui.drag_to (gt label, relgen.ui.drag_to_fn with the teacher's drag)
    from rrp.harness.data.relgen import LABELS, TokenIndex
    from rrp.harness.data.relgen.ui import teacher_drag_view
    view = teacher_drag_view(env2)
    ids = [view.token_entity("widgets", sl) for sl in range(len(t2))]
    lab = LABELS["drag_to"].fn(view, TokenIndex(sets={"ctx": ids}))
    val, valid = np.asarray(lab.value)[..., 0], np.asarray(lab.valid)
    src, dst = [int(x) for x in np.argwhere(val > 0.5)[0]]
    drag = view.teacher_drag()
    assert (drag["src_slot"], drag["dst_slot"]) == (src, dst), (drag, src, dst)
    P = MF.Panel(0, "ui.drag_to", "x", GRAPH, "", r"$\mathrm{drag\_to}(i,j)$", BADGE["gt_probe"], "RESET + TEACHER GOAL (privileged)", full2.img)
    bs, bd = full2.box(t2[src]["box"]), full2.box(t2[dst]["box"])
    rect(P, bs, "#E4572E", lw=2.4, z=6)
    rect(P, bd, GRAPH, lw=1.8, fill=True, alpha=0.5, z=5)
    MF.text(P, (300, 250), f"j = drop target '{short(t2[dst], 26)}' (filled): value 1", size=0.85)
    cs = full2.pt((t2[src]["box"][0] + t2[src]["box"][2]) / 2, (t2[src]["box"][1] + t2[src]["box"][3]) / 2)
    dp = full2.pt(*drag["dest_px"])
    line_arrow(P, cs, dp, color="#E4572E", lw=1.8)
    MF.dot(P, dp, "#FFFFFF", r=4)
    MF.text(P, dp + np.array([60, 40]), f"drag end (dx={env2.goal['dx']:+d}, dy={env2.goal['dy']:+d} px)", size=0.8)
    MF.line(P, [dp, ((bd[0] + bd[2]) / 2, bd[3])], color=GRAPH, lw=1.0)
    MF.text(P, (300, 274), "i = title-bar drag handle (red, query)", size=0.85)
    save_tile(P, caption_of("ui.drag_to", "pair label: the teacher's drag handle -> the nearest widget outside the dragged window to where the drag ends (cw/drag_window); 1 pair of "
                            f"{int(valid.sum())} valid", "op bilinear, form aug (the bias IS the pair probe)", "gt label (teacher goal), training-only; deployed = estimated-probe"),
              "computed", dict(base, task="cw/drag_window", goal=env2.goal, drag=drag, n_valid_pairs=int(valid.sum()), n_true=int((val > 0.5).sum()),
                               note="relgen.ui.drag_to_fn via teacher_drag_view(env); the deployed term is the learned bilinear probe (NOT computed here)"), records)


def build_pointer(records):
    """Scripted CWTeacher on cw/fill_form, labels exactly as harness.train.pointer.collect.DemoTeacher / data.Demos."""
    from rrp.harness.train.pointer.collect import phase_of
    from rrp.policies.pointer.spec import PHASES
    from rrp.policies.teachers.computerworld import CWTeacher
    LATE = (0, 2, 4, 6)
    NW = 80
    env = cw_env("cw/fill_form")
    tt = CWTeacher(env, "cw/fill_form")
    fr = env.frame
    half = np.array([fr.width / 2 * fr.m_per_px, fr.height / 2 * fr.m_per_px])
    rows, frames, prev_btn = [], [], False
    for tick in range(200):
        q = np.array(fr.px_to_m(env.pointer.u, env.pointer.v))
        frames.append((np.asarray(env.render())[..., :3].copy(), (env.pointer.u, env.pointer.v), cw_table(env)))
        c = tt.act()
        if c is None:
            break
        g = {k: list(v) for k, v in c.groups.items()}
        ph = phase_of(g, prev_btn, q)
        tslot = env.slots.slots.get(tt.target, -1) if tt.target else -1
        tpx = tt.target_px
        rows.append(dict(ptr=q / half, slot=tslot if tslot < NW else -1, txy=(np.array(fr.px_to_m(*tpx)) / half) if tpx is not None else None,
                         tpx=tpx, phase=ph, target=tt.target))
        prev_btn = g["button"][0] >= 0.5
        env.step(c)
    T = len(rows)
    # tick on the way from the Name textbox to the Email textbox, with all four knots inside the episode
    email_slot = next(sl for k, sl in env.slots.slots.items() if "content:email" in k)
    t = next(i for i in range(T - 6) if rows[i]["slot"] == email_slot and PHASES[rows[i]["phase"]] == "move")
    knots = [t + j for j in LATE]
    lab = dict(slot=[rows[k]["slot"] for k in knots], rel=[(rows[k]["txy"] - rows[t]["ptr"]).tolist() for k in knots],
               phase=[int(rows[k]["phase"]) for k in knots])
    img, (pu, pv), table = frames[t]
    table = [w for w in table if w is not None]
    key_of = {sl: k for k, sl in env.slots.slots.items()}
    crop = Crop(img, 100, 150, 400)
    tag = f"SCRIPTED TEACHER (privileged) t={t}"
    base = dict(env="computerworld (cw-site)", task="cw/fill_form", seed=SEED, tick=t, knots=knots, LATE=list(LATE), labels=lab,
                episode_ticks=T, phase_sequence=[PHASES[r["phase"]] for r in rows],
                scene_source="scripted privileged CWTeacher, stepped from reset; frame rendered before the tick's command",
                code_path="DemoTeacher row (slot = env.slots.slots[target], txy = target_px/half, phase = phase_of) -> Demos labels at t+LATE (rel = txy[t+j] - ptr[t])")
    ptr_uv = crop.pt(pu, pv)

    # slot
    P = MF.Panel(0, "probe.pointer.slot", "x", VIOLET, "", r"slot index of the step's target widget (80-way)", BADGE["gt"], tag, crop.img)
    MF.ring(P, ptr_uv, r=12)
    drawn = set()
    for k, sl in zip(knots, lab["slot"]):
        if sl < 0 or sl in drawn:
            continue
        drawn.add(sl)
        w = next(w for w in table if w["key"] == key_of.get(sl))
        b = crop.box(w["box"])
        rect(P, b, VIOLET, lw=1.8, fill=True, alpha=0.35)
    MF.text(P, (300, 360), "knots t+0,2,4,6: slot " + ", ".join(str(x) for x in lab["slot"]), size=0.95)
    MF.text(P, (300, 385), f"slot {lab['slot'][0]} = {short(next(w for w in table if w['key'] == key_of.get(lab['slot'][0])), 24)}", size=0.85)
    save_tile(P, caption_of("probe.pointer.slot", "readout of WHICH widget the current plan step targets, as its stable slot index (opaque codes only, no widget content)",
                            "readout (inert), address knot x assembly, ce over 80 slots", "gt teacher target, training-only"),
              "computed", base, records)

    # rel
    P = MF.Panel(0, "probe.pointer.rel", "x", GEOM, "", r"$xy_{\rm target}(t{+}j)-xy_{\rm ptr}(t)$ [screen/2]", BADGE["gt"], tag, crop.img)
    MF.ring(P, ptr_uv, r=12)
    for k, r_ in zip(knots, lab["rel"]):
        end = ptr_uv + np.array([r_[0] * fr.width / 2, -r_[1] * fr.height / 2]) * crop.s
        line_arrow(P, ptr_uv, end, color=GEOM, lw=1.8)
        MF.dot(P, end, "#FFFFFF", r=3.5)
    MF.text(P, (300, 360), "rel (x, y up) at knots t+0,2,4,6:", size=0.85)
    MF.text(P, (300, 385), "  ".join(f"({r_[0]:+.3f},{r_[1]:+.3f})" for r_ in lab["rel"]), size=0.75)
    save_tile(P, caption_of("probe.pointer.rel", "readout of the target point relative to the pointer, normalized screen units (x right, y up), per knot",
                            "readout (inert), address knot x assembly, gauss (2-d mean + logvar)", "gt teacher target, training-only"),
              "computed", base, records)

    # phase
    P = MF.Panel(0, "probe.pointer.phase", "x", VIOLET, "", r"phase of the commanded tick (6-way)", BADGE["gt"], tag, crop.img)
    MF.ring(P, ptr_uv, r=12)
    for i_, name in enumerate(PHASES):
        on = i_ in lab["phase"]
        MF.text(P, (470, 120 + 26 * i_), f"{i_} {name}", color=VIOLET if on else "#8A8F98", size=1.05 if on else 0.85, ha="left")
    MF.text(P, (300, 360), "knots t+0,2,4,6: " + ", ".join(PHASES[p_] for p_ in lab["phase"]), size=0.9)
    seq = base["phase_sequence"]
    runs = [seq[0]] + [b for a, b in zip(seq, seq[1:]) if a != b]
    MF.text(P, (300, 385), "episode: " + " ".join(runs[:12]) + (" …" if len(runs) > 12 else ""), size=0.7)
    save_tile(P, caption_of("probe.pointer.phase", "readout of the interaction phase of the commanded tick (idle, move, press, release, drag, type) at each knot",
                            "readout (inert), address knot x assembly, ce", "gt teacher command, training-only"),
              "computed", base, records)


# ------------------------------------------------------------------------------------------------ legged (go2 / pquad4)
def legged_episode(body, seed, sigma, task="waypoint_contact"):
    """The REAL legged latent collector (harness.data.legged_latent_collect.collect_episode: scripted waypoint teacher,
    frozen RL tracker / CPG, RecordingTracker per 50 Hz tick). RecordingTracker.act is wrapped in THIS process only to
    keep the session and a qpos snapshot per recorded tick for rendering; the recorded arrays are untouched."""
    import rrp.harness.data.legged_latent_collect as LLC
    keep = dict(q=[], s=None)
    orig = LLC.RecordingTracker.act

    def act(self, data, cmd):
        if self.rec is not None:
            keep["q"].append(data.qpos.copy())
            keep["s"] = self.s
        return orig(self, data, cmd)
    LLC.RecordingTracker.act = act
    try:
        arr, meta, morph = LLC.collect_episode(body, seed, sigma, task=task)
    finally:
        LLC.RecordingTracker.act = orig
    assert len(keep["q"]) == len(arr["contact"])
    return arr, meta, morph, keep["s"], keep["q"]


def legged_labels(arr, task, i):
    """LeggedData.labels (harness.train.legged_latent_train) for row i of one episode, in numpy (same math)."""
    from rrp.harness.train.legged_latent_train import goal_columns
    from rrp.policies.features.legged import KNOT_TICKS, H as LH
    end = len(arr["contact"]) - 1
    kt = np.minimum(i + np.asarray(KNOT_TICKS), end)
    pose = arr["pose"][i]
    c, s_ = np.cos(pose[2]), np.sin(pose[2])
    p2 = arr["pose"][min(i + LH, end)]
    ex, ey = p2[0] - pose[0], p2[1] - pose[1]
    dyaw = (p2[2] - pose[2] + np.pi) % (2 * np.pi) - np.pi
    goal, gok = goal_columns(task, arr["ctx"], arr["ev"])
    return dict(contact_k=arr["contact"][kt], knot_ticks=kt, goal=goal[i], goal_valid=bool(gok[i]),
                disp=np.array([(c * ex + s_ * ey) / 0.5, (-s_ * ex + c * ey) / 0.5, dyaw]), disp_end=min(i + LH, end),
                subtask=int(min(arr["ev"][i], 3)))


def legged_color(m):
    def f(bn, typ, g):
        if typ == mujoco.mjtGeom.mjGEOM_PLANE:
            return np.array([0.86, 0.87, 0.87, 1.0])
        if bn == "world":
            return np.array([0.78, 0.79, 0.80, 1.0])
        return np.array([0.66, 0.68, 0.71, 1.0])
    return f


def legged_render(s, qs, i, cam):
    d = s.data
    d.qpos[:] = qs[i]
    d.qvel[:] = 0
    mujoco.mj_forward(s.model, d)
    return MF.render_scene(s.model, d, cam, legged_color(s.model), min_agree=0.9)


def foot_world(s):
    return [s.data.xpos[b].copy() for b in s.binding.foot_bids]


def base_cam(arr, i, dist=1.5, az=None, el=-32, dz=0.2):
    x, y, yaw = arr["pose"][i][:3]
    return MF.Cam(lookat=[x, y, dz], dist=dist, az=np.degrees(yaw) + (az if az is not None else -130), el=el, fovy=34.0)


def yaw_to_world(pose, v):
    c, s_ = np.cos(pose[2]), np.sin(pose[2])
    return np.array([pose[0] + c * v[0] - s_ * v[1], pose[1] + s_ * v[0] + c * v[1]])


def event_names(task):
    from rrp.envs.mujoco.scenario import load_task
    from rrp.tasks.spec import get_task
    g = load_task(get_task(task).graph)
    return [e.get("id", f"event{k}") for k, e in enumerate(g["events"])]


def build_legged(records):
    from rrp.policies.features.legged import KNOT_TIMES
    from rrp.harness.data.legged_latent_collect import foothold_cells  # noqa: F401  (the label used below is arr['foothold_cell'])
    from rrp.policies.nets.legged_latent import scan_xy
    task = "waypoint_contact"
    arr, meta, morph, s, qs = legged_episode("go2", SEED, 0.0, task)
    T = len(qs)
    nf = arr["contact"].shape[1]
    feet = list(s.robots[0].meta["legged"]["foot_bodies"]) if "legged" in s.robots[0].meta else [f"foot{k}" for k in range(nf)]
    evn = event_names(task) + ["done"]
    tag = "SCRIPTED TEACHER + RL TRACKER"
    base = dict(env="mujoco/legged", body="go2", task=task, seed=SEED, sigma=0.0, status=meta["status"], ticks=T,
                tracker=meta.get("tracker_source"), teacher=meta.get("teacher"), waypoints=meta.get("waypoints"),
                scene_source="harness.data.legged_latent_collect.collect_episode (scripted waypoint teacher -> frozen go2 RL tracker), qpos replayed per 50 Hz tick",
                label_code="legged_latent_train.LeggedData.labels math (numpy) on the collector arrays; goal = goal_columns(public context)")
    wp = meta.get("waypoints") or {}

    def wp_marks(P, sc):
        for k_, xy in wp.items():
            uv = MF.uv_of(sc, [xy[0], xy[1], 0.0])[0]
            if MF.in_frame(uv, 12):
                MF.dot(P, uv, "#FFFFFF", r=5)
                MF.text(P, uv + np.array([0, -16]), f"waypoint {k_}", size=0.8)

    # ---- probe.legged.contact (knot x asm, bce): per foot contact at t + 0.1/0.3/0.5/0.7 s
    i = next(t for t in range(40, T) if arr["contact"][t].sum() == 2 and arr["ev"][t] == 0)
    L = legged_labels(arr, task, i)
    sc = legged_render(s, qs, i, base_cam(arr, i, dist=1.7, az=-105, el=-20, dz=0.18))
    P = MF.Panel(0, "probe.legged.contact", "x", AMBER, "", r"$c_{f}(t+\tau_k)$, $\tau$=0.1,0.3,0.5,0.7 s", BADGE["gt"], f"{tag} t={i}", sc["rgb"])
    fw = foot_world(s)
    for f in range(nf):
        uv = MF.uv_of(sc, fw[f])[0]
        MF.dot(P, uv, AMBER if arr["contact"][i, f] else "#FFFFFF", r=5)
        MF.text(P, uv + np.array([0, 22]), " ".join(str(int(x)) for x in L["contact_k"][:, f]), size=0.95)
    MF.text(P, (300, 395), "per foot: contact at the 4 knots (1 = on ground); dot = contact now", size=0.8)
    save_tile(P, caption_of("probe.legged.contact", "readout of each foot's ground contact at the 4 packet knots (+0.1/0.3/0.5/0.7 s) of a trot",
                            "readout (inert), address knot x assembly (feet), bce", "gt sim truth (future contacts), training-only"),
              "computed", dict(base, tick=i, feet=feet, contact_k=L["contact_k"].astype(int).tolist(), knot_ticks=L["knot_ticks"].tolist(), knot_times=list(KNOT_TIMES)), records)

    # ---- goal (asm, gauss 2-d): active event's goal entity, body yaw frame / 2 m, from the PUBLIC context
    i = 60
    L = legged_labels(arr, task, i)
    pose = arr["pose"][i]
    g_w = yaw_to_world(pose, 2.0 * L["goal"])
    mid = 0.62 * np.array(pose[:2]) + 0.38 * g_w
    cam_g = MF.Cam(lookat=[mid[0], mid[1], 0.0], dist=1.2 * np.linalg.norm(g_w - pose[:2]) + 2.6, az=np.degrees(pose[2]) - 150, el=-58, fovy=34.0)
    sc = legged_render(s, qs, i, cam_g)
    P = MF.Panel(0, "probe.legged.goal", "x", VIOLET, "", r"goal$_{\rm active}$ in body yaw frame / 2 m", BADGE["public_label"], f"{tag} t={i}", sc["rgb"])
    wp_marks(P, sc)
    g_w = yaw_to_world(pose, 2.0 * L["goal"])
    a_, b_ = MF.uv_of(sc, [pose[0], pose[1], 0.05])[0], MF.uv_of(sc, [g_w[0], g_w[1], 0.0])[0]
    line_arrow(P, a_, b_, color=VIOLET, lw=2.2)
    MF.text(P, (a_ + b_) / 2 + np.array([0, -18]), f"({L['goal'][0]:+.2f}, {L['goal'][1]:+.2f})", size=0.95)
    MF.text(P, (300, 395), f"active event: {evn[L['subtask']]}; label = public estimate (detector track - noisy localization)", size=0.72)
    save_tile(P, caption_of("probe.legged.goal", "readout of the active event's goal entity (here the waypoint) in the body yaw frame, /2 m, as the PUBLIC context estimates it",
                            "readout (inert), address assembly (body row), gauss (2-d)", "public label (public context estimate)"),
              "computed", dict(base, tick=i, goal=L["goal"].tolist(), goal_valid=L["goal_valid"], active_event=evn[L["subtask"]]), records)

    # ---- disp (asm, gauss 3-d): base displacement over H = 0.8 s, start body frame (xy / 0.5 m, dyaw rad)
    i = 150
    L = legged_labels(arr, task, i)
    pose, pose2 = arr["pose"][i], arr["pose"][L["disp_end"]]
    sc = legged_render(s, qs, i, base_cam(arr, i, dist=2.4, az=-110, el=-38, dz=0.1))
    P = MF.Panel(0, "probe.legged.disp", "x", GEOM, "", r"$R_{\psi_t}^{\top}(p_{t+H}-p_t)/0.5,\ \Delta\psi$", BADGE["gt"], f"{tag} t={i}", sc["rgb"])
    a_, b_ = MF.uv_of(sc, [pose[0], pose[1], 0.02])[0], MF.uv_of(sc, [pose2[0], pose2[1], 0.02])[0]
    line_arrow(P, a_, b_, color=GEOM, lw=2.4)
    hd = MF.uv_of(sc, [pose2[0] + 0.25 * np.cos(pose2[2]), pose2[1] + 0.25 * np.sin(pose2[2]), 0.02])[0]
    MF.line(P, [b_, hd], color="#1D1F22", lw=1.2)
    MF.text(P, b_ + np.array([90, 26]), "base at t+0.8 s (line: heading)", size=0.8)
    MF.text(P, (300, 395), f"disp = ({L['disp'][0]:+.2f}, {L['disp'][1]:+.2f}) x0.5 m, dyaw {L['disp'][2]:+.2f} rad", size=0.9)
    save_tile(P, caption_of("probe.legged.disp", "readout of the base displacement over the next 0.8 s in the start body frame (xy in 0.5 m units, yaw change in rad)",
                            "readout (inert), address assembly, gauss (3-d)", "gt sim truth (future base pose), training-only"),
              "computed", dict(base, tick=i, tick_end=L["disp_end"], disp=L["disp"].tolist()), records)

    # ---- subtask (asm, ce 4): active event index (walk_to_a, walk_to_b, halt, done)
    i = next(t for t in range(T) if arr["ev"][t] == 2) + 10
    L = legged_labels(arr, task, i)
    sc = legged_render(s, qs, i, base_cam(arr, i, dist=2.6, az=-140, el=-36, dz=0.0))
    P = MF.Panel(0, "probe.legged.subtask", "x", VIOLET, "", r"index of the active task event (4-way)", BADGE["public_label"], f"{tag} t={i}", sc["rgb"])
    wp_marks(P, sc)
    for k_, nm in enumerate(evn[:4]):
        on = k_ == L["subtask"]
        MF.text(P, (16, 80 + 26 * k_), f"{k_} {nm}", color=VIOLET if on else "#8A8F98", size=1.05 if on else 0.85, ha="left")
    runs = []
    for e in arr["ev"]:
        if not runs or runs[-1][0] != int(e):
            runs.append([int(e), 0])
        runs[-1][1] += 1
    MF.text(P, (300, 395), "episode: " + " | ".join(f"{evn[min(e, 3)]} {n * 0.02:.1f}s" for e, n in runs), size=0.72)
    save_tile(P, caption_of("probe.legged.subtask", "readout of which task event is active (walk_to_a, walk_to_b, halt, done) from the public task runtime",
                            "readout (inert), address assembly, ce", "public label (task runtime)"),
              "computed", dict(base, tick=i, subtask=L["subtask"], name=evn[L["subtask"]], runs=runs), records)

    # ---- leg.com_support (asm readout, gauss 1-d): signed margin of the COM (IMU site) xy inside the stance-foot hull
    cands = [t for t in range(30, T) if arr["contact"][t].sum() == 3]
    i = cands[0] if cands else next(t for t in range(30, T) if arr["contact"][t].sum() == 4)
    sc = legged_render(s, qs, i, base_cam(arr, i, dist=1.9, az=-70, el=-62, dz=0.0))
    margin = float(arr["com_support"][i])
    sid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_SITE, s.robots[0].tcp_sites["body"])
    com = s.data.site_xpos[sid].copy()
    fw = foot_world(s)
    st = [f for f in range(nf) if arr["contact"][i, f]]
    P = MF.Panel(0, "leg.com_support", "x", AMBER, "", r"signed dist(COM$_{xy}$, hull(stance feet))", BADGE["gt"], f"{tag} t={i}", sc["rgb"])
    pts = [np.array([fw[f][0], fw[f][1], 0.0]) for f in st]
    if len(pts) >= 3:
        P2 = np.array([p[:2] for p in pts])
        c0 = P2.mean(0)
        order = np.argsort(np.arctan2(P2[:, 1] - c0[1], P2[:, 0] - c0[0]))
        poly = [MF.uv_of(sc, pts[k])[0] for k in order]
        MF.line(P, poly + [poly[0]], color="#B87414", lw=2.0)
    for f in range(nf):
        MF.dot(P, MF.uv_of(sc, [fw[f][0], fw[f][1], 0.0])[0], AMBER if f in st else "#FFFFFF", r=5)
    cu = MF.uv_of(sc, [com[0], com[1], 0.0])[0]
    MF.line(P, [MF.uv_of(sc, com)[0], cu], color="#1D1F22", lw=0.8)
    MF.ring(P, cu, r=8)
    MF.text(P, cu + np.array([0, -24]), f"margin {margin * 100:+.1f} cm", size=1.0)
    MF.text(P, (300, 395), f"{len(st)} stance feet (filled); ring = COM (IMU site) projected; >0 inside", size=0.78)
    save_tile(P, caption_of("leg.com_support", "readout of the stability margin: signed planar distance of the COM projection to the stance-foot support polygon (m, + inside)",
                            "readout (inert), address assembly, gauss (1-d)", "gt sim truth (contacts), training-only"),
              "computed", dict(base, tick=i, margin_m=margin, stance_feet=[feet[f] if f < len(feet) else f for f in st],
                               note="value = legged_collect.com_support(s) recorded per tick (= relgen.body.support_polygon_margin)"), records)

    # ---- leg.foothold (pair probe, gt label foothold_cell): swing foot -> scan cell of its next touchdown
    fh = arr["foothold_cell"]
    i = next(t for t in range(40, T) if (fh[t, :nf] >= 0).sum() >= 2 and arr["ev"][t] == 0)
    pose = arr["pose"][i]
    sc = legged_render(s, qs, i, base_cam(arr, i, dist=2.0, az=-150, el=-58, dz=0.0))
    P = MF.Panel(0, "leg.foothold", "x", AMBER, "", r"foothold(foot$_f$, cell$_c$)", BADGE["gt_probe"], f"{tag} t={i}", sc["rgb"])
    cells = scan_xy().numpy()
    for k_, off in enumerate(cells):
        w_ = yaw_to_world(pose, off)
        MF.dot(P, MF.uv_of(sc, [w_[0], w_[1], 0.0])[0], "#C9CDD3", r=1.6, edge="#8A8F98")
    fw = foot_world(s)
    pairs = []
    for f in range(nf):
        c_ = int(fh[i, f])
        uvf = MF.uv_of(sc, fw[f])[0]
        if c_ >= 0:
            w_ = yaw_to_world(pose, cells[c_])
            uvc = MF.uv_of(sc, [w_[0], w_[1], 0.0])[0]
            half_ = 0.05
            corners = [yaw_to_world(pose, cells[c_] + np.array(d_)) for d_ in ((-half_, -half_), (half_, -half_), (half_, half_), (-half_, half_))]
            poly = [MF.uv_of(sc, [q[0], q[1], 0.0])[0] for q in corners]
            MF.line(P, poly + [poly[0]], color="#B87414", lw=2.0)
            line_arrow(P, uvf, uvc, color="#B87414", lw=1.6)
            pairs.append(dict(foot=feet[f] if f < len(feet) else f, cell=c_, ix=c_ // 7, iy=c_ % 7))
        else:
            MF.dot(P, uvf, AMBER, r=4)
    MF.text(P, (300, 345), "grey: 11x7 terrain-scan cells (0.1 m, body yaw frame)", size=0.8)
    MF.text(P, (300, 368), f"{len(pairs)} swing feet -> next-touchdown cell (=1); stance feet (dots): no pair", size=0.8)
    save_tile(P, caption_of("leg.foothold", "pair label: each swinging foot -> the terrain-scan cell containing its next touchdown (hindsight); planted feet have no true pair",
                            "op bilinear, form aug (the bias IS the pair probe)", "gt label (future touchdown), training-only; deployed = estimated-probe"),
              "computed", dict(base, tick=i, pairs=pairs, note="arr['foothold_cell'] = legged_latent_collect.foothold_cells (the training form one-hot onto the 77 scan-cell ctx tokens by nets.legged_latent.legged_graph); terrain flat (waypoint_contact)"), records)

    # ---- fall (asm, bce): pquad4 (CPG controller) episode that the dataset records as fell (seed 11, sigma 0.3)
    arr2, meta2, morph2, s2, qs2 = legged_episode("pquad4", 11, 0.3, task)
    T2 = len(qs2)
    fell = meta2["status"] == "fell"
    from rrp.policies.features.legged import H as LH
    lab = np.array([fell and (T2 - 1 - t) <= LH for t in range(T2)])
    i = T2 - 1 - LH // 2 if fell else T2 // 2
    sc = legged_render(s2, qs2, i, base_cam(arr2, i, dist=1.4, az=-110, el=-25, dz=0.1))
    P = MF.Panel(0, "probe.legged.fall", "x", AMBER, "", r"$\mathbb{1}[\mathrm{fell}]\cdot\mathbb{1}[T_{\rm end}-t\leq 0.8\,\mathrm{s}]$", BADGE["gt"],
                 f"SCRIPTED TEACHER + CPG t={i}", sc["rgb"])
    x0, x1, y0 = 40, 560, 380
    P.marks.append(dict(kind="rect", x0=x0, y0=y0, x1=x1, y1=y0 + 12, color="#8A8F98", lw=0.8, z=6))
    a = x0 + (x1 - x0) * (np.argmax(lab) if lab.any() else T2) / T2
    if lab.any():
        P.marks.append(dict(kind="rect", x0=a, y0=y0, x1=x1, y1=y0 + 12, color="#B87414", lw=0.8, fill=True, fc=AMBER, alpha=0.9, z=6))
    MF.line(P, [(x0 + (x1 - x0) * i / T2, y0 - 6), (x0 + (x1 - x0) * i / T2, y0 + 18)], color="#E4572E", lw=2.0)
    MF.text(P, (300, y0 - 18), f"fall label over the episode ({T2} ticks, status '{meta2['status']}'); red = shown tick, label {int(lab[i])}", size=0.75)
    status = "computed" if fell else "illustrative"
    save_tile(P, caption_of("probe.legged.fall", "readout of an imminent fall: 1 in the last 0.8 s of an episode that ends in a fall (non-foot ground contact, low base or tilt)",
                            "readout (inert), address assembly, bce", "gt sim truth (episode outcome), training-only")
              + ("" if fell else " Illustrative: the re-collected episode did not fall on this code version."),
              status, dict(base, body="pquad4", seed=11, sigma=0.3, tracker=meta2.get("tracker_source"), status=meta2["status"], ticks=T2, tick=i,
                           fall_label_ticks=int(lab.sum()),
                           note="episode chosen because artifacts/datasets/legged_latent_v1/pquad4/s0-99.json records seed 11 sigma 0.3 as status 'fell'; re-collected here"), records)


# ------------------------------------------------------------------------------------------------ psi0 (recorded SIMPLE replays)
PSI_TASK = "G1WholebodyXMovePickTeleop-v0"
PSI_EP = 0


def psi_frame(task, ep, t):
    """Frame t of the recorded SIMPLE egocentric video (50 fps, one frame per label tick), decoded with PyAV from the
    isolated psi venv (the repo venv has no video decoder); cached under tiles/cache."""
    import subprocess
    from PIL import Image
    cache = WORK / "psi_frames" / f"{task}_ep{ep:06d}_t{t:04d}.png"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        vid = PSI_DATA / task / "videos" / "chunk-000" / "egocentric" / f"episode_{ep:06d}.mp4"
        code = ("import av,sys\nc=av.open(sys.argv[1])\nfor i,f in enumerate(c.decode(video=0)):\n"
                "    if i==int(sys.argv[2]):\n        f.to_image().save(sys.argv[3]); break\n")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        subprocess.run([str(PSI_PY), "-c", code, str(vid), str(t), str(cache)], check=True, env=env)
    im = Image.open(cache).convert("RGB")
    w_, h_ = im.size                                            # 640x360 -> centre 4:3 crop -> 600x450
    cw = int(h_ * 4 / 3)
    x0 = (w_ - cw) // 2
    return np.asarray(im.crop((x0, 0, x0 + cw, h_)).resize((W, H), Image.LANCZOS))


class Inset:
    """Top-down robot-frame plot (x forward = up, y left = left) drawn as Panel marks in a box of the tile."""

    def __init__(self, P, box=(330, 90, 590, 330), half=0.6, title="robot frame at t (top view)"):
        self.P, self.box, self.half = P, box, half
        x0, y0, x1, y1 = box
        P.marks.append(dict(kind="rect", x0=x0, y0=y0, x1=x1, y1=y1, color="#8A8F98", lw=0.8, fill=True, fc="#FFFFFF", alpha=0.88, z=1))
        P.marks.append(dict(kind="rect", x0=x0, y0=y0, x1=x1, y1=y1, color="#8A8F98", lw=0.8, z=2))
        MF.text(P, ((x0 + x1) / 2, y0 + 10), title, size=0.7, bg=False)
        self.c = np.array([(x0 + x1) / 2, y1 - 30])               # robot origin near the bottom of the box
        self.s = (y1 - y0 - 50) / (1.6 * half)
        o = self.c
        MF.line(P, [o, o + np.array([0, -22])], color="#1D1F22", lw=1.4, arrow=True)
        MF.dot(P, o, "#1D1F22", r=3)
        MF.text(P, o + np.array([0, 14]), "pelvis", size=0.65, bg=False)

    def uv(self, xy):
        return self.c + np.array([-xy[1], -xy[0]]) * self.s


def build_psi0(records):
    from rrp.policies.psi0.data import EpisodeLabels, CachedDataset, collate, _to_frame, _yaw
    from rrp.policies.psi0.nets import add_cmd_labels, KNOT_STEPS
    task, ep = PSI_TASK, PSI_EP
    npz = PSI_RUNS / "replay_labels" / task / f"episode_{ep:06d}.npz"
    E = EpisodeLabels(npz)
    raw = np.load(npz)
    TT = E.T
    tag = "RECORDED TELEOP REPLAY (SIMPLE)"
    base = dict(env="SIMPLE (Isaac Sim) recorded replay, read from disk", task=task, episode=ep, ticks=TT,
                label_file=str(npz), video=f"{task}/videos/chunk-000/egocentric/episode_{ep:06d}.mp4",
                scene_source="recorded egocentric frame of the teleop demonstration (no rendering, no policy here)",
                label_code="rrp.policies.psi0.data.EpisodeLabels.labels(t) on the recorded replay npz (sim truth via SIMPLE worker _sim_truth)")
    KS = list(KNOT_STEPS)

    def frame_of(t):
        o, yaw = E.pelvis[t, :3], float(_yaw(E.pelvis[t, 3:7]))
        return lambda p: _to_frame(np.asarray(p, float), o, yaw)

    def knots(t):
        return [E.at(t + k) for k in KS]

    # tick choices: hands approach / first right-hand contact; walking; lift at the end
    c_first = int(np.argmax(E.contact[:, 1] > 0))
    t_hand = max(0, c_first - 17)
    sp = np.linalg.norm(np.diff(E.pelvis[:, :2], axis=0), axis=1)
    t_walk = int(np.argmax(np.convolve(sp, np.ones(15), "same")))
    t_walk = min(max(t_walk - 10, 0), TT - 31)
    li = np.nonzero(E.target[:, 2] >= E.z0 + 0.03)[0]
    t_lift = max(0, int(li[0]) - 17) if len(li) else TT - 30

    def panel(name, color, formula, badge, t):
        return MF.Panel(0, name, "x", color, "", formula, badge, f"{tag} t={t}", psi_frame(task, ep, t))

    def knot_table(P, rows, x=16, y=300, title=""):
        MF.text(P, (x, y - 24), title, size=0.8, ha="left")
        MF.text(P, (x, y), "knot  " + "  ".join(f"+{k:<4d}" for k in KS), size=0.75, ha="left")
        for r_, (lab_, vals) in enumerate(rows):
            MF.text(P, (x, y + 20 * (r_ + 1)), f"{lab_:<5s} " + "  ".join(f"{v:<5s}" for v in vals), size=0.75, ha="left")

    # hand_dist
    t = t_hand
    L = E.labels(t)
    tf = frame_of(t)
    P = panel("probe.psi0.hand_dist", AMBER, r"$\|p_{\rm palm,h}(t{+}k)-p_{\rm target}(t{+}k)\|$", BADGE["gt"], t)
    ins = Inset(P, half=0.35)
    for k_, kk in enumerate(knots(t)):
        tg = tf(E.target[kk])
        for h, col in ((0, "#6B7078"), (1, "#B87414")):
            pa = tf(E.palm[kk, h])
            MF.line(P, [ins.uv(pa), ins.uv(tg)], color=col, lw=0.8)
            MF.dot(P, ins.uv(pa), col, r=3.5)
        MF.dot(P, ins.uv(tg), "#FFFFFF", r=4.5)
    knot_table(P, [("L", [f"{v:.2f}" for v in L["hand_dist"][:, 0]]), ("R", [f"{v:.2f}" for v in L["hand_dist"][:, 1]])], title="hand_dist [m] (L grey, R amber; white = target)")
    save_tile(P, caption_of("probe.psi0.hand_dist", "readout of each hand's palm-to-target distance at the 5 packet knots (+5..+29 ticks of 20 ms)",
                            "readout (inert), address knot x pair (pair = L/R hand), gauss", "gt sim truth (recorded SIMPLE replay), training-only"),
              "computed", dict(base, tick=t, knots=knots(t), value=L["hand_dist"].tolist()), records)

    # contact
    P = panel("probe.psi0.contact", AMBER, r"$\mathbb{1}[\mathrm{hand}_h \mathrm{\ touches\ target}](t{+}k)$", BADGE["gt"], t)
    knot_table(P, [("L", [str(int(v)) for v in L["contact"][:, 0]]), ("R", [str(int(v)) for v in L["contact"][:, 1]])], title="contact (any geom of the hand subtree vs target)")
    save_tile(P, caption_of("probe.psi0.contact", "readout of hand-target contact per hand at the 5 knots (here the right hand closes on the target)",
                            "readout (inert), address knot x pair, bce", "gt sim truth (recorded contacts), training-only"),
              "computed", dict(base, tick=t, knots=knots(t), value=L["contact"].tolist()), records)

    # active_hand
    P = panel("probe.psi0.active_hand", AMBER, r"hand bound to the target over the packet", BADGE["gt"], t)
    ah = int(L["active_hand"])
    for k_, nm in enumerate(("0 left", "1 right")):
        MF.text(P, (16, 300 + 26 * k_), nm, color="#B87414" if k_ == ah else "#8A8F98", size=1.1 if k_ == ah else 0.9, ha="left")
    MF.text(P, (16, 274), "active_hand (contact at t+29, else first future contact)", size=0.8, ha="left")
    save_tile(P, caption_of("probe.psi0.active_hand", f"readout of which hand the packet binds to the target (here {'right' if ah == 1 else 'left' if ah == 0 else 'none'}); -1 (no contact ever) is masked",
                            "readout (inert), address body (per packet), ce over 2", "gt sim truth (future contacts), training-only"),
              "computed", dict(base, tick=t, value=ah), records)

    # lift + target_pos
    t = t_lift
    L = E.labels(t)
    tf = frame_of(t)
    P = panel("probe.psi0.lift", AMBER, r"$\mathbb{1}[z_{\rm target}(t{+}k)\geq z_0+3\,\mathrm{cm}]$", BADGE["gt"], t)
    dz = [E.target[kk, 2] - E.z0 for kk in knots(t)]
    knot_table(P, [("dz cm", [f"{d * 100:.1f}" for d in dz]), ("lift", [str(int(v)) for v in L["lift"]])], title="target height above its start, and the label")
    save_tile(P, caption_of("probe.psi0.lift", "readout of whether the target is lifted >= 3 cm above its initial height at each knot",
                            "readout (inert), address knot x pair (slot 0), bce", "gt sim truth (target pose), training-only"),
              "computed", dict(base, tick=t, knots=knots(t), value=L["lift"].tolist(), dz=[float(x) for x in dz]), records)

    P = panel("probe.psi0.target_pos", GEOM, r"$R_{\psi_t}^{\top}(p_{\rm target}(t{+}k)-p_{\rm pelvis}(t))$", BADGE["gt"], t)
    ins = Inset(P, half=0.35)
    pts = [L["target_pos"][k_] for k_ in range(len(KS))]
    for k_, p_ in enumerate(pts):
        MF.dot(P, ins.uv(p_), GEOM, r=3.5)
    MF.line(P, [ins.uv(p_) for p_ in pts], color=GEOM, lw=1.0)
    knot_table(P, [("x", [f"{p_[0]:+.2f}" for p_ in pts]), ("y", [f"{p_[1]:+.2f}" for p_ in pts]), ("z", [f"{p_[2]:+.2f}" for p_ in pts])], y=330, title="target in the robot frame at t [m] (z not rotated)")
    save_tile(P, caption_of("probe.psi0.target_pos", "readout of the target's position at each knot in the robot (pelvis yaw) frame at t",
                            "readout (inert), address knot x pair (slot 0), gauss (3-d)", "gt sim truth (target pose), training-only"),
              "computed", dict(base, tick=t, knots=knots(t), value=L["target_pos"].tolist()), records)

    # base_disp + base_cmd (walking)
    t = t_walk
    L = E.labels(t)
    tf = frame_of(t)
    P = panel("probe.psi0.base_disp", GEOM, r"pelvis $(\Delta x,\Delta y,\Delta\psi)$ over 30 ticks", BADGE["gt"], t)
    ins = Inset(P, half=0.4)
    tr = [tf(E.pelvis[E.at(t + j), :3]) for j in range(0, 30, 3)]
    MF.line(P, [ins.uv(p_) for p_ in tr], color=GEOM, lw=1.0)
    MF.line(P, [ins.uv((0, 0)), ins.uv(L["base_disp"][:2])], color=GEOM, lw=2.0, arrow=True)
    MF.text(P, (16, 360), f"base_disp = ({L['base_disp'][0]:+.3f} m, {L['base_disp'][1]:+.3f} m, {L['base_disp'][2]:+.3f} rad)", size=0.85, ha="left")
    save_tile(P, caption_of("probe.psi0.base_disp", "readout of the pelvis displacement over the packet (t -> t+29) in the robot frame at t, plus yaw change",
                            "readout (inert), address body, gauss (3-d)", "gt sim truth (pelvis pose), training-only"),
              "computed", dict(base, tick=t, value=L["base_disp"].tolist()), records)

    ds = CachedDataset(str(PSI_RUNS / "features" / task), str(PSI_RUNS / "replay_labels" / task), episodes={ep}, load_hidden=False)
    j = next(k for k in range(len(ds)) if int(ds[k]["fr"]) == t)
    b = add_cmd_labels(collate([ds[j]]))
    bc = b["labels"]["base_cmd"][0].numpy()
    P = panel("probe.psi0.base_cmd", VIOLET, r"demonstrated (vx, turn) command at knot $k$", BADGE["gt"], t)
    knot_table(P, [("vx", [f"{v:+.2f}" for v in bc[:, 0]]), ("turn", [f"{v:+.2f}" for v in bc[:, 1]])], y=320, title="normalized action dims 32 (vx) and 34 at the knots")
    save_tile(P, caption_of("probe.psi0.base_cmd", "readout of the demonstrated base command at each knot (normalized action dims 32 = vx, 34 = turning flag per bodies.g1_simple)",
                            "readout (inert), address knot x pair (slot 0), gauss (2-d)", "label from the recorded demonstration actions, training-only"),
              "computed", dict(base, tick=t, dataset_index=j, value=bc.tolist(), note="nets.add_cmd_labels: actions[:, KNOT_STEPS][..., CMD_DIMS]; the code comment calls dim 34 the yaw-rate command, the g1_simple docstring the turning flag"), records)

    # grasp_pt / grasp_face: not computable from data on this host (no cpos__* in any recorded replay)
    assert "cpos__left" not in raw.files
    c_i = int(np.argmax(E.contact[:, 1] > 0))
    for name, formula, what in (
            ("probe.psi0.grasp_pt", r"$R(q)^{\top}(p_{\rm contact}-p_{\rm obj})$", "readout of each hand's FIRST contact point on the target in the object frame (last knot)"),
            ("probe.psi0.grasp_face", r"$2\,\arg\max|p|+[p_{\rm ax}<0]$ (6 faces)", "readout of the object face (+x,-x,+y,-y,+z,-z) of each hand's first contact")):
        P = panel(name, AMBER, formula, BADGE["illustrative"], c_i)
        MF.text(P, (300, 300), "NOT COMPUTED: the recorded replays carry no contact", size=0.85)
        MF.text(P, (300, 322), "positions (cpos__left/right); the label needs a SIMPLE", size=0.85)
        MF.text(P, (300, 344), "(Isaac Sim) re-run with LabelRecorder, not on this host", size=0.85)
        MF.text(P, (300, 372), f"frame = first right-hand contact (t={c_i}) of the same episode", size=0.75)
        save_tile(P, caption_of(name, what, "readout (inert), address knot x pair (last knot), " + ("gauss (3-d)" if name.endswith("pt") else "ce over 6"),
                                "gt sim truth, training-only (default OFF)") + " Illustrative: scene real, label value not computable here.",
                  "illustrative", dict(base, tick=c_i, reason="no cpos__left/right in any recorded replay npz (EpisodeLabels.grasp is None); computing it needs SIMPLE/Isaac Sim"), records)


# ##############################################################################################################
# build, cache, grid pages, CLI
# ##############################################################################################################
GRID_ROWS = 3                                                # max tile rows per page (family headers in between)
PROBE_RULE = ("colour = what the readout encodes: cyan geometric, amber physical interaction, violet task / procedure")
C_BUILDERS = {"arm": "build_arm", "ui": "build_ui", "pointer": "build_pointer", "legged": "build_legged", "psi0": "build_psi0"}


def implemented():
    from rrp.policies.relations import base as RB
    RB._ensure_catalog()
    return sorted(n for n, d in RB.FACTORS.items() if d.status == "implemented")


def table6_order():
    """[(group key, display name, [factor, ...])] in the row order of Table 6 (make_figures.impl_table)."""
    B, R, L, P, T = MF.load_registries()
    rows = MF.classify_rows(B, R, L, P, T, [])
    out = []
    for key, disp in MF.GROUPS:
        grp = [r for r in rows if r["group"] == key and r["status"] == "implemented"]
        fam = [m for r in MF.collapse(grp) for m in sorted(r["members"])]
        if fam:
            out.append((key, disp, fam))
    return out


def _seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def build(sel=()):
    """Build tiles (all, or groups A / B / C, C scene builders arm / ui / pointer / legged / psi0, or A factor names)
    into BUILT, then cache each (pickle) and render it standalone with a caption-fit check."""
    from rrp.harness.data import relgen as RG
    RG.load_families()
    sel = list(sel) or ["A", "B", "C"]
    failed = []
    for s_ in sel:
        if s_ == "A" or s_ in TILES:
            for n in (TILES if s_ == "A" else [s_]):
                _seed()
                try:
                    TILES[n]()
                except Exception as e:                         # one broken tile must not hide the others
                    import traceback
                    traceback.print_exc()
                    failed.append((n, repr(e)))
        elif s_ == "B":
            _seed()
            for t in build_ix_tiles(None) + build_task_tiles(None):
                emit_btile(t)
        elif s_ == "C" or s_ in C_BUILDERS:
            for k in (C_BUILDERS if s_ == "C" else [s_]):
                _seed()
                globals()[C_BUILDERS[k]]({})
        else:
            raise SystemExit(f"unknown tile selection {s_!r}")
    cache, review = WORK / "cache", WORK / "tiles"
    cache.mkdir(parents=True, exist_ok=True)
    checks = json.loads((WORK / "tile_check.json").read_text()) if (WORK / "tile_check.json").exists() else {}
    for f, rec in BUILT.items():
        (cache / f"{f}.pkl").write_bytes(pickle.dumps(rec))
        checks[f] = render_standalone(f, review)
    (WORK / "tile_check.json").write_text(json.dumps(checks, indent=1, default=str))
    log(f"[int] built {len(BUILT)} tiles; failed: {failed or 'none'}")
    return failed


def load_cache():
    recs = {}
    for p in sorted((WORK / "cache").glob("*.pkl")):
        recs[p.stem] = pickle.loads(p.read_bytes())
    return recs


def plan_pages(order, rows=GRID_ROWS, cols=GRID_COLS):
    """Pages of family blocks: a family starts a new tile row under its own header; a page holds <= `rows` tile rows;
    a family longer than the space left continues on the next page."""
    pages, cur, used = [], [], 0
    for key, disp, facs in order:
        k = 0
        while k < len(facs):
            free = rows - used
            if free <= 0:
                pages.append(cur)
                cur, used = [], 0
                continue
            need = math.ceil((len(facs) - k) / cols)
            if need > free and used > 0 and need <= rows:      # whole family fits on a fresh page: move it there
                pages.append(cur)
                cur, used = [], 0
                continue
            take = min(need, free) * cols
            cur.append((key, disp, facs[k:k + take], k > 0, len(facs)))
            used += math.ceil(len(facs[k:k + take]) / cols)
            k += take
    if cur:
        pages.append(cur)
    return pages


def _jpeg_images(pdf, qfactor=0.5):
    """Re-encode the page's raster images (rendered scenes, resampled to RASTER_DPI at print size) as JPEG; vector
    marks and text are untouched. Needs ghostscript."""
    tmp = pdf.with_suffix(".gs.pdf")
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5",
                    "-dAutoFilterColorImages=false", "-dColorImageFilter=/DCTEncode", "-dDownsampleColorImages=false",
                    f"-sOutputFile={tmp}", "-c", f"<< /ColorImageDict << /QFactor {qfactor} /Blend 1 /HSamples [1 1 1 1] "
                    "/VSamples [1 1 1 1] >> >> setdistillerparams", "-f", str(pdf)],
                   check=True)
    tmp.replace(pdf)


def tile_file(f):
    return f"figures/tiles/{f.replace('.', '_')}.pdf"


def compose_pages(out_dir=None):
    """Render the cached tiles into figures/tiles/<factor>.pdf (scene only; rasters at RASTER_DPI printed, capped at the
    native scene px, stored as JPEG), plan the grid pages (three tiles per row, family blocks in Table-6 order) and write
    fig_tiles.tex with the per-tile LaTeX captions (name, term-source glyphs, description, equation, ILLUSTRATIVE note)."""
    out_dir = Path(out_dir or FIG_DIR)
    tdir = out_dir / "tiles"
    tdir.mkdir(parents=True, exist_ok=True)
    for old in list(out_dir.glob("factor_grid_p*.pdf")) + list(tdir.glob("*.pdf")):
        old.unlink()
    recs = load_cache()
    order = table6_order()
    want = [f for _, _, fs_ in order for f in fs_]
    assert sorted(want) == implemented(), "Table-6 order does not cover the implemented registry"
    missing = [f for f in want if f not in recs]
    assert not missing, f"tiles not built: {missing}"
    caps = {f: tile_caption(recs[f]) for f in want}
    lines = tex_line_counts({f: c["desc"] for f, c in caps.items()})
    over = {f: n for f, n in lines.items() if n > DESC_MAX_LINES}
    assert not over, f"descriptions longer than {DESC_MAX_LINES} lines at {DESC_PT} pt: {over}"
    pages = plan_pages(order)
    index, stats = {}, {}
    for pi, blocks in enumerate(pages, 1):
        prow = 0
        for key, disp, facs, cont, ntot in blocks:
            for k, f in enumerate(facs):
                rr, cc = divmod(k, GRID_COLS)
                pdf = REPO / "docs" / "paper" / tile_file(f) if out_dir == FIG_DIR else tdir / Path(tile_file(f)).name
                stats[f] = st = render_tile(recs[f], pdf)
                _jpeg_images(pdf)
                index[f] = dict(page=pi, row=prow + rr, col=cc, family=key, status=recs[f]["status"],
                                file=tile_file(f), desc_lines=lines[f], colour=recs[f]["tile"].color,
                                caption=caps[f], meta=recs[f]["meta"], **st)
            prow += math.ceil(len(facs) / GRID_COLS)
        log(f"[grid3] grid page {pi}/{len(pages)} ({', '.join(b[0] for b in blocks)}; "
            f"{sum(len(b[2]) for b in blocks)} tiles, {prow} rows)")
    nl, ng = sum(s["labels"] for s in stats.values()), sum(s["labels_ge_min"] for s in stats.values())
    size = sum((tdir / Path(tile_file(f)).name).stat().st_size for f in want)
    log(f"[grid3] {len(want)} tile images, {size / 1e6:.2f} MB; in-scene labels >= {MIN_PT} pt printed: {ng}/{nl}; "
        f"description lines max {max(lines.values())}")
    (out_dir / "factor_grid.json").write_text(json.dumps(dict(
        pages=len(pages), order=want, cols=GRID_COLS, print_scale=PRINT_SCALE, tile_print_in=TILE_PRINT_IN,
        raster_dpi_printed=round(min(RASTER_DPI * PRINT_SCALE, W / TILE_IN) / PRINT_SCALE), tile_px=[W, H],
        caption=dict(name_pt=NAME_PT, desc_pt=DESC_PT, eq_pt=EQ_PT, tag_pt=TAG_PT, desc_max_lines=DESC_MAX_LINES),
        source_badges={k: v[0] for k, v in SRC_STYLE.items()}, probe_colour_rule=PROBE_RULE, tiles=index),
        indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    write_tex(pages, want, recs, caps)
    return len(pages)


TEX = MF.OUT / "fig_tiles.tex"
_TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "×": r"$\times$", "→": r"$\to$", "↔": r"$\leftrightarrow$",
            "≥": r"$\geq$", "≤": r"$\leq$", "°": r"$^\circ$", "–": "--", "—": "---", "`": "'",
            "’": "'", "−": "$-$"}


def tex_esc(s):
    out = "".join(_TEX_ESC.get(c, c) for c in s)
    bad = sorted({c for c in out if ord(c) > 126})
    assert not bad, f"unmapped characters for LaTeX: {bad} in {s[:60]!r}"
    return out


def eq_tex(s):
    """A tile formula (matplotlib mathtext, LaTeX-compatible) as LaTeX: text runs escaped, math runs kept, with the
    indicator \\mathbb{1} set as \\tileind (double-struck one)."""
    parts = s.split("$")
    assert len(parts) % 2 == 1, f"unbalanced $ in {s!r}"
    return "".join(("$" + p.replace(r"\mathbb{1}", r"\tileind") + "$") if k % 2 else tex_esc(p)
                   for k, p in enumerate(parts))


def tile_caption(rec):
    """LaTeX caption parts of a tile, generated from its record: name, term-source glyphs, description (what it encodes
    on the shown state, operator and form with the hand-set coefficients, and the source; condensed by DESC where the
    full text exceeds DESC_MAX_LINES), equation, ILLUSTRATIVE reason."""
    t = rec["tile"]
    f = t.factor
    body, src = desc_parts(rec)
    full = body + (f" Source: {src}." if src else "")
    seen = []
    for c, _ in badge_chips(rec):
        if c not in seen:
            seen.append(c)
    note = ILLUS.get(f) or (rec["meta"].get("reason") if rec["status"] == "illustrative" else "")
    assert bool(note) == (rec["status"] == "illustrative"), f"ILLUSTRATIVE note mismatch: {f}"
    return dict(name=tex_esc(f), colour=ink(t.color).lstrip("#").upper(), glyphs=r"\,".join(SRC_TEX[c] for c in seen),
                desc=tex_esc(DESC.get(f, full)), condensed=f in DESC, full=full, eq=eq_tex(formula_of(rec)),
                illus=tex_esc(note or ""))


TECTONIC = os.environ.get("RRP_TECTONIC", str(Path.home() / ".local" / "bin" / "tectonic"))
TILE_MACROS = r"""\providecommand{\tileind}{\mathds{1}}
\newlength{\tilew}\newlength{\tilegap}\setlength{\tilegap}{%(gap).2fin}
\newcommand{\tilesetw}{\setlength{\tilew}{\dimexpr(\textwidth-%(ngap)d\tilegap)/%(cols)d-0.5pt\relax}}%% -0.5pt: sp rounding must not wrap a row
\newcommand{\tilenamefont}{\fontsize{%(name)s}{%(namel)s}\selectfont\ttfamily}
\newcommand{\tiledescfont}{\fontsize{%(desc)s}{%(descl)s}\selectfont\rmfamily\raggedright}
\newcommand{\tileeqfont}{\fontsize{%(eq)s}{%(eql)s}\selectfont\rmfamily\raggedright}
\newcommand{\tilesrcpub}{\textcolor[HTML]{2B2F36}{\ding{108}}}
\newcommand{\tilesrcgt}{\textcolor[HTML]{B3261E}{$\blacktriangle$}}
\newcommand{\tilesrcprobe}{\textcolor[HTML]{1F5FAD}{$\blacklozenge$}}
\newcommand{\tilecell}[7]{%% #1 image, #2 family colour, #3 name, #4 source glyphs, #5 description, #6 equation, #7 illustrative
\begin{minipage}[t]{\tilew}\vspace{0pt}\includegraphics[width=\tilew]{#1}\par\vspace{1pt}%%
{\tilenamefont\textcolor[HTML]{#2}{#3}\hfill#4\par}\vspace{0.5pt}%%
{\tiledescfont #5\par}\vspace{1pt}%%
{\tileeqfont #6\par}%%
\if\relax\detokenize{#7}\relax\else\vspace{1pt}{\tiledescfont\textcolor[HTML]{B3261E}{\textsc{illustrative}:} #7\par}\fi
\end{minipage}}
\newcommand{\tilefamily}[4]{%% #1 colour, #2 family, #3 count note, #4 right-hand note
\par\noindent{\fontsize{8.5}{10}\selectfont\ttfamily\bfseries\textcolor[HTML]{#1}{#2}}{\fontsize{7.5}{9}\selectfont\ (#3)}\hfill{\fontsize{7}{8.4}\selectfont #4}\par
\nointerlineskip\vspace{1.5pt}\noindent\textcolor[HTML]{#1}{\rule{\textwidth}{0.6pt}}\par\nointerlineskip\vspace{4pt}}
""" % dict(gap=GRID_GAP_IN, ngap=GRID_COLS - 1, cols=GRID_COLS, name=NAME_PT, namel=round(NAME_PT * 1.2, 2), desc=DESC_PT,
           descl=round(DESC_PT * 1.2, 2), eq=EQ_PT, eql=round(EQ_PT * 1.2, 2))


def tex_line_counts(texts):
    """{key: LaTeX text} -> {key: number of lines at DESC_PT in a \\tilew-wide ragged-right box}, typeset by tectonic
    with the paper's page geometry and fonts (the exact line breaks of the report)."""
    keys = list(texts)
    doc = [r"\documentclass[10pt,twocolumn]{article}", r"\usepackage[margin=0.57in]{geometry}",
           r"\usepackage{array,amsmath,amssymb,graphicx,xcolor,microtype,pifont,dsfont}", TILE_MACROS,
           r"\begin{document}\tilesetw"]
    for i, k in enumerate(keys):
        doc.append(r"\setbox0\vbox{\tiledescfont " + texts[k] + r"\par\xdef\nl{\the\prevgraf}}\typeout{NLINES " + str(i)
                   + r" \nl}")
    doc += [r"x\end{document}"]
    d = WORK / "linecount"
    d.mkdir(parents=True, exist_ok=True)
    (d / "lc.tex").write_text("\n".join(doc) + "\n")
    subprocess.run([TECTONIC, "--keep-logs", "-c", "minimal", "lc.tex"], cwd=d, check=True, capture_output=True)
    out = {}
    for ln in (d / "lc.log").read_text().splitlines():
        if ln.startswith("NLINES "):
            _, i, n = ln.split()
            out[keys[int(i)]] = int(n)
    assert len(out) == len(keys), "line counts missing from the tectonic log"
    return out


def write_tex(pages, want, recs, caps):
    """fig_tiles.tex: the tile macros and \\tilegrid = one float page per grid page (pages 2.. continue Fig. tiles);
    each tile = image + its LaTeX caption (tile_caption)."""
    npages = len(pages)
    first = (r"\captionsetup{font=footnotesize}\caption{\textbf{Every implemented relation factor, one tile each} ("
             + f"{len(want)} tiles on {npages} pages" + r", three per row, families in the row order of "
             r"Table~\ref{tab:relations}). Each tile shows one factor's term for one query token (ring or outlined chip) "
             r"on a real simulator or environment state, computed by the repository's factor operators or label code; "
             r"\emph{nothing is trained attention or a trained probe output}. Frame colour = family (Fig.~\ref{fig:fig2}); "
             r"probe readouts are coloured by what they encode. Top-left chip: scene source (\textsc{scripted teacher} "
             r"states are privileged). Under each tile: the factor and its term source, \tilesrcpub~\emph{public}: "
             r"computed from public inputs, deployable as shown; \tilesrcgt~\emph{gt train-only}: the colour is the "
             r"simulator-truth label that trains a probe or readout, never a deployed input; \tilesrcprobe~\emph{est-probe}: "
             r"the deployed term is the probe's estimate (not shown); then what the term encodes on the shown state, its "
             r"operator and form (learned coefficients are \emph{set by hand}, values stated) and source, and its equation. "
             r"\textsc{illustrative}: not fully computable from shipped code or data (reason under the tile). "
             r"\texttt{make\_figures.py tiles}.}\label{fig:tiles}")
    out = ["% generated by make_figures.py tiles (factor_tiles.write_tex); do not edit", TILE_MACROS.rstrip("\n"),
           r"\newcommand{\tilegrid}{%"]
    for k, blocks in enumerate(pages, 1):
        cap = first if k == 1 else (r"\ContinuedFloat\captionsetup{font=footnotesize}\caption[]{\textbf{Every implemented "
                                     r"relation factor} (continued, page " + f"{k}/{npages}" + r"). Chips, glyphs and "
                                     r"colours as on the first page.}")
        # a table* float typed as a figure: double-column floats keep their order only within one type, so this keeps
        # the grid pages after Table 6 (and Table 7 after them) instead of overtaking the body's pending tables
        out += [r"\begin{table*}[p]", r"\captionsetup{type=figure}", r"\tilesetw"]
        if k == 1:
            out += [cap]
        for bi, (key, disp, facs, cont, ntot) in enumerate(blocks):
            col = ink(FAMILY.get(key, INK)).lstrip("#").upper()
            note = f"{ntot} implemented" + (", continued" if cont else "")
            right = tex_esc(PROBE_RULE) if key == "probe" and not cont else ""
            if bi:
                out.append(r"\vspace{6pt}")
            out.append(rf"\tilefamily{{{col}}}{{{tex_esc(disp.upper())}}}{{{note}}}{{{right}}}")
            for r0 in range(0, len(facs), GRID_COLS):
                cells = []
                for f in facs[r0:r0 + GRID_COLS]:
                    c = caps[f]
                    cells.append(rf"\tilecell{{{tile_file(f)}}}{{{c['colour']}}}{{{c['name']}}}{{{c['glyphs']}}}%"
                                 + "\n" + rf"  {{{c['desc']}}}%" + "\n" + rf"  {{{c['eq']}}}{{{c['illus']}}}")
                out.append(r"\noindent" + "\\hspace{\\tilegap}%\n".join(c + "" for c in cells) + r"\par")
                if r0 + GRID_COLS < len(facs):
                    out.append(r"\vspace{3.5pt}")
        if k > 1:
            out += [r"\vspace{2pt}", cap]
        out.append(r"\end{table*}")
    out.append("}")
    TEX.write_text("\n".join(out) + "\n")


def main(argv):
    cmd, rest = (argv[0], argv[1:]) if argv else ("all", [])
    if cmd in ("build", "all"):
        build(rest if cmd == "build" else ())
    if cmd in ("grid", "all"):
        print("factor grid pages:", compose_pages())
    if cmd not in ("build", "grid", "all"):
        raise SystemExit("usage: make_figures.py tiles [build [A|B|C|<C scene>|<A factor> ...] | grid | all]")


if __name__ == "__main__":
    main(sys.argv[1:])
