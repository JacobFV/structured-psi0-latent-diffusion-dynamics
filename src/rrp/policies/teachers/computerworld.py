"""Scripted teachers for the ComputerWorld cw/* tasks (source `scripted_teacher`, privileged: they read the full CW
scene through the env, including occluded widgets). One plan per task; the pointer moves at MAX_STEP_PX per tick toward
the target widget's centre, clicks are press/release on consecutive ticks, text is typed one symbol per tick.
`CWTeacher.target` / `.target_px` are the widget slot key and pointer destination of the current plan step (labels for
the pointer probes, rrp.policies.pointer; never policy inputs)."""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

from rrp.core.action import NativeCommand
from rrp.envs.computerworld import CONTROLLER_VERSION, CW_TASKS, ComputerWorldEnv, find_widget, key_index

MAX_STEP_PX = 60          # 60 px/tick = 0.6 m/s at 1 mm/px and 10 Hz
TEACHER_VERSION = "cw_teacher.v1"


def command(env: ComputerWorldEnv, pointer=None, button=None, key=None) -> dict[str, list[float]]:
    """Groups for one tick; pointer/button default to holding the current state."""
    g = {"pointer": list(pointer if pointer is not None else env.frame.px_to_m(env.pointer.u, env.pointer.v)),
         "button": [float(env.pointer.button if button is None else button)]}
    if key is not None:
        g["key"] = [float(key)]
    return g


def goto_px(env, u, v, button=None, tt=None) -> Iterator[dict]:
    if tt is not None:
        tt.target_px = (u, v)
    while (env.pointer.u, env.pointer.v) != (u, v):
        du, dv = u - env.pointer.u, v - env.pointer.v
        n = max(1.0, np.hypot(du, dv) / MAX_STEP_PX)
        yield command(env, env.frame.px_to_m(round(env.pointer.u + du / n), round(env.pointer.v + dv / n)), button)


def goto(env, pred: Callable[[dict], bool], tt=None) -> Iterator[dict]:
    w = find_widget(env, pred)
    if w is None or w["box"] is None:
        raise RuntimeError("teacher target not found")
    if tt is not None:
        tt.target = w["key"]
    x0, y0, x1, y1 = w["box"]
    yield from goto_px(env, (x0 + x1) // 2, (y0 + y1) // 2, tt=tt)


def click(env, pred, tt=None) -> Iterator[dict]:
    yield from goto(env, pred, tt)
    yield command(env, button=1)
    yield command(env, button=0)


def type_text(env, text: str) -> Iterator[dict]:
    for ch in text:
        yield command(env, key=key_index(ch))


def _calc(env, tt=None):
    g = env.goal
    for label in (str(g["a"]), "+", str(g["b"]), "="):
        yield from click(env, lambda w, l=label: w["interaction"].endswith(f":calc:{l}"), tt)


def _open_type(env, tt=None):
    yield from click(env, lambda w: w["interaction"] == "shell:launch:editor", tt)
    if tt is not None:
        tt.target = None                     # typing goes to the editor's text area (no scene widget)
    yield command(env)
    yield from type_text(env, env.goal["text"])


def _drag(env, tt=None):
    g = env.goal
    yield from goto(env, lambda w: w["interaction"] == f"window:{g['window']}:drag", tt)
    yield command(env, button=1)
    yield from goto_px(env, env.pointer.u + g["dx"], env.pointer.v + g["dy"], button=1, tt=tt)
    yield command(env, button=0)


def _form(env, tt=None):
    for fld in ("name", "email"):
        yield from click(env, lambda w, f=fld: w["role"] == "textbox" and w["interaction"].endswith(f":content:{f}"), tt)
        yield from type_text(env, env.goal[fld])
    yield from click(env, lambda w: w["role"] == "button" and w["label"] == "Submit", tt)


PLANS: dict[str, Callable[[ComputerWorldEnv], Iterator[dict]]] = {
    "cw/calc_sum": _calc, "cw/open_type": _open_type, "cw/drag_window": _drag, "cw/fill_form": _form}
assert set(PLANS) == set(CW_TASKS)


class CWTeacher:
    """One episode's plan; act() -> NativeCommand, or None (hold) once the plan has finished."""

    def __init__(self, env: ComputerWorldEnv, task: str):
        self.target: str | None = None                 # slot key of the current step's widget
        self.target_px: tuple[int, int] | None = None  # pointer destination of the current step (px)
        self._plan = PLANS[task](env, self)

    def act(self) -> NativeCommand | None:
        g = next(self._plan, None)
        return None if g is None else NativeCommand(controller_version=CONTROLLER_VERSION, groups=g,
                                                    source="scripted_teacher")


def teacher_factory(task: str, version: str | None = None, options: dict | None = None):
    """For rrp.policies.teachers (teacher:cw/*): (make(env) -> teacher, version string, action kinds it emits)."""
    if task not in PLANS:
        raise KeyError(f"no ComputerWorld teacher for {task!r}; known: {sorted(PLANS)}")
    return (lambda env: CWTeacher(env, task)), version or TEACHER_VERSION, ("cartesian_position", "button", "discrete")
