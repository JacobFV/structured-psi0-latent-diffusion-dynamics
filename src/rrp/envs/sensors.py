"""Declared public sensors: proprioception, touch, a simulated object detector + tracker.

The detector is a SENSOR MODEL: it uses simulator state to synthesize measurements
(visibility by camera frustum + ray occlusion, Gaussian position noise, dropout). Its
outputs are public; the underlying truth never leaves the privileged bus. The tracker
keeps per-slot beliefs whose covariance grows while unobserved (unknown is not false).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from rrp.contracts.observation import ObjectDescriptor


@dataclass
class DetectorConfig:
    camera: str = "front"
    pos_sigma: float = 0.004          # m, per axis, when visible
    dropout: float = 0.02
    growth_sigma_per_s: float = 0.02  # belief std growth while unobserved
    max_range: float = 3.0
    process_sigma: float = 0.03       # per-update motion allowance (objects can be carried)


@dataclass
class TrackState:
    slot: int
    descriptor: str
    mean: list | None = None
    var: list | None = None
    last_seen: float | None = None
    visible: bool = False


class ObjectTracker:
    """Public belief over canonical object slots (slot order fixed at reset)."""

    def __init__(self, descriptors: list[str], cfg: DetectorConfig):
        self.cfg = cfg
        self.tracks = [TrackState(i, d) for i, d in enumerate(descriptors)]

    def update(self, t: float, measurements: dict[int, np.ndarray | None]):
        for tr in self.tracks:
            z = measurements.get(tr.slot)
            if z is not None:
                r = self.cfg.pos_sigma ** 2
                if tr.mean is None:
                    tr.mean, tr.var = list(map(float, z)), [r] * 3
                else:
                    m, v = np.array(tr.mean), np.array(tr.var) + self.cfg.process_sigma ** 2  # predict
                    k = v / (v + r)
                    m = m + k * (z - m)
                    v = (1 - k) * v
                    tr.mean, tr.var = m.tolist(), v.tolist()
                tr.last_seen, tr.visible = t, True
            else:
                tr.visible = False
                if tr.var is not None and tr.last_seen is not None:
                    g = self.cfg.growth_sigma_per_s
                    tr.var = [v + (g ** 2) * 0.05 for v in tr.var]

    def descriptors(self, t: float) -> list[ObjectDescriptor]:
        out = []
        for tr in self.tracks:
            out.append(ObjectDescriptor(slot=tr.slot, descriptor=tr.descriptor, position_estimate=tr.mean,
                                        position_cov_diag=tr.var, visible=tr.visible, timestamp=t))
        return out

    def state(self) -> list[dict]:
        return [dict(tr.__dict__) for tr in self.tracks]

    def load(self, st: list[dict]):
        self.tracks = [TrackState(**d) for d in st]


def camera_visibility(model: mujoco.MjModel, data: mujoco.MjData, cam: str, body: str, point: np.ndarray,
                      max_range: float = 3.0) -> bool:
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    if cid < 0:
        return False
    cpos = data.cam_xpos[cid].copy()
    cmat = data.cam_xmat[cid].reshape(3, 3)
    v = point - cpos
    dist = float(np.linalg.norm(v))
    if dist > max_range or dist < 1e-6:
        return False
    # camera looks along -z of its frame
    cam_dir = -cmat[:, 2]
    fovy = math.radians(model.cam_fovy[cid])
    cosang = float(v @ cam_dir) / dist
    if cosang < math.cos(fovy * 0.5 * 1.3):
        return False
    geomid = np.array([-1], dtype=np.int32)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    frac = mujoco.mj_ray(model, data, cpos, v / dist, None, 1, -1, geomid)
    if geomid[0] < 0 or frac < 0:
        return True   # nothing hit (e.g. non-colliding visual marker) -> line of sight clear
    hit_body = model.geom_bodyid[geomid[0]]
    if hit_body == bid:
        return True
    return frac >= dist - 0.03


def read_sensor(model, data, name: str) -> np.ndarray | None:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        return None
    adr, dim = model.sensor_adr[sid], model.sensor_dim[sid]
    return data.sensordata[adr:adr + dim].copy()
