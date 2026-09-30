"""Out-of-distribution packet detector for system 0 (D-126 roadmap #29). numpy only.

A per-bundle density model of the packet z, fitted on TRAINING packets (the frozen encoder's posterior means on the
training split, i.e. exactly what system 0 was trained to realize), stored next to the bundle and fingerprinted:
  - features: `packet` = the packet's valid (knot, assembly) entries flattened (one model per body / assembly count),
    or `token` = each (knot, assembly) row scored separately (body-agnostic; the packet score is the max over tokens);
  - model: standardize, PCA to r components (r <= n-1), squared Mahalanobis distance in the PC space plus the
    residual (out-of-subspace) energy divided by its mean training value; with a diagonal-shrinkage floor on the
    eigenvalues. This is the Gaussian log-density up to constants (a PPCA-style score);
  - thresholds: quantiles of the scores of a held-out calibration split (default: monitor at q0.99, enforce at
    q0.999), so the nominal false-alarm rate is stated, not guessed.
The model file records the bundle's latent_space_version (compatibility: a detector fitted for another encoder
is refused), the body spec hash / assembly count it applies to, the data it was fitted on, and a sha256 over its
arrays (`fingerprint`).

Runtime (`PacketOODMonitor`), option `packet_ood`: off (default; nothing runs) | monitor (score and log every
packet, behaviour unchanged) | enforce (a packet scoring above the enforce threshold is rejected with code
`packet_ood`; system 0 then applies the declared fallback: `hold_default` (the body's default stance, the existing
legged fallback) | `hold_measured` (hold the measured joint positions) | `safe_stop` (ramp to the hold posture,
rrp.policies.safety)).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

OOD_VERSION = "pood-1"
OOD_MODES = ("off", "monitor", "enforce")
OOD_FALLBACKS = ("hold_default", "hold_measured", "safe_stop")
FEATURES = ("packet", "token")


def packet_features(z, asm_mask=None) -> np.ndarray:
    """z [K, M, d] (or [B, K, M, d]) -> [B, K*M_valid*d] over valid assemblies."""
    z = np.asarray(z, np.float64)
    if z.ndim == 3:
        z = z[None]
    if asm_mask is not None:
        z = z[:, :, np.asarray(asm_mask, bool)]
    return z.reshape(z.shape[0], -1)


def token_features(z, asm_mask=None) -> np.ndarray:
    """z [K, M, d] -> [K*M_valid, d] (one row per knot x valid assembly)."""
    z = np.asarray(z, np.float64)
    if asm_mask is not None:
        z = z[:, np.asarray(asm_mask, bool)]
    return z.reshape(-1, z.shape[-1])


@dataclass
class GaussianSubspaceModel:
    mean: np.ndarray
    scale: np.ndarray                  # per-feature std used to standardize
    components: np.ndarray             # [r, p] orthonormal rows
    eigvals: np.ndarray                # [r]
    resid_mean: float                  # mean residual energy on the fit split (normalizer)

    @classmethod
    def fit(cls, X: np.ndarray, r: int = 32, shrink: float = 1e-3) -> "GaussianSubspaceModel":
        X = np.asarray(X, np.float64)
        n, p = X.shape
        if n < 3:
            raise ValueError(f"need >= 3 samples to fit, got {n}")
        mean = X.mean(0)
        scale = X.std(0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        Y = (X - mean) / scale
        r = int(min(r, n - 1, p))
        _, s, vt = np.linalg.svd(Y, full_matrices=False)
        ev = s[:r] ** 2 / (n - 1)
        ev = np.maximum(ev, shrink * max(float(ev.mean()) if len(ev) else 1.0, 1e-12))
        comps = vt[:r]
        res = Y - (Y @ comps.T) @ comps
        rm = float((res ** 2).sum(1).mean())
        return cls(mean, scale, comps, ev, max(rm, 1e-12))

    def score(self, X: np.ndarray) -> np.ndarray:
        """PPCA-style score per row: squared Mahalanobis distance in the PC subspace + residual energy over the
        isotropic residual variance (resid_mean / n_res). In distribution its mean is ~ p (the feature dim)."""
        Y = (np.atleast_2d(np.asarray(X, np.float64)) - self.mean) / self.scale
        c = Y @ self.components.T
        res = Y - c @ self.components
        n_res = self.mean.shape[0] - self.components.shape[0]          # isotropic residual dims (PPCA)
        res_term = (res ** 2).sum(1) * n_res / self.resid_mean if n_res > 0 else 0.0
        return (c ** 2 / self.eigvals).sum(1) + res_term

    def arrays(self) -> dict:
        return dict(mean=self.mean, scale=self.scale, components=self.components, eigvals=self.eigvals,
                    resid_mean=np.array([self.resid_mean]))


@dataclass
class PacketOODModel:
    """A fitted detector for one bundle (latent_space_version) and one body (spec hash / assembly count)."""
    model: GaussianSubspaceModel
    feature: str
    thresholds: dict                   # {"monitor": float, "enforce": float}
    latent_space_version: str
    n_assemblies: int | None = None
    body: str | None = None
    spec_hash: str | None = None
    fit_info: dict = field(default_factory=dict)
    version: str = OOD_VERSION

    # ------------------------------------------------------------------ scoring
    def score(self, z, asm_mask=None) -> float:
        if self.feature == "packet":
            return float(self.model.score(packet_features(z, asm_mask))[0])
        return float(self.model.score(token_features(z, asm_mask)).max())

    def scores(self, zs, asm_mask=None) -> np.ndarray:
        return np.array([self.score(z, asm_mask) for z in zs])

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, z_fit, z_cal, *, latent_space_version: str, feature: str = "packet", asm_mask=None, r: int = 32,
            q_monitor: float = 0.99, q_enforce: float = 0.999, body=None, spec_hash=None,
            fit_info: dict | None = None) -> "PacketOODModel":
        """z_fit / z_cal: [N, K, M, d] training packets (fit split / held-out calibration split)."""
        if feature not in FEATURES:
            raise ValueError(f"feature {feature!r} not in {FEATURES}")
        z_fit, z_cal = np.asarray(z_fit), np.asarray(z_cal)
        X = (packet_features(z_fit, asm_mask) if feature == "packet"
             else np.concatenate([token_features(z, asm_mask) for z in z_fit]))
        m = GaussianSubspaceModel.fit(X, r=r)
        M = int(np.asarray(asm_mask, bool).sum()) if asm_mask is not None else int(z_fit.shape[2])
        tmp = cls(m, feature, {}, latent_space_version, M, body, spec_hash)
        cal = tmp.scores(z_cal, asm_mask)
        tmp.thresholds = dict(monitor=float(np.quantile(cal, q_monitor)), enforce=float(np.quantile(cal, q_enforce)),
                              q_monitor=q_monitor, q_enforce=q_enforce)
        tmp.fit_info = dict(fit_info or {}, n_fit=int(len(z_fit)), n_cal=int(len(z_cal)), r=int(m.components.shape[0]),
                            cal_score_median=float(np.median(cal)), cal_score_max=float(cal.max()))
        return tmp

    # ------------------------------------------------------------------ io
    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for k, v in sorted(self.model.arrays().items()):
            h.update(k.encode())
            h.update(np.ascontiguousarray(v, np.float64).tobytes())
        h.update(json.dumps(self._meta(with_fp=False), sort_keys=True).encode())
        return h.hexdigest()[:16]

    def _meta(self, with_fp=True) -> dict:
        d = dict(version=self.version, feature=self.feature, thresholds=self.thresholds,
                 latent_space_version=self.latent_space_version, n_assemblies=self.n_assemblies, body=self.body,
                 spec_hash=self.spec_hash, fit_info=self.fit_info)
        if with_fp:
            d["fingerprint"] = self.fingerprint()
        return d

    def save(self, path: Path) -> Path:
        """<path>.npz (arrays) + <path>.json (metadata incl. fingerprint). Not committed (weights-like data)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path.with_suffix(".npz"), **self.model.arrays())
        path.with_suffix(".json").write_text(json.dumps(self._meta(), indent=1))
        return path.with_suffix(".json")

    @classmethod
    def load(cls, path: Path, *, expect_lsv: str | None = None) -> "PacketOODModel":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        a = np.load(path.with_suffix(".npz"))
        m = GaussianSubspaceModel(a["mean"], a["scale"], a["components"], a["eigvals"], float(a["resid_mean"][0]))
        obj = cls(m, meta["feature"], meta["thresholds"], meta["latent_space_version"], meta.get("n_assemblies"),
                  meta.get("body"), meta.get("spec_hash"), meta.get("fit_info", {}), meta.get("version", OOD_VERSION))
        if obj.fingerprint() != meta.get("fingerprint"):
            raise ValueError(f"{path}: OOD model fingerprint mismatch (file changed after fitting)")
        if expect_lsv is not None and expect_lsv != obj.latent_space_version:
            raise ValueError(f"{path}: OOD model fitted for latent space {obj.latent_space_version}, "
                             f"bundle is {expect_lsv}")
        return obj


class PacketOODMonitor:
    """Runtime wrapper: mode off | monitor | enforce. `check` returns a record; `reject` says whether to drop."""

    def __init__(self, model: PacketOODModel | None, mode: str = "off", fallback: str = "hold_default"):
        if mode not in OOD_MODES:
            raise ValueError(f"packet_ood mode {mode!r} not in {OOD_MODES}")
        if fallback not in OOD_FALLBACKS:
            raise ValueError(f"packet_ood fallback {fallback!r} not in {OOD_FALLBACKS}")
        if mode != "off" and model is None:
            raise ValueError("packet_ood monitor/enforce needs a fitted model (--ood-model)")
        self.model, self.mode, self.fallback = model, mode, fallback
        self.n = self.n_monitor = self.n_enforce = 0
        self.log: list[dict] = []

    def check(self, z, asm_mask=None, t: float | None = None) -> dict | None:
        if self.mode == "off":
            return None
        s = self.model.score(z, asm_mask)
        th = self.model.thresholds
        rec = dict(t=t, score=s, over_monitor=bool(s > th["monitor"]), over_enforce=bool(s > th["enforce"]))
        rec["rejected"] = bool(self.mode == "enforce" and rec["over_enforce"])
        self.n += 1
        self.n_monitor += rec["over_monitor"]
        self.n_enforce += rec["over_enforce"]
        self.log.append(rec)
        return rec

    def summary(self) -> dict:
        sc = [r["score"] for r in self.log]
        return dict(version=OOD_VERSION, mode=self.mode, fallback=self.fallback,
                    model_fingerprint=self.model.fingerprint() if self.model else None,
                    thresholds=self.model.thresholds if self.model else None, n=self.n,
                    over_monitor=self.n_monitor, over_enforce=self.n_enforce,
                    rejected=sum(r["rejected"] for r in self.log),
                    score_max=max(sc) if sc else None, score_median=float(np.median(sc)) if sc else None,
                    first_over_enforce_t=next((r["t"] for r in self.log if r["over_enforce"]), None))
