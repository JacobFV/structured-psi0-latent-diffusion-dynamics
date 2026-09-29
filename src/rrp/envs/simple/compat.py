"""Run upstream SIMPLE (written for Isaac Sim 4.5, py3.10, x86_64) on Isaac Sim 5.1 / py3.11 / aarch64 (GB10) WITHOUT
editing the upstream clone (moved from psi1z `simple_compat` + `_simple_autocompat` + `simple_run`, D-140; P-002).

Stdlib-only at import: this module is imported inside the SIMPLE venv (the SimpleEnv worker, and every interpreter via
the `.pth` line that `scripts/psi0_ext.sh simple-env` writes):

    import os; os.environ.get("RRP_SIMPLE_COMPAT") == "1" and __import__("rrp.envs.simple.compat", fromlist=["x"]).install()

Isaac Sim 4.5 has no aarch64 wheel; 5.1 is the first. 5.1 still ships the deprecated `omni.isaac.*` extensions, so the
alias hook below is opt-in (`RRP_SIMPLE_ISAAC_ALIASES=1`, for builds without the shims). cuRobo (motion planning / IK
only) and envlogger (recording only) are stubbed when absent: any use raises, so a silent wrong result is impossible.
Also: AMO weights path + TorchScript GPU fusers off, Isaac 5.1 add_reference fallback, render profiles
(`RRP_SIMPLE_RENDER`, default pt4_iso55, P-007), one render per control step (`RRP_SIMPLE_RENDER_EVERY`), task-uid fix.

Run a SIMPLE entry point under the layer:  python -m rrp.envs.simple.compat <module> <args...>
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import types
from pathlib import Path

EXT = Path(os.environ.get("RRP_PSI0_EXT", Path.home() / "work/ext"))
SIMPLE_ROOT = EXT / "psi0/third_party/SIMPLE"

# old module -> (new module, {old attr: new attr})
ALIASES: dict[str, tuple[str, dict[str, str]]] = {
    "omni.isaac.kit": ("isaacsim.simulation_app", {}),
    "omni.isaac.core.utils.prims": ("isaacsim.core.utils.prims", {}),
    "omni.isaac.core.utils.stage": ("isaacsim.core.utils.stage", {}),
    "omni.isaac.core.utils.bounds": ("isaacsim.core.utils.bounds", {}),
    "omni.isaac.core.utils.extensions": ("isaacsim.core.utils.extensions", {}),
    "omni.isaac.core.objects": ("isaacsim.core.api.objects", {}),
    "omni.isaac.core.robots.robot": ("isaacsim.core.api.robots.robot", {}),
    "omni.isaac.core.world.world": ("isaacsim.core.api.world.world", {}),
    "omni.isaac.core.prims": ("isaacsim.core.prims", {"GeometryPrim": "SingleGeometryPrim",
                                                      "RigidPrim": "SingleRigidPrim",
                                                      "XFormPrim": "SingleXFormPrim"}),
    "omni.isaac.sensor": ("isaacsim.sensors.camera", {}),
}
PARENTS = ("omni.isaac", "omni.isaac.core", "omni.isaac.core.utils", "omni.isaac.core.robots", "omni.isaac.core.world")


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, name):
        self.name = name

    def create_module(self, spec):
        if self.name in PARENTS:
            m = types.ModuleType(self.name)
            m.__path__ = []
            return m
        target, renames = ALIASES[self.name]
        tgt = importlib.import_module(target)
        m = types.ModuleType(self.name)
        m.__dict__.update({k: v for k, v in tgt.__dict__.items() if not k.startswith("__")})
        for old, new in renames.items():
            m.__dict__[old] = getattr(tgt, new)
        m.__rrp_alias_of__ = target
        return m

    def exec_module(self, module):
        return None


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ALIASES or fullname in PARENTS:
            try:   # a real module wins (e.g. an Isaac build that still ships the shim)
                for f in sys.meta_path:
                    if f is self or not hasattr(f, "find_spec"):
                        continue
                    s = f.find_spec(fullname, path, target)
                    if s is not None:
                        return s
            except Exception:
                pass
            return importlib.machinery.ModuleSpec(fullname, _AliasLoader(fullname),
                                                  is_package=fullname in PARENTS)
        return None


def _stub_curobo():
    try:   # SIMPLE's empty third_party/curobo submodule dir is a namespace package: test a real submodule
        importlib.import_module("curobo.types.base")
        return False
    except Exception:
        for k in [k for k in sys.modules if k == "curobo" or k.startswith("curobo.")]:
            del sys.modules[k]

    class _Stub:
        def __init__(self, *a, **k):
            raise RuntimeError("curobo stub: cuRobo is not installed in this env (not needed for policy eval)")

    class _StubMod(types.ModuleType):
        def __getattr__(self, item):
            if item.startswith("__"):
                raise AttributeError(item)
            return _Stub

    names = ["curobo", "curobo.types", "curobo.types.base", "curobo.types.math", "curobo.types.state",
             "curobo.types.robot", "curobo.wrap", "curobo.wrap.reacher", "curobo.wrap.reacher.ik_solver",
             "curobo.wrap.reacher.motion_gen", "curobo.util_file", "curobo.util", "curobo.util.logger",
             "curobo.util.usd_helper", "curobo.cuda_robot_model", "curobo.cuda_robot_model.cuda_robot_model",
             "curobo.geom", "curobo.geom.types", "curobo.geom.sdf", "curobo.geom.sdf.world",
             "curobo.rollout", "curobo.rollout.cost", "curobo.rollout.cost.pose_cost"]
    for n in names:
        m = _StubMod(n)
        m.__path__ = []
        m.__rrp_stub__ = True
        sys.modules[n] = m
    return True


def _stub_optional(names=("envlogger",)):
    """Stub optional recorder deps (envlogger/RLDS: data recording only, no aarch64 wheels); any use raises."""
    done = []
    for n in names:
        try:
            importlib.import_module(n)
            continue
        except Exception:
            pass

        class _Use:
            def __init__(self, *a, **k):
                raise RuntimeError("stub: optional recorder dependency not installed (data recording only)")

        class _S(types.ModuleType):
            def __getattr__(self, item):
                if item.startswith("__"):
                    raise AttributeError(item)
                return _Use
        m = _S(n); m.__path__ = []; m.__rrp_stub__ = True
        sys.modules[n] = m
        done.append(n)
    return done


def simple_sys_paths() -> list[str]:
    return [str(SIMPLE_ROOT / "src"), str(SIMPLE_ROOT / "third_party"),
            str(SIMPLE_ROOT / "third_party/openpi-client/src"),
            str(SIMPLE_ROOT / "third_party/unitree_sdk2_python")]


def _patch_add_reference(mod):
    """Isaac Sim 5.1's add_reference_to_stage routes stages with divergent metrics (SIMPLE's scene vs the graspnet
    USDs) through the kit command `AddReference` (omni.metrics.assembler.ui), which on this build returns without
    authoring the reference (verified: prim valid, no children, no references; 4.5 authored it directly). We keep
    the upstream call and, only if no reference got authored, author it directly (the 4.5 behaviour)."""
    orig = mod.add_reference_to_stage
    if getattr(orig, "__rrp__", False):
        return

    def add_reference_to_stage(usd_path, prim_path, prim_type="Xform"):
        prim = orig(usd_path=usd_path, prim_path=prim_path, prim_type=prim_type)
        if prim is not None and prim.IsValid() and not prim.HasAuthoredReferences():
            if not prim.GetReferences().AddReference(usd_path):
                raise FileNotFoundError(usd_path)
            _PATCH_LOG.append(prim_path)
        return prim
    add_reference_to_stage.__rrp__ = True
    mod.add_reference_to_stage = add_reference_to_stage


_PATCH_LOG: list = []

# Rendering profiles (research/tracks/psi0.md, P-002/P-007): the RTX real-time path on Isaac 5.1/aarch64 renders the
# SIMPLE HSSD scenes ~8x darker than the released data (NRD denoiser shaders fail to compile); path tracing with the
# OptiX denoiser and film ISO 65 matches the released eval frames (MSE 100-200 vs 4100 on two XMovePick L0 frames).
RENDER_PROFILES = {
    "rt_default": {},
    "pt4_iso55": {"/rtx/rendermode": "PathTracing", "/rtx/pathtracing/spp": 4, "/rtx/pathtracing/totalSpp": 4,
                  "/rtx/pathtracing/optixDenoiser/enabled": True, "/rtx/post/tonemap/filmIso": 55.0},
    "pt4_iso65": {"/rtx/rendermode": "PathTracing", "/rtx/pathtracing/spp": 4, "/rtx/pathtracing/totalSpp": 4,
                  "/rtx/pathtracing/optixDenoiser/enabled": True, "/rtx/post/tonemap/filmIso": 65.0},
}


def _apply_render_profile():
    name = os.environ.get("RRP_SIMPLE_RENDER", "pt4_iso55")      # P-007 (was pt4_iso65, calibrated on eval frames)
    import carb
    st = carb.settings.get_settings()
    for k, v in RENDER_PROFILES[name].items():
        st.set(k, v)
    _PATCH_LOG.append(f"render:{name}")


def _patch_simulator(module):
    cls = getattr(module, "IsaacSimSimulator", None)
    if cls is None or getattr(cls, "__rrp__", False):
        return
    every = int(os.environ.get("RRP_SIMPLE_RENDER_EVERY", "4"))
    orig_step = cls.step

    def step(self, *args, **kw):
        # SIMPLE mirrors MuJoCo into Isaac every physics substep (4 per 50 Hz control step) and renders each time;
        # only the frame after the last substep is ever observed. Render every `every` substeps (4 = once per control
        # step; 1 = upstream behaviour). Physics stays MuJoCo's, so the observed frames are unchanged.
        import omni.replicator.core as rep
        n = getattr(self, "_rrp_sub", every - 1)
        self._rrp_sub = n + 1
        if every > 1 and (n % every) != every - 1:
            orig = rep.orchestrator.step
            rep.orchestrator.step = lambda *a, **k: None
            try:
                return orig_step(self, *args, **kw)
            finally:
                rep.orchestrator.step = orig
        return orig_step(self, *args, **kw)
    cls.step = step
    orig_reset = getattr(cls, "reset", None)
    if orig_reset is not None:
        def reset(self, *a, _f=orig_reset, **k):
            self._rrp_sub = every - 1        # the reset step renders
            return _f(self, *a, **k)
        cls.reset = reset
    for meth in ("__init__", "update_layout", "reset"):
        f = getattr(cls, meth, None)
        if f is None:
            continue

        def wrap(*args, _f=f, **kw):
            r = _f(*args, **kw)
            _apply_render_profile()
            return r
        setattr(cls, meth, wrap)
    cls.__rrp__ = True


class _AmoPath(importlib.abc.MetaPathFinder):
    """AMO_Policy loads amo_jit.pt / adapter_*.pt from its own dir, where our clone has only Git-LFS pointers (LFS
    skipped). Identical files (sha256 == LFS oid, verified) are in SIMPLE's data/robots/g1: point BASE_DIR there."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "simple.robots.policy.AMO_Policy":
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(fullname, path, target)
            if spec is not None:
                orig_exec = spec.loader.exec_module

                def exec_module(module, _orig=orig_exec):
                    _orig(module)
                    module.BASE_DIR = str(SIMPLE_ROOT / "data/robots/g1")
                    # torch 2.7+cu128 cannot runtime-compile (nvrtc) for sm_121; AMO's TorchScript trace has CUDA
                    # baked in, so keep it on the GPU but disable the TorchScript GPU fusers (no nvrtc codegen)
                    import torch
                    torch._C._jit_override_can_fuse_on_gpu(False)
                    torch._C._jit_set_texpr_fuser_enabled(False)
                    try:
                        torch._C._jit_set_nvfuser_enabled(False)
                    except Exception:
                        pass
                spec.loader.exec_module = exec_module
                return spec
        return None


class _PostImport(importlib.abc.MetaPathFinder):
    """After `simple.engines.isaacsim` is imported (i.e. after SimulationApp started), patch Isaac utilities."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "simple.engines.isaacsim":
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(fullname, path, target)
            if spec is not None:
                orig_exec = spec.loader.exec_module

                def exec_module(module, _orig=orig_exec):
                    _orig(module)          # imports Isaac itself (only happens once SimulationApp runs)
                    if hasattr(module, "isaacsim_stage"):
                        _patch_add_reference(module.isaacsim_stage)
                    _patch_simulator(module)
                spec.loader.exec_module = exec_module
                return spec
        return None


_INSTALLED = False


def install(stub_curobo: bool = True) -> dict:
    global _INSTALLED
    if _INSTALLED:
        return {}
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")          # accepted by the owner (rrp D-098)
    for p in reversed(simple_sys_paths()):
        if p not in sys.path:
            sys.path.insert(0, p)
    # Isaac Sim 5.1 still ships the deprecated omni.isaac.* extensions (verified: psi1z isaac_smoke, P-002), and they only
    # become importable once SimulationApp has started. Aliasing them earlier would hand SIMPLE half-initialized
    # modules ("loaded before SimulationApp was started"), so the alias hook is opt-in for Isaac builds without shims.
    if os.environ.get("RRP_SIMPLE_ISAAC_ALIASES") == "1":
        sys.meta_path.insert(0, _AliasFinder())
    sys.meta_path.insert(0, _PostImport())
    sys.meta_path.insert(0, _AmoPath())
    stubbed = _stub_curobo() if stub_curobo else False
    opt = _stub_optional()
    _INSTALLED = True
    return dict(curobo_stubbed=stubbed, optional_stubbed=opt, paths=simple_sys_paths())


# Upstream SIMPLE @ 803db7e: some task classes declare a `uid` that differs from their registry id, while the eval
# configs carry the registry id, so Task.reset's uid assertion fails (e.g. G1WholebodyXMoveBendPickTeleop: class uid
# "g1_sonic_xmove_bend_pick_teleop", registered/eval "g1_wholebody_xmove_bend_pick_teleop"). Align class uid with the
# registry id (only for the task being evaluated, and only when they differ); logged.
UID_FIXES: list = []


def fix_task_uid(registry_id: str):
    from simple.tasks.registry import TaskRegistry
    cls = TaskRegistry._registry.get(registry_id)
    if cls is not None and getattr(cls, "uid", registry_id) != registry_id:
        UID_FIXES.append((registry_id, cls.uid))
        print(f"[rrp.envs.simple.compat] task uid fix: {cls.__name__}.uid {cls.uid!r} -> {registry_id!r}", flush=True)
        cls.uid = registry_id


if __name__ == "__main__":
    import runpy
    print(f"[rrp.envs.simple.compat] installed: {install()}", flush=True)
    mod = sys.argv[1]
    sys.argv = [mod] + sys.argv[2:]
    runpy.run_module(mod, run_name="__main__", alter_sys=True)
