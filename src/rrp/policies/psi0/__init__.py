"""Ψ₀ policies on the `simple` env (W10; the policy side of psi1z's serve_psi0 / serve_ours, D-140).

    psi0_direct      Ψ₀ action head over the 30x36 chunk: the RELEASED upstream checkpoint (`weights="released"`,
                     source `learned:psi0-released/<run>/ckpt_<step>`: upstream weights, not trained by us; RTC as
                     released) or our matched fine-tune (`weights=<final.pt>`, source `learned:<final.pt>`, no RTC).
    psi0_structured  Ψ₀ + structure: system i samples the packet z[5, 6, 64] (StructuredHead), `packet_edit=` (a registered
                     `rrp.policies.packets` edit) or `packet_hook(i, z)` may edit it (harness packet edits); `Act.info["packet"]`
                     carries source kind, head / stage-A hashes, packet-use gate and edit; the head checkpoint must
                     carry a passed gate for the stage A it is loaded with and its factor structure is checked, system 0 (Realizer) turns it into the 30x36 chunk. No RTC.
    psi0_replay      recorded training rows of one episode (source `replay:<task>/episode_<e>`; labels / fidelity
                     checks, psi1z replay_labels). Needs `split="train"` on the env.

Pre/post-processing is upstream's: the policies hold an upstream `psi.deploy.serve_psi0_simple.Server` object (image
resize/crop, state padding + normalization, action denormalization, RTC state) and call its model exactly as its
HTTP handler does; our heads replace `Psi0Model.from_pretrained` as psi1z's serve_ours did. No listener is opened.
Every tick: when the observation's `chunk_request` flag is set, return `Act(chunk=...)` with the 30 rows (the env
executes the first 24, as the upstream server returns them); otherwise `Act(command=None)` (the env executes the next
queued row). Needs the psi venv (`ops/bin/psi0_ext.sh psi-env`, extra `rrp[psi0]`); every heavy import is lazy.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from rrp.core.provenance import file_digest
from rrp.envs.simple.compat import psi_home      # THE one ext-dir / PSI_HOME resolver (stdlib-only)

TP, DA, TA = 30, 36, 24

# What BOTH of our arms (direct and structured) are given at every tick (`OursModel.batch`, one code path); a structured
# arm gets nothing else, and its packet is the only new intermediate (D-146 P3). The packet-level edit is recorded apart.
INPUT_SPEC = dict(
    images="ego view through the upstream server's resize + center crop", instruction="task text, lower-cased",
    state="last 36 dims of the proprio state (state0)",
    vlm="frozen Ψ₀ base VLM last hidden states + attention mask",
    entity="mask of the instruction's object-name tokens (pooled), optionally read from an override prompt")


def released_run(task: str) -> Path:
    from rrp.tasks.spec import SIMPLE_TASKS
    return psi_home() / "cache/checkpoints/psi0/simple-checkpoints" / SIMPLE_TASKS[task.split("/", 1)[-1]][0]


def base_vlm() -> Path:
    return psi_home() / "cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"


def object_name(instruction: str) -> str | None:
    m = re.search(r"pick up the (.+?)(?: and |$|\.|,)", instruction)
    return m.group(1).strip() if m else None


def find_subseq(ids: list[int], sub: list[int]) -> list[int]:
    for s in range(len(ids) - len(sub), -1, -1):              # last occurrence (the user turn)
        if ids[s:s + len(sub)] == sub:
            return list(range(s, s + len(sub)))
    return []


def load_launch_config(run_dir: Path, data_root: str | None = None, repo_id: str = "unused"):
    """Upstream LaunchConfig of a released SIMPLE run (model config + normalization + data transform)."""
    from psi.config.config import LaunchConfig  # noqa: F401
    from psi.utils import apply_legacy_model_config_defaults, parse_args_to_tyro_config
    config_ = parse_args_to_tyro_config(Path(run_dir) / "argv.txt")
    conf = apply_legacy_model_config_defaults(json.loads((Path(run_dir) / "run_config.json").read_text()))
    conf["data"]["root_dir"] = data_root or str(psi_home() / "data/simple")
    conf["data"]["train_repo_ids"] = [repo_id]
    return config_.model_validate(conf)


def psi_runtime():
    """Once per process: psi-env environment (was psi1z scripts/psienv.sh) and the lighter checkpoint load (P-008:
    upstream mean-resizes the 151k-row embedding and then overwrites every row with the strict load; disabling mean
    resizing is result-neutral and removes several GB from the load peak)."""
    for k, v in dict(PSI_HOME=str(psi_home()), HF_HOME=str(psi_home() / "cache/hf"), TOKENIZERS_PARALLELISM="false",
                     PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python", NO_ALBUMENTATIONS_UPDATE="1", WANDB_MODE="disabled").items():
        os.environ.setdefault(k, v)
    import transformers
    orig = transformers.PreTrainedModel.resize_token_embeddings
    if getattr(orig, "__rrp__", False):
        return

    def resize_token_embeddings(self, *a, **k):
        k["mean_resizing"] = False
        return orig(self, *a, **k)
    resize_token_embeddings.__rrp__ = True
    transformers.PreTrainedModel.resize_token_embeddings = resize_token_embeddings


def structured_provenance(state: dict, stage_a: str) -> dict:
    """What a structured-head checkpoint (`save_checkpoint` state) says about its own lineage, verified against the
    stage-A file the policy loads: the packet-use gate it was trained under must have passed for THIS stage A
    (architecture 14.5 c), else the policy refuses (D-141: a head over an R that ignores the packet is not the arm)."""
    gate = (state.get("config") or {}).get("packet_gate")
    sha = file_digest(stage_a)
    if not gate or not gate.get("passed") or gate.get("stage_a_sha256_16") != sha:
        raise ValueError(f"psi0_structured: head was not trained under a passed packet-use gate for stage A {stage_a} "
                         f"(gate={gate}, stage-A sha256_16={sha}); retrain through `rrp train psi0 gate`")
    return dict(stage_a=sha, packet_gate=dict(gap=gate["gap"], margin=gate["margin"]),
                factors=(state.get("versions") or {}).get("factors"))


class OursModel:
    """Frozen Ψ₀ base VLM (the SAME weights that produced the cached training features) + our trained head, with
    upstream Psi0Model's `predict_action` signature (psi1z serve_ours)."""

    def __init__(self, arm, ckpt, stage_a, vlm, run_dir, device="cuda:0"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from rrp.policies.psi0 import nets as N
        self.device, self.arm = device, arm
        self.vlm_processor = AutoProcessor.from_pretrained(vlm)
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(vlm, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        mcfg = load_launch_config(Path(run_dir)).model
        self.provenance = dict(arm=arm, head=file_digest(ckpt), inputs=INPUT_SPEC)
        if arm == "direct":
            self.head = N.DirectHead(mcfg)
            self.head.load_state_dict(torch.load(ckpt, weights_only=False)["model"], strict=True)
        else:
            A = N.load_stage_a(stage_a)
            zs = torch.load(Path(stage_a).parent / "z_stats.pt")
            self.head = N.StructuredHead(mcfg, A, zs["mean"], zs["std"])
            self.provenance.update(structured_provenance(torch.load(ckpt, weights_only=False, map_location="cpu"), stage_a))
            N.load_structured(self.head, ckpt)              # factor structure (context tokens + stage A) must match
        self.head.to(device).eval()
        self.entity_override = None      # input-channel edit: pool the object token from another object's name
        self.packet_hook = None          # z -> z (per-episode harness hook); structured only
        self.packet_edit = None          # (name, kwargs) of a registered `rrp.policies.packets` edit; structured only
        self.last_z = self.generated_z = None

    # upstream Server calls .to(device) and .eval() on the model
    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        yield from self.head.parameters()
        yield from self.vlm.parameters()

    def vlm_features(self, observations, instructions):
        import torch
        from qwen_vl_utils import process_vision_info
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im} for im in obs] + [{"type": "text", "text": ins}]}]
                for obs, ins in zip(observations, instructions)]
        texts = [self.vlm_processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        ii, vi = process_vision_info(msgs, image_patch_size=16)
        inp = self.vlm_processor(text=texts, images=ii, videos=vi, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.vlm(**inp, output_hidden_states=True, return_dict=True).hidden_states[-1]
        mask = inp["attention_mask"].bool()
        ent = torch.zeros_like(mask)
        tok = self.vlm_processor.tokenizer
        for j, ins in enumerate(instructions):
            ids = inp["input_ids"][j][mask[j]].tolist()
            name = object_name(ins) or ""
            pos = []
            for variant in (" " + name, name):
                pos = find_subseq(ids, tok.encode(variant, add_special_tokens=False)) if name else []
                if pos:
                    break
            if pos:
                ent[j, pos] = True
        return h.to(torch.bfloat16), mask, ent

    def batch(self, observations, states, instructions) -> dict:
        """The head input, identical for both arms (`INPUT_SPEC`)."""
        h, mask, ent = self.vlm_features(observations, instructions)
        b = dict(hidden=h, mask=mask, ent=ent, state0=states[:, -1, :36].float())
        if self.entity_override:
            alt = [re.sub(r"pick up the .+?($| and |\.|,)", f"pick up the {self.entity_override}\\1", i) for i in instructions]
            h2, _, e2 = self.vlm_features(observations, alt)
            b["ent_hidden"], b["ent"] = h2, e2
        return b

    def edited(self, z):
        """The packet system 0 receives: the generated z after the optional edit (registered edit or per-episode hook)."""
        if self.packet_edit is not None and self.packet_hook is not None:
            raise ValueError("set packet_edit or packet_hook, not both")
        if self.packet_edit is not None:
            from rrp.policies.packets import apply_edit
            name, kw = self.packet_edit
            return apply_edit(name, z, **({"mean": self.head.z_mean} if name == "mean_packet" and "mean" not in kw else {}), **kw)
        return self.packet_hook(z) if self.packet_hook is not None else z

    def predict_action(self, observations, states, instructions, num_inference_steps=10, traj2ds=None, **kw):
        import torch
        with torch.no_grad():
            b = self.batch(observations, states, instructions)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if self.arm == "structured":
                    self.generated_z = self.head.sample_z(b, nfe=num_inference_steps)
                    z = self.last_z = self.edited(self.generated_z)
                    a = self.head.realize(z, b["state0"], torch.zeros(z.shape[0], device=z.device))
                else:
                    a = self.head.sample(b, nfe=num_inference_steps)
        return a.float()


def build_server(run_dir: Path, *, ours: dict | None = None, rtc: bool = True, device: str = "cuda:0", seed: int | None = None):
    """Upstream Server (not started): released weights, or `ours=dict(arm, ckpt, stage_a, vlm)` in place of Psi0Model."""
    psi_runtime()
    import psi.deploy.serve_psi0_simple as S
    import psi.models.psi0 as P
    if seed is not None:
        from psi.utils import seed_everything as _se
        S.seed_everything = lambda _s: _se(seed)
    orig = P.Psi0Model.from_pretrained
    if ours is not None:
        P.Psi0Model.from_pretrained = staticmethod(lambda run_dir, ckpt_step, lc, device: OursModel(run_dir=run_dir, device=device, **ours))
    try:
        return S.Server("psi0", Path(run_dir), 40000, device, enable_rtc=rtc, action_exec_horizon=TA)
    finally:
        P.Psi0Model.from_pretrained = orig


def server_infer(srv, image: np.ndarray, state32: np.ndarray, instruction: str, reset: bool) -> np.ndarray:
    """The body of upstream Server.predict_action without HTTP: returns the DENORMALIZED 30 x 36 chunk (the server
    returns its first Ta rows; the env executes those)."""
    import torch
    from PIL import Image
    from torchvision.transforms import v2
    t = v2.Compose([srv.model_transform.resize(), srv.model_transform.center_crop()])
    images = [[t(Image.fromarray(np.clip(np.asarray(image), 0, 255).astype(np.uint8)))]]
    states = srv._prepare_states({"states": np.asarray(state32, np.float32)[None]})
    common = dict(observations=images, states=states.unsqueeze(0), instructions=[str(instruction).lower()],
                  num_inference_steps=srv.num_inference_steps, traj2ds=None)
    with torch.no_grad():
        if not srv.enable_rtc or srv.previous_action is None or reset:
            raw = srv.model.predict_action(**common)
        elif srv.rtc_mode == "test_time":
            raw = srv.model.predict_action_with_rtc_flow(**common, prev_actions=srv._shifted_prev_actions(),
                                                        inference_delay=srv.rtc_inference_delay,
                                                        execution_horizon=srv.min_exec_horizon,
                                                        mask_schedule=srv.pig_mask_schedule,
                                                        guidance_alpha=srv.pig_guidance_alpha)
        else:
            raw = srv.model.predict_action_with_training_rtc_flow(**common, prev_actions=srv._shifted_prev_actions(),
                                                                 inference_delay=srv.rtc_inference_delay,
                                                                 max_delay=srv.rtc_max_delay)
    raw = raw.reshape(-1, srv.Da).float().cpu().numpy()
    srv.previous_action = raw.copy().astype(np.float32)
    return np.asarray(srv.maxmin.denormalize(raw), np.float32)


def _channel(obs, name):
    for c in obs.declared_sensor_channels:
        if c.name == name:
            return np.asarray(c.values)
    raise KeyError(name)


class Psi0Policy:
    """`psi0_direct` / `psi0_structured` / `psi0_replay` (see module docstring). One instance serves `len(seeds)`
    parallel envs sequentially (one model; the RTC state is per episode). Nothing heavy happens before `reset`."""

    def __init__(self, kind: str, *, task: str = "simple/G1WholebodyTabletopGraspMP-v0", weights: str = "released", stage_a: str | None = None,
                 vlm: str | None = None, rtc: bool | None = None, seed: int = 0, nfe: int = 10,
                 entity_override: str | None = None, packet_edit: dict | None = None, data_root: str | None = None,
                 device: str = "cuda:0"):
        """`packet_edit=dict(name=<rrp.policies.packets.EDITS key>, **kwargs)` (structured only): system 0 receives the
        edited packet; `packet_hook(i, z)` is the free-form per-episode alternative (not both)."""
        if kind not in ("psi0_direct", "psi0_structured", "psi0_replay"):
            raise ValueError(kind)
        self.kind, self.task, self.nfe, self.device, self.seed = kind, task.split("/", 1)[-1], nfe, device, seed
        self.packet_hook = None
        if packet_edit is not None:
            from rrp.policies.packets import EDITS
            if kind != "psi0_structured" or packet_edit.get("name") not in EDITS:
                raise ValueError(f"packet_edit={packet_edit!r}: needs psi0_structured and a name in {sorted(EDITS)}")
            packet_edit = dict(packet_edit)
            packet_edit = (packet_edit.pop("name"), packet_edit)
        self.packet_edit = packet_edit
        run = released_run(self.task)
        if kind == "psi0_replay":
            self.data_root = Path(data_root or psi_home() / "data/simple") / self.task
            self.ours, self.rtc = None, False
            source, version = f"replay:{self.task}", "recorded"
        elif kind == "psi0_direct" and weights == "released":
            self.ours, self.rtc = None, True if rtc is None else rtc
            source, version = f"learned:psi0-released/{run.name}/ckpt_40000", f"{run.name}/ckpt_40000"
        else:
            if kind == "psi0_structured" and weights and not stage_a:
                raise ValueError("psi0_structured needs stage_a=<stage_a.pt> (with z_stats.pt next to it)")
            self.ours = dict(arm="direct" if kind == "psi0_direct" else "structured", ckpt=weights, stage_a=stage_a,
                             vlm=vlm or str(base_vlm()))
            self.rtc = False if rtc is None else rtc     # our arms were trained without RTC
            source, version = f"learned:{weights or '<weights>'}", "unhashed"   # sha256 of the weights at reset
        self.run, self.entity_override = run, entity_override
        self.info = _policy_info(kind, source, version, self.task)
        self._server = None

    # ------------------------------------------------------------------ Policy
    def reset(self, spec, task, seeds, *, envs=None):
        import dataclasses
        if self.ours is not None and not self.ours["ckpt"]:
            raise ValueError(f"{self.kind} needs weights=<final.pt>")
        if self.info.version == "unhashed":
            self.info = dataclasses.replace(self.info, version=file_digest(self.ours["ckpt"]))
        self._reset_flags = [True] * len(seeds)
        self.seeds = list(seeds)
        if self.kind == "psi0_replay":
            self._rows, self._ptr = [self.recorded_rows(e) for e in self.seeds], [0] * len(seeds)
            return
        if self._server is None:
            self._server = build_server(self.run, ours=self.ours, rtc=self.rtc, device=self.device, seed=self.seed)
            if self.ours is not None:
                self._server.model.entity_override = self.entity_override
        self._prev = [None] * len(seeds)

    def recorded_rows(self, episode: int) -> np.ndarray:
        """[T, 36] recorded Ψ₀ command rows of a training episode (LeRobot parquet)."""
        import pandas as pd
        df = pd.read_parquet(self.data_root / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet")
        return np.stack(df["action"].to_numpy()).astype(np.float32)

    def act(self, obs):
        from rrp.policies.base import Act
        acts = {}
        for i, o in obs.items():
            if _channel(o, "chunk_request")[0] < 0.5:
                acts[i] = Act(command=None)
                continue
            if self.kind == "psi0_replay":
                p = self._ptr[i]; rows = self._rows[i][p:p + TA]; self._ptr[i] = p + TA
                acts[i] = (Act(command=None, info=dict(end_of_recording=True)) if len(rows) == 0 else
                           Act(command=None, chunk=_chunk(rows, o, self.info)))
                continue
            srv = self._server
            srv.previous_action = self._prev[i]                     # RTC state per env
            if self.ours is not None:
                srv.model.packet_hook = (lambda z, _i=i: self.packet_hook(_i, z)) if self.packet_hook else None
                srv.model.packet_edit = self.packet_edit
            chunk = server_infer(srv, o.sensor_images[0].pixels, _channel(o, "psi0_state"), o.instruction,
                                 reset=self._reset_flags[i])
            self._prev[i], self._reset_flags[i] = srv.previous_action, False
            info = {}
            if self.ours is not None:
                info["inputs"] = INPUT_SPEC                       # identical for the direct and structured arm
            if self.kind == "psi0_structured":
                m = srv.model
                edited = self.packet_edit is not None or self.packet_hook is not None
                info["packet_z"] = m.last_z.float().cpu().numpy()[0]
                info["packet"] = dict(m.provenance, source="edited:debug" if edited else "system_i_generated",
                                      edit=self.packet_edit[0] if self.packet_edit else ("hook" if self.packet_hook else None))
                if edited:
                    info["packet_z_generated"] = m.generated_z.float().cpu().numpy()[0]     # before the edit
            acts[i] = Act(command=None, chunk=_chunk(chunk, o, self.info), info=info)
        return acts


def make_direct(**kw) -> Psi0Policy:
    """Registry factory: `make_policy("psi0_direct", task=..., weights="released" | <final.pt>)`."""
    return Psi0Policy("psi0_direct", **kw)


def make_structured(**kw) -> Psi0Policy:
    """Registry factory: `make_policy("psi0_structured", task=..., weights=<final.pt>, stage_a=<stage_a.pt>)`."""
    kw.setdefault("weights", None)
    return Psi0Policy("psi0_structured", **kw)


def make_replay(**kw) -> Psi0Policy:
    """Recorded training rows (labels / fidelity; env split="train"). Not a learned policy."""
    return Psi0Policy("psi0_replay", **kw)


def _chunk(rows, obs, info):
    from rrp.core.action import ActionChunk, GroupCommand
    rows = np.asarray(rows, np.float32)
    return ActionChunk(observation_id=obs.observation_id, graph_version=0, runtime_version=0,
                       robot_spec_hash=obs.robot_spec_hash, controller_version="simple_agent", policy_version=info.version,
                       codec_version=None, start_time=obs.sensor_time, dt=0.02, horizon=len(rows),
                       command_groups=[GroupCommand(group="psi0", values=rows, mask=np.ones(rows.shape, bool))],
                       sampling_seed=None, source="replay" if info.source.startswith("replay") else "learned")


def _policy_info(kind, source, version, task):
    from rrp.policies.base import PolicyInfo, Requirements
    req = Requirements(action_kinds=frozenset({"psi0"}), groups=frozenset({"psi0"}),
                       observations=frozenset({"proprio"} if kind == "psi0_replay" else {"proprio", "images", "language"}),
                       bodies=frozenset({"g1_simple"}), tasks=frozenset({f"simple/{task}"}),
                       env_capabilities=frozenset({"chunk_executor"}))
    return PolicyInfo(name=kind, source=source, version=version, requires=req,
                      variant="released" if "psi0-released" in source else None)
