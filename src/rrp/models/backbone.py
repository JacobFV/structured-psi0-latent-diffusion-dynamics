"""Frozen VLM backbone adapter (system ii) + bounded learned resampler.

(images, task text) -> Qwen3-VL hidden sequence at tapped layers -> selected tokens
(all visual tokens + the last `n_text` sequence tokens: end of instruction + generation
prompt) -> Resampler (Q learned queries) -> image_tokens [B, Q, D] for the scene bank.

Preferred weights: psi0 System-II `USC-PSI-Lab/psi-model/psi0/pre.fast.2605160748.ckpt.ego390k`
(standard HF Qwen3VLForConditionalGeneration dir; vocab 153,792 incl. FAST action tokens).
Weights have NO declared license on HF -> local research use only, never redistribute.
Fallback (explicitly labelled): base `Qwen/Qwen3-VL-2B-Instruct` (Apache-2.0).

Every feature carries provenance: repo/revision/subfolder, tokenizer + processor classes,
processor hash, hidden width, number of layers, tapped layers, token selection, precision.
The VLM is frozen (requires_grad False, eval mode); only the Resampler learns.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PSI0 = dict(label="psi0_system2", repo_id="USC-PSI-Lab/psi-model",
            revision="4c6f9776fc5b18d87945254175e38bb74b9d7748",
            subfolder="psi0/pre.fast.2605160748.ckpt.ego390k",
            license="none declared on HF (GitHub repo Apache-2.0; weight coverage UNVERIFIED) - local research only")
FALLBACK = dict(label="fallback_not_psi0_qwen3vl_2b_instruct", repo_id="Qwen/Qwen3-VL-2B-Instruct",
                revision="89644892e4d85e24eaac8bacfd4f463576704203", subfolder="", license="apache-2.0")


@dataclass
class BackboneSpec:
    source: dict = field(default_factory=lambda: dict(PSI0))
    taps: tuple = (16, 28)          # hidden_states indices (0 = embeddings, 28 = last layer)
    n_text: int = 8                 # last sequence tokens kept (instruction tail + generation prompt)
    dtype: str = "bfloat16"
    attn: str = "sdpa"              # flash-attn on aarch64 unverified; not used

    def key(self) -> str:
        return hashlib.sha256(json.dumps(dict(src=self.source["repo_id"], rev=self.source["revision"],
                                              sub=self.source["subfolder"], taps=list(self.taps),
                                              n_text=self.n_text, dtype=self.dtype), sort_keys=True).encode()
                              ).hexdigest()[:16]


def resolve_local(source: dict) -> Path:
    """Local snapshot dir (downloads only if absent; HF_HOME decides where)."""
    from huggingface_hub import snapshot_download
    pat = [f"{source['subfolder']}/*"] if source["subfolder"] else None
    root = snapshot_download(source["repo_id"], revision=source["revision"], allow_patterns=pat)
    return Path(root) / source["subfolder"] if source["subfolder"] else Path(root)


def _file_hash(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        p = Path(p)
        if p.exists():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def image_hash(img: np.ndarray) -> str:
    a = np.ascontiguousarray(img)
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()[:16]


def text_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class VLMBackbone:
    """Frozen Qwen3-VL feature extractor. Not an nn.Module on purpose: it is never trained and
    is never saved inside policy checkpoints (only its provenance is)."""

    def __init__(self, spec: BackboneSpec | None = None, device="cuda", local_dir: str | None = None):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.spec = spec or BackboneSpec()
        self.device = torch.device(device)
        path = Path(local_dir) if local_dir else resolve_local(self.spec.source)
        self.path = path
        dt = getattr(torch, self.spec.dtype)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(str(path), dtype=dt,
                                                                     attn_implementation=self.spec.attn)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # processor from the checkpoint dir (tokenizer includes FAST tokens; prompts contain none).
        # psi0 inference loads the processor from the base id instead; equivalence UNVERIFIED.
        self.processor = AutoProcessor.from_pretrained(str(path))
        cfg = self.model.config
        tc = cfg.text_config
        self.width = tc.hidden_size
        self.n_layers = tc.num_hidden_layers
        assert all(0 <= t <= self.n_layers for t in self.spec.taps), self.spec.taps
        self.image_token_id = cfg.image_token_id
        self.proc_hash = _file_hash(sorted(path.glob("*.json")) + sorted(path.glob("*.jinja")))

    def provenance(self) -> dict:
        tok = self.processor.tokenizer
        return dict(source=self.spec.source, local_path=str(self.path), spec_key=self.spec.key(),
                    architecture=type(self.model).__name__, taps=list(self.spec.taps), n_text=self.spec.n_text,
                    hidden_width=self.width, n_layers=self.n_layers, vocab=len(tok),
                    embed_rows=self.model.get_input_embeddings().num_embeddings,
                    tokenizer=type(tok).__name__, processor=type(self.processor).__name__,
                    image_processor=type(self.processor.image_processor).__name__, processor_hash=self.proc_hash,
                    dtype=self.spec.dtype, attn=self.spec.attn, frozen=True,
                    n_params=sum(p.numel() for p in self.model.parameters()))

    def _messages(self, n_images: int, text: str):
        content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": text}]
        return [{"role": "user", "content": content}]

    @torch.no_grad()
    def encode(self, images: list[list[np.ndarray]], texts: list[str]) -> dict:
        """images: per sample, a list of HxWx3 uint8 arrays (same count/size across the batch).
        Returns tokens [B, L, n_taps, width] (fp16, CPU) and n_visual (tokens per sample)."""
        from PIL import Image
        B = len(images)
        prompts = [self.processor.apply_chat_template(self._messages(len(ims), tx), tokenize=False,
                                                      add_generation_prompt=True) for ims, tx in zip(images, texts)]
        flat = [Image.fromarray(np.asarray(im, np.uint8)) for ims in images for im in ims]
        self.processor.tokenizer.padding_side = "left"      # last n_text tokens align at the end
        inputs = self.processor(text=prompts, images=flat, padding=True, return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hs = torch.stack([out.hidden_states[t] for t in self.spec.taps], 2)     # [B, S, T, W]
        ids = inputs["input_ids"]
        is_img = ids == self.image_token_id
        nv = int(is_img[0].sum())
        if not bool((is_img.sum(1) == nv).all()):
            raise ValueError("variable visual token counts in batch; render all images at one size")
        vis = hs[is_img].view(B, nv, *hs.shape[2:])
        txt = hs[:, -self.spec.n_text:]
        tokens = torch.cat([vis, txt], 1).to(torch.float16).cpu()
        return dict(tokens=tokens, n_visual=nv, grid_thw=inputs["image_grid_thw"].cpu().tolist()[:len(images[0])])


# ------------------------------------------------------------------ learned resampler
class Resampler(nn.Module):
    """Bounded learned resampler: per-tap LayerNorm + softmax tap mix -> Linear(W->D)
    + type embedding (visual/text) + position -> `queries` learned queries cross-attending
    (layers x [xattn, MLP]) -> [B, queries, D]. Bounded output size regardless of input length."""

    def __init__(self, in_width: int, n_taps: int, dim: int = 256, queries: int = 16, layers: int = 2,
                 heads: int = 4, max_tokens: int = 512):
        super().__init__()
        self.cfg = dict(in_width=in_width, n_taps=n_taps, dim=dim, queries=queries, layers=layers, heads=heads,
                        max_tokens=max_tokens)
        self.norms = nn.ModuleList([nn.LayerNorm(in_width) for _ in range(n_taps)])
        self.tap_w = nn.Parameter(torch.zeros(n_taps))
        self.proj = nn.Linear(in_width, dim)
        self.type_emb = nn.Embedding(2, dim)
        self.pos = nn.Embedding(max_tokens, dim)
        self.q = nn.Parameter(torch.randn(queries, dim) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict(dict(nq=nn.LayerNorm(dim), nk=nn.LayerNorm(dim),
                                                  att=nn.MultiheadAttention(dim, heads, batch_first=True),
                                                  nm=nn.LayerNorm(dim),
                                                  mlp=nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                                                    nn.Linear(4 * dim, dim)))))
        self.out = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, n_visual: int) -> torch.Tensor:
        """tokens [B, L, T, W] -> [B, Q, D]"""
        B, L, T, W = tokens.shape
        x = tokens.float()
        w = torch.softmax(self.tap_w, 0)
        x = sum(w[t] * self.norms[t](x[:, :, t]) for t in range(T))
        typ = torch.zeros(L, dtype=torch.long, device=x.device)
        typ[n_visual:] = 1
        x = self.proj(x) + self.type_emb(typ)[None] + self.pos.weight[:L][None]
        q = self.q[None].expand(B, -1, -1)
        for b in self.blocks:
            k = b["nk"](x)
            q = q + b["att"](b["nq"](q), k, k, need_weights=False)[0]
            q = q + b["mlp"](b["nm"](q))
        return self.out(q)


def task_text(session) -> str:
    """Instruction text built from the PUBLIC task/object declarations (descriptors a detector reports)."""
    objs = {o.task_entity: o.descriptor for o in session.scenario.objects if getattr(o, "task_entity", None)}
    if session.scenario.name == "pick_place" and "cube" in objs:
        return f"Pick up the {objs['cube']} and place it in the {objs.get('target', 'target zone')}."
    return f"Perform the task: {session.scenario.name.replace('_', ' ')}."


class Renderer:
    """Offscreen RGB renderer bound to one session's model (MUJOCO_GL=egl on the peer)."""

    def __init__(self, model, size: int = 256):
        import mujoco
        self.r = mujoco.Renderer(model, size, size)
        self.cams = [model.camera(i).name for i in range(model.ncam)]

    def cameras(self, want: list[str]) -> list[str]:
        out = []
        for w in want:
            if w in self.cams:
                out.append(w)
            else:                       # robot cameras are prefixed by surgery (e.g. r0/wrist_cam)
                m = [c for c in self.cams if c.endswith(w)]
                if m:
                    out.append(m[0])
        return out

    def render(self, data, cams: list[str]) -> list[np.ndarray]:
        out = []
        for c in cams:
            self.r.update_scene(data, camera=c)
            out.append(self.r.render().copy())
        return out

    def close(self):
        self.r.close()
