"""Frozen psi0 System-II VLM (Qwen3-VL backbone) readout used by system II (moved from `rrp.research.system2`, D-126 #31).

Canonical home of the VLM wrapper; `rrp.research.system2.System2` re-exports this class (same object). torch /
transformers are imported lazily (inside `__init__` / `run`), so importing this module never loads or downloads weights.

Readouts (frozen VLM): `score` = logit(" orange") - logit(" cyan") at the answer position (constrained answer scoring),
`feat` = pooled hidden states (text tail + image tokens, layers 16 and 28), `gen` = 4 greedy tokens, `top1`.
"""
from __future__ import annotations

COLORS = ("orange", "cyan")
QUESTION = " Question: which marker should the robot walk to first? Answer with one word: orange or cyan."


class System2:
    """Frozen psi0 System-II VLM wrapper: answer scoring + pooled features."""

    def __init__(self, device=None, local_dir=None):
        import os

        import torch
        from rrp.models.backbone import VLMBackbone, BackboneSpec
        device = device or os.environ.get("RRP_S2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        spec = BackboneSpec()
        if device == "cpu":
            spec.dtype = "float32"
        self.bb = VLMBackbone(spec, device=device, local_dir=local_dir)
        tok = self.bb.processor.tokenizer
        self.cand = {c: tok.encode(" " + c, add_special_tokens=False)[0] for c in COLORS}
        self.cand_nospace = {c: tok.encode(c, add_special_tokens=False)[0] for c in COLORS}

    def provenance(self):
        return self.bb.provenance()

    def run(self, images, texts):
        """images: list of HxWx3 uint8 (one per sample); returns dict(score [B] = logit(orange) - logit(cyan) at the
        answer position, feat [B, F] pooled hidden states, generated text of 4 greedy tokens)."""
        import torch
        with torch.no_grad():
            return self._run(images, texts)

    def _run(self, images, texts):
        import torch
        from PIL import Image
        bb = self.bb
        prompts = [bb.processor.apply_chat_template(bb._messages(1, t + QUESTION), tokenize=False,
                                                    add_generation_prompt=True) for t in texts]
        bb.processor.tokenizer.padding_side = "left"
        inputs = bb.processor(text=prompts, images=[Image.fromarray(im) for im in images], padding=True,
                              return_tensors="pt").to(bb.device)
        out = bb.model(**inputs, output_hidden_states=True, return_dict=True)
        lg = out.logits[:, -1].float()
        sc = torch.maximum(lg[:, self.cand["orange"]], lg[:, self.cand_nospace["orange"]]) - \
            torch.maximum(lg[:, self.cand["cyan"]], lg[:, self.cand_nospace["cyan"]])
        ids = inputs["input_ids"]
        is_img = ids == bb.image_token_id
        feats = []
        for t in (16, 28):
            h = out.hidden_states[t].float()
            txt = h[:, -8:].mean(1)
            vis = (h * is_img[..., None]).sum(1) / is_img.sum(1, keepdim=True).clamp(min=1)
            feats += [txt, vis]
        gen = bb.model.generate(**inputs, max_new_tokens=4, do_sample=False)
        texts_out = bb.processor.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)
        return dict(score=sc.cpu().numpy(), feat=torch.cat(feats, -1).cpu().numpy(), gen=texts_out,
                    top1=[bb.processor.tokenizer.decode([int(i)]) for i in lg.argmax(-1).tolist()])
