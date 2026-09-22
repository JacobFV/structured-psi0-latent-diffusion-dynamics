"""Object QA: gradient path (projector/readout/action expert yes, decoder weights no) and
blank/shuffled-object controls. Needs transformers + the Qwen2.5 tokenizer (peer venv/HF cache);
the decoder here is a tiny RANDOM Qwen2 (frozen) so the test is about the mechanics, not the LM."""
import os
import random

import pytest
import torch

transformers = pytest.importorskip("transformers")

from rrp.data.collect import featurizer_for  # noqa: E402
from rrp.model.batch import collate_inputs  # noqa: E402
from rrp.model.flow import FlowPolicy, PolicyConfig  # noqa: E402
from rrp.model.qa import ObjectQA, DECODER, policy_hidden  # noqa: E402
from rrp.sim.fixtures import make_pick_place_session  # noqa: E402


@pytest.fixture(scope="module")
def tiny_lm():
    try:
        tok = transformers.AutoTokenizer.from_pretrained(DECODER["repo_id"], revision=DECODER["revision"],
                                                         local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen2.5 tokenizer not in local HF cache")
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    lm = transformers.Qwen2ForCausalLM(cfg).float()
    return lm, tok


def test_gradients_reach_projector_readout_policy_not_decoder(tiny_lm):
    lm, tok = tiny_lm
    s = make_pick_place_session(seed=3, n_distractors=2)
    f = featurizer_for(s)
    batch = collate_inputs([f(s.observe())] * 2)
    pol = FlowPolicy(PolicyConfig(width=64, heads=2, ctx_layers=1, blocks=2, horizon=4))
    qa = ObjectQA(64, 2, lm, tok, n_prefix=2, source="action")
    N = batch.node_feats.shape[1]
    a = torch.randn(2, 4, N, 1)
    v = torch.ones(2, 4, N, dtype=torch.bool)
    hidden, cache = policy_hidden(pol, batch, a, v, mode="teacher", gen=torch.Generator().manual_seed(0))
    slots = qa.readout(hidden, cache, batch)
    items = [dict(bi=0, a=0, b=-1, q="What color is this object?", ans="red", qtype="color"),
             dict(bi=1, a=0, b=1, q="Which object is closer to the gripper, the first or the second?", ans="first",
                  qtype="closer")]
    qa(slots, items)["nll"].mean().backward()
    gn = lambda ps: sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None)  # noqa: E731
    assert gn(qa.proj.parameters()) > 0
    assert gn(qa.readout.parameters()) > 0
    assert gn(pol.blocks.parameters()) > 0            # through the action-expert hidden states
    assert all(p.grad is None for p in lm.parameters())
    assert not any(p.requires_grad for p in lm.parameters())
    assert not any(k.startswith("_lm") or "embed_tokens" in k for k in qa.state_dict())


def test_blank_and_shuffled_controls_degrade_object_dependent_answers(tiny_lm):
    """Slots carry color identity; after training only the projector, real slots answer color questions and
    blank/shuffled substitutions must fall toward chance."""
    lm, tok = tiny_lm
    torch.manual_seed(0)
    colors = ["red", "blue", "yellow", "purple", "green"]
    codes = torch.randn(5, 32)
    qa = ObjectQA(32, 2, lm, tok, n_prefix=2, source="action")
    opt = torch.optim.Adam(qa.proj.parameters(), lr=3e-3)
    rng = random.Random(0)

    def make(B=16, S=3):
        cid = torch.randint(0, 5, (B, S))
        slots = codes[cid] + 0.05 * torch.randn(B, S, 32)
        items = [dict(bi=b, a=rng.randrange(S), b=-1, q="What color is this object?", qtype="color") for b in range(B)]
        for it in items:
            it["ans"] = colors[int(cid[it["bi"], it["a"]])]
        return slots, items
    for _ in range(400):
        sl, it = make()
        loss = qa(sl, it)["nll"].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    sl, it = make(64)
    with torch.no_grad():
        acc = {m: float(qa(sl, it, substitute=m)["correct"].float().mean()) for m in (None, "blank", "shuffled")}
    # tiny RANDOM frozen decoder => prefix-only learning is capacity-limited; the property under test is
    # that answers depend on the object vector: real >> chance (0.2) and blank/shuffled fall toward chance.
    assert acc[None] > 0.6, acc
    assert acc[None] - max(acc["blank"], acc["shuffled"]) > 0.3, acc
