"""D-146 C2 (architecture 14.6): the pointer copy head, the `cw_pointer_eng.v2` key code and the v2 split."""
import random
import zlib

import numpy as np
import pytest
import torch

from rrp.policies.pointer import (ENG_VERSION, ENG_VERSION_V2, KEY_BITS, LC, LI, NH, NW, WF, EventHistory, codes,
                                  decode_slot, encode_commands, key_bits, key_from_bits, nets)


def test_v2_key_code_roundtrips_over_the_whole_key_vocab():
    from rrp.envs.computerworld import KEY_VOCAB
    n = len(KEY_VOCAB)
    assert n + 1 <= 2 ** KEY_BITS
    for key in [-1] + list(range(n)):
        assert key_from_bits(key_bits(key + 1, KEY_BITS)) == key + 1
        z = encode_commands([{"pointer": [0.1, 0.2], "button": [0.0], **({"key": [float(key)]} if key >= 0 else {})}],
                            0.1, n, version=ENG_VERSION_V2)
        assert z.shape == (4, 1, 26)
        d = decode_slot(z, 0, 1, n, version=ENG_VERSION_V2)
        assert d["key"] == key and (d["x"], d["y"]) == pytest.approx((0.1, 0.2))
    assert set(np.unique(z[0, 0, 13 + 6:13 + 13])) <= {-1.0, 1.0}
    bad = z.copy()
    bad[0, 0, 13 + 6:13 + 13] = 1.0                                   # class 127 is past the vocabulary
    assert decode_slot(bad, 0, 1, n, version=ENG_VERSION_V2)["key"] == -1


def test_v1_scalar_key_stays_decodable_by_version_tag():
    cmds = [None, {"pointer": [0.1, 0.0], "button": [1.0], "key": [17.0]}]
    z1 = encode_commands(cmds, 0.1, 109)
    assert z1.shape == (4, 1, 14) and ENG_VERSION == "cw_pointer_eng.v1"
    assert decode_slot(z1, 1, 0, 109)["key"] == 17 == decode_slot(z1, 1, 0, 109, version=ENG_VERSION)["key"]
    assert [decode_slot(encode_commands(cmds, 0.1, 109, version=v), 1, 0, 109, version=v)["key"]
            for v in (ENG_VERSION, ENG_VERSION_V2)] == [17, 17]
    with pytest.raises(ValueError, match="one of"):
        decode_slot(z1, 1, 0, 109, version="cw_pointer_eng.v0")


def test_event_history_counts_printable_keys_only():
    h = EventHistory()
    from rrp.envs.computerworld import key_index
    for t, sym in enumerate(["Enter", "a", "b", "Backspace", "c"]):
        h.push(t, {"pointer": [0.0, 0.0], "button": [0.0], "key": [float(key_index(sym))]}, (1.0, 1.0))
    assert h.n_typed == 3


def _synthetic_batch(strings, ntyped, rng):
    """A public batch for `type '<s>'.` instructions with no widgets (the copy head needs only the instruction)."""
    B = len(strings)
    b = dict(wch=np.zeros((B, NW, LC), int), wrole=np.zeros((B, NW), int), wbound=np.zeros((B, NW), int),
             wf=np.zeros((B, NW, WF), np.float32), wmask=np.zeros((B, NW), bool),
             wpos3d=np.zeros((B, NW, 3), np.float32), wcamuvd=np.zeros((B, NW, 3), np.float32),
             instr=np.stack([codes(f"Open the Text Editor and type '{s}'.", LI) for s in strings]).astype(int),
             ptr=np.zeros((B, 2), np.float32), btn=np.zeros(B, np.float32), tick=np.zeros(B, np.float32),
             hist=np.zeros((B, NH, 5), np.float32), ntyped=np.asarray(ntyped, np.float32))
    return {k: torch.from_numpy(np.asarray(v)) for k, v in b.items()}


def _strings(rng, n, alphabet, held: bool):
    out = []
    while len(out) < n:
        s = "".join(rng.choice(alphabet) for _ in range(rng.choice((3, 5, 8))))
        if (zlib.crc32(s.encode()) % 8 == 0) == held:
            out.append(s)
    return out


def _targets(strings, ntyped, H=7):
    from rrp.envs.computerworld import key_index
    y = torch.zeros(len(strings), H, dtype=torch.long)
    for i, (s, n) in enumerate(zip(strings, ntyped)):
        for j in range(H):
            if n + j < len(s):
                y[i, j] = 1 + key_index(s[n + j])
    return y


@pytest.mark.slow                                                   # 115 s on the host (trains the copy head)
def test_copy_head_types_instr_n_typed_on_unseen_strings_and_characters():
    """Trained on strings from the train hash bucket over letters a-l, the mixture head's next character
    (slot 0) equals instr[n_typed] on UNSEEN strings (held-out bucket) and on strings over letters never trained on
    (m-z): the copy path reads the instruction, not a memorized character table."""
    torch.manual_seed(0)
    rng = random.Random(0)
    torch.set_num_threads(2)                                            # tiny net: more threads only add contention
    net = nets()["PointerBC"](D=32, heads=4, layers=1, copy_key=True)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    for _ in range(250):
        ss = _strings(rng, 32, "abcdefghijkl", held=False)
        nt = [rng.randrange(0, len(s) + 1) for s in ss]
        _, _, lp = net(_synthetic_batch(ss, nt, rng))
        loss = torch.nn.functional.nll_loss(lp.reshape(-1, lp.shape[-1]), _targets(ss, nt).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    for alphabet, held in (("abcdefghijkl", True), ("mnopqrstuvwxyz", True), ("mnopqrstuvwxyz", False)):
        ss = _strings(rng, 200, alphabet, held)
        nt = [rng.randrange(0, len(s)) for s in ss]
        with torch.no_grad():
            lp = net(_synthetic_batch(ss, nt, rng))[2]
        y = _targets(ss, nt)
        assert float((lp.argmax(-1)[:, 0] == y[:, 0]).float().mean()) > 0.95, (alphabet, held)


def test_copy_head_mixture_is_a_normalised_distribution_and_default_nets_are_unchanged():
    torch.manual_seed(0)
    rng = random.Random(1)
    ss = _strings(rng, 4, "abcdef", False)
    b = _synthetic_batch(ss, [0, 1, 2, 3], rng)
    net = nets()["PointerBC"](D=32, heads=4, layers=1, copy_key=True).eval()
    t, _ = net.ctx(b)
    lp, alpha, g = net.copy.terms(t[:, :7], t, b["instr"])
    assert torch.allclose(lp.exp().sum(-1), torch.ones(4, 7), atol=1e-4)
    assert torch.allclose(alpha.sum(-1), torch.ones(4, 7), atol=1e-4) and float(g.detach().max()) <= 1.0
    assert not hasattr(nets()["PointerBC"](D=32, heads=4, layers=1), "copy")           # opt-in: default has no head
    assert not nets()["PointerFlow"](dz=8, D=32, heads=4, layers=1).copy_key


def test_flow_copy_key_writes_the_head_argmax_code_and_backprops_into_the_head():
    torch.manual_seed(0)
    rng = random.Random(2)
    ss = _strings(rng, 3, "abcdef", False)
    b = _synthetic_batch(ss, [0, 1, 2], rng)
    S = nets()["PointerFlow"](dz=26, D=32, heads=4, layers=1, copy_key=True)
    z = torch.zeros(3, 4, 1, 26)
    from rrp.envs.computerworld import key_index
    for i, w in enumerate(ss):                                          # slot 0 = the first character of the string (copyable)
        z[i, :, 0, 6:13] = torch.as_tensor(key_bits(1 + key_index(w[0]), 7))
        z[i, :, 0, 19:26] = torch.as_tensor(key_bits(0, 7))
    loss, logs = S.loss(b, z)
    loss.backward()
    assert "key" in logs and S.copy.q.weight.grad is not None and float(S.copy.q.weight.grad.abs().sum()) > 0
    S.eval()
    zs = S.sample(b, nfe=2)
    cls = S.key_logp(b, S.ctx(b)).argmax(-1)
    for sl in range(2):
        o = sl * 13 + 6
        assert [key_from_bits(zs[i, k, 0, o:o + 7]) for i in range(3) for k in range(4)] == cls[..., sl].flatten().tolist()
    with pytest.raises(ValueError, match="v2 packet"):
        nets()["PointerFlow"](dz=14, D=32, heads=4, layers=1, copy_key=True)


def test_split_v2_is_disjoint_and_declared_by_rule():
    from rrp.envs.computerworld import SEED_GOALS, STRING_SOURCES, WORDS, goal_for
    from rrp.harness.train.pointer import TASKS, heldout_goal
    from rrp.harness.train.pointer.split import SPLIT_PATH_V2, excluded_seeds, load_split
    s = load_split(SPLIT_PATH_V2)
    assert s["split_id"] == "cworld_pointer_v2" and s["env_kw"] == {"strings": "procedural"} and "procedural" in STRING_SOURCES
    lo, hi = s["seed_ranges"]["train"]
    for t in TASKS:
        sd = s["seeds"][t]
        ev = [x for k in ("dev", "sealed_id", "sealed_heldout") for x in sd.get(k, [])]
        assert ev and len(ev) == len(set(ev)) and not any(lo <= x < hi for x in ev)
        assert excluded_seeds(s, t) == set(ev)
        for k, want in (("dev", False), ("sealed_id", False), ("sealed_heldout", True)):
            for x in sd.get(k, []):
                g = goal_for(t, x, "procedural") if t in SEED_GOALS else {}
                assert heldout_goal(t, g, s) == want, (t, k, x)
    # a string is held out by its own bytes: train (non-held-out) and sealed_heldout string sets are disjoint
    held = {goal_for("cw/open_type", x, "procedural")["text"] for x in s["seeds"]["cw/open_type"]["sealed_heldout"]}
    ind = {goal_for("cw/open_type", x, "procedural")["text"] for k in ("dev", "sealed_id")
           for x in s["seeds"]["cw/open_type"][k]}
    train = {goal_for("cw/open_type", x, "procedural")["text"] for x in range(0, 3000)
             if not heldout_goal("cw/open_type", goal_for("cw/open_type", x, "procedural"), s)}
    assert held and not (held & ind) and not (held & train) and not (held & set(WORDS))
    assert all(zlib.crc32(w.encode()) % 8 == 0 for w in held)


def test_procedural_strings_are_injective_seeded_and_not_pool_words():
    from rrp.envs.computerworld import NAMES, WORDS, goal_for, proc_string
    a = [proc_string(random.Random(i)) for i in range(2000)]
    assert a == [proc_string(random.Random(i)) for i in range(2000)]
    assert len(set(a)) > 1950 and not set(a) & (set(WORDS) | {n.lower() for n in NAMES})
    assert {len(x) for x in a} == {4, 6, 8}
    assert goal_for("cw/open_type", 7) == goal_for("cw/open_type", 7, "pool") and \
        goal_for("cw/open_type", 7, "procedural")["text"] not in WORDS
    f = goal_for("cw/fill_form", 7, "procedural")
    assert f["email"] == f["name"].lower() + "@example.org" and f["name"][0].isupper()
    with pytest.raises(ValueError, match="one of"):
        goal_for("cw/open_type", 7, "bogus")


@pytest.mark.computerworld
def test_goal_for_equals_the_envs_reset_goal_and_procedural_instruction_fits_the_context():
    from rrp.envs.base import make_env
    from rrp.envs.computerworld import goal_for
    for strings in ("pool", "procedural"):
        for task in ("cw/calc_sum", "cw/open_type", "cw/fill_form"):
            env = make_env("computerworld", task=task, body="cw_pointer", seed=0, strings=strings)
            for seed in (3, 11):
                env.reset(seed)
                assert env.goal == goal_for(task, seed, strings)
                assert len(env.instruction) <= LI                           # the longest (4-syllable fill_form) fits
            assert env.spec.provenance["strings"] == strings
            assert env.spec.provenance["adapter"] == ("cw_env.v2" if strings == "pool" else "cw_env.v3")
            env.close()
    longest = "Fill in the sign-up form with name '%s' and email '%s@example.org', then submit it." % ("Zzzzzzzz", "zzzzzzzz")
    assert len(longest) <= LI
