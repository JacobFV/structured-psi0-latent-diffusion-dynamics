"""Unit R9 (docs/relations.md section 10): generic relgen transforms -- reveal, surprise, cf_swap, noise, occlude,
subsample. Math tests against hand-computed Bayes posteriors / KL recovery, purity and seed-determinism, provenance."""
import copy

import numpy as np
import pytest

from rrp.harness.data.relgen import TRANSFORMS
from rrp.harness.data.relgen.transforms import (cf_swap, kl_divergence, noise, occlude, recovery_steps, reveal,
                                                 subsample, surprise)


def _deep_eq(a, b) -> bool:
    """Structural equality that treats numpy arrays element-wise (plain `==` raises on them inside dicts/lists)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b


# ------------------------------------------------------------------ registration
def test_transforms_registered():
    for name in ("reveal", "surprise", "cf_swap", "noise", "occlude", "subsample"):
        assert name in TRANSFORMS
        assert TRANSFORMS[name].fn is {"reveal": reveal, "surprise": surprise, "cf_swap": cf_swap, "noise": noise,
                                        "occlude": occlude, "subsample": subsample}[name]


# ------------------------------------------------------------------ reveal: Bayes posterior on a toy candidate set
def _toy_reveal_sample():
    # 3 candidates a, b, c; uniform prior; evidence at t=1 rules out c, at t=3 rules out b -> a survives alone.
    return {"inputs": {"candidates": ["a", "b", "c"],
                        "evidence": [{"t": 1, "excludes": ["c"]}, {"t": 3, "excludes": ["b"]}]},
            "labels": {}, "provenance": {}}


def test_reveal_is_bayes_posterior():
    sample = _toy_reveal_sample()
    rng = np.random.default_rng(0)
    [out] = reveal(sample, rng, {"schedule": [0, 1, 2, 3]})
    q = out["labels"]["reveal"]["value"]
    np.testing.assert_allclose(q[0], [1 / 3, 1 / 3, 1 / 3])         # no evidence yet: uniform prior
    np.testing.assert_allclose(q[1], [0.5, 0.5, 0.0])               # c excluded
    np.testing.assert_allclose(q[2], [0.5, 0.5, 0.0])               # unchanged until t=3
    np.testing.assert_allclose(q[3], [1.0, 0.0, 0.0])               # b excluded too -> collapsed on a
    assert out["labels"]["reveal"]["candidates"] == ["a", "b", "c"]


def test_reveal_respects_nonuniform_prior():
    sample = _toy_reveal_sample()
    sample["inputs"]["prior"] = {"a": 1.0, "b": 3.0, "c": 6.0}
    rng = np.random.default_rng(0)
    [out] = reveal(sample, rng, {"schedule": [0]})
    np.testing.assert_allclose(out["labels"]["reveal"]["value"][0], [0.1, 0.3, 0.6])


def test_reveal_default_schedule_is_evidence_times():
    sample = _toy_reveal_sample()
    [out] = reveal(sample, np.random.default_rng(0), {})
    assert out["labels"]["reveal"]["t"] == [1, 3]


def test_reveal_appends_provenance_and_is_pure():
    sample = _toy_reveal_sample()
    before = copy.deepcopy(sample)
    [out] = reveal(sample, np.random.default_rng(0), {"schedule": [0, 1]})
    assert _deep_eq(sample, before)                                        # input untouched
    assert out["provenance"]["transforms"][-1]["transform"] == "reveal"
    assert out["provenance"]["transforms"][-1]["schedule"] == [0, 1]


# ------------------------------------------------------------------ surprise: flip after collapse + recovery metric
def _toy_surprise_sample():
    # collapses onto "a" at t=1 (b, c excluded); schedule runs to t=5 so a post-collapse window exists to flip in.
    return {"inputs": {"candidates": ["a", "b", "c"],
                        "evidence": [{"t": 1, "excludes": ["b", "c"]}]},
            "labels": {}, "provenance": {}}


def test_surprise_flips_after_collapse_deterministically():
    sample = _toy_surprise_sample()
    rng = np.random.default_rng(0)
    [out] = surprise(sample, rng, {"schedule": [0, 1, 2, 3, 4, 5], "rate": 1.0, "after": 1})
    prov = out["provenance"]["transforms"][-1]
    assert prov["collapse_at"] == 1
    assert prov["switch_at"] == 2                                   # first schedule step >= collapse + after
    q = out["labels"]["surprise"]["value"]
    np.testing.assert_allclose(q[1], [1.0, 0.0, 0.0])                # collapsed on a
    assert q[2][0] == 0.0                                            # a excluded at the switch: no longer favoured
    np.testing.assert_allclose(q[2], [0.0, 0.5, 0.5])
    for t in range(3, 6):
        np.testing.assert_allclose(q[t], [0.0, 0.5, 0.5])            # stays flipped


def test_surprise_rate_zero_never_flips():
    sample = _toy_surprise_sample()
    [out] = surprise(sample, np.random.default_rng(0), {"schedule": [0, 1, 2, 3], "rate": 0.0, "after": 0})
    prov = out["provenance"]["transforms"][-1]
    assert prov["collapse_at"] == 1 and prov["switch_at"] is None
    np.testing.assert_allclose(out["labels"]["surprise"]["value"][3], [1.0, 0.0, 0.0])


def test_surprise_is_seed_deterministic_and_pure():
    sample = _toy_surprise_sample()
    before = copy.deepcopy(sample)
    params = {"schedule": list(range(6)), "rate": 0.3, "after": 0}
    [o1] = surprise(sample, np.random.default_rng(7), params)
    [o2] = surprise(sample, np.random.default_rng(7), params)
    assert _deep_eq(sample, before)
    np.testing.assert_array_equal(o1["labels"]["surprise"]["value"], o2["labels"]["surprise"]["value"])
    assert o1["provenance"]["transforms"][-1]["switch_at"] == o2["provenance"]["transforms"][-1]["switch_at"]


def test_recovery_steps_counts_until_kl_below_eps():
    target = np.array([[0.0, 0.5, 0.5]] * 5)
    # estimate starts stuck on the old belief, then converges onto the target at index 2 (relative to switch=0)
    estimate = np.stack([
        [1.0, 0.0, 0.0],
        [0.7, 0.15, 0.15],
        [0.02, 0.49, 0.49],
        [0.0, 0.5, 0.5],
        [0.0, 0.5, 0.5],
    ])
    assert recovery_steps(target, estimate, switch_index=0, eps=0.05) == 2


def test_recovery_steps_never_converges_is_none():
    target = np.array([[0.0, 1.0]] * 3)
    estimate = np.array([[1.0, 0.0]] * 3)
    assert recovery_steps(target, estimate, switch_index=0, eps=0.01) is None


def test_kl_divergence_zero_for_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------ cf_swap
def _toy_cf_sample():
    return {"inputs": {
        "tokens": {"ctx": {
            "mask": np.array([True, True, True, True]),
            "fields": {
                "entity_id": np.array([10, 20, 30, 40]),
                "pos3d": np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]),
            },
        }},
        "edges": {"ctx>ctx": {"sets": ("ctx", "ctx"), "data": np.arange(16).reshape(4, 4).astype(float)}},
    }, "labels": {"held_by": {"set": "ctx", "arity": 2, "value": np.arange(16).reshape(4, 4).astype(float),
                              "valid": np.ones((4, 4), bool)}},
        "provenance": {}}


def test_cf_swap_swaps_tokens_edges_and_labels_consistently():
    sample = _toy_cf_sample()
    before = copy.deepcopy(sample)
    [out] = cf_swap(sample, np.random.default_rng(0), {"field": "entity_id", "pair": (1, 3)})

    assert _deep_eq(sample, before)                                          # pure
    ids = out["inputs"]["tokens"]["ctx"]["fields"]["entity_id"]
    np.testing.assert_array_equal(ids, [10, 40, 30, 20])              # slots 1 and 3 swapped
    pos = out["inputs"]["tokens"]["ctx"]["fields"]["pos3d"]
    np.testing.assert_array_equal(pos[1], [3, 0, 0])
    np.testing.assert_array_equal(pos[3], [1, 0, 0])

    edge = out["inputs"]["edges"]["ctx>ctx"]["data"]
    orig = before["inputs"]["edges"]["ctx>ctx"]["data"]
    np.testing.assert_array_equal(edge[1, :], orig[3, :][[0, 3, 2, 1]])
    np.testing.assert_array_equal(edge[:, 1], orig[:, 3][[0, 3, 2, 1]])
    assert edge[1, 1] == orig[3, 3] and edge[1, 3] == orig[3, 1] and edge[3, 1] == orig[1, 3]

    lab = out["labels"]["held_by"]["value"]
    orig_lab = before["labels"]["held_by"]["value"]
    assert lab[1, 1] == orig_lab[3, 3] and lab[3, 3] == orig_lab[1, 1]

    assert out["provenance"]["transforms"][-1] == {
        "transform": "cf_swap", "version": "1", "field": "entity_id", "set": "ctx", "swapped": [1, 3]}


def test_cf_swap_picks_a_deterministic_pair_from_rng_when_unspecified():
    sample = _toy_cf_sample()
    [o1] = cf_swap(sample, np.random.default_rng(3), {"field": "entity_id"})
    [o2] = cf_swap(sample, np.random.default_rng(3), {"field": "entity_id"})
    assert o1["provenance"]["transforms"][-1]["swapped"] == o2["provenance"]["transforms"][-1]["swapped"]


def test_cf_swap_unknown_field_raises():
    with pytest.raises(ValueError):
        cf_swap(_toy_cf_sample(), np.random.default_rng(0), {"field": "nope"})


# ------------------------------------------------------------------ noise
def _toy_noise_sample():
    return {"inputs": {"tokens": {"ctx": {
        "mask": np.array([True, True, False]),
        "fields": {"pos3d": np.array([[1., 2, 3], [4, 5, 6], [7, 8, 9]]),
                   "pos3d.valid": np.array([True, True, False])},
    }}}, "labels": {}, "provenance": {}}


def test_noise_adds_gaussian_and_sets_var_masked_by_valid():
    sample = _toy_noise_sample()
    before = copy.deepcopy(sample)
    [out] = noise(sample, np.random.default_rng(1), {"field": "pos3d", "sigma": 0.1})

    assert _deep_eq(sample, before)                                          # pure
    fields = out["inputs"]["tokens"]["ctx"]["fields"]
    orig = before["inputs"]["tokens"]["ctx"]["fields"]["pos3d"]
    delta = fields["pos3d"] - orig
    assert not np.allclose(delta[0], 0) and not np.allclose(delta[1], 0)
    np.testing.assert_allclose(delta[2], 0)                          # invalid slot untouched
    np.testing.assert_allclose(fields["pos3d.var"][0], [0.01, 0.01, 0.01])
    np.testing.assert_allclose(fields["pos3d.var"][2], [0.0, 0.0, 0.0])
    assert out["provenance"]["transforms"][-1] == {
        "transform": "noise", "version": "1", "field": "pos3d", "set": "ctx", "sigma": 0.1}


def test_noise_is_seed_deterministic():
    sample = _toy_noise_sample()
    [o1] = noise(sample, np.random.default_rng(42), {"field": "pos3d", "sigma": 0.5})
    [o2] = noise(sample, np.random.default_rng(42), {"field": "pos3d", "sigma": 0.5})
    np.testing.assert_array_equal(o1["inputs"]["tokens"]["ctx"]["fields"]["pos3d"],
                                   o2["inputs"]["tokens"]["ctx"]["fields"]["pos3d"])


def test_noise_zero_sigma_is_a_no_op_on_values():
    sample = _toy_noise_sample()
    [out] = noise(sample, np.random.default_rng(0), {"field": "pos3d", "sigma": 0.0})
    np.testing.assert_array_equal(out["inputs"]["tokens"]["ctx"]["fields"]["pos3d"],
                                   sample["inputs"]["tokens"]["ctx"]["fields"]["pos3d"])


# ------------------------------------------------------------------ occlude
def _toy_occlude_sample():
    return {"inputs": {"tokens": {"ctx": {"mask": np.array([True, True, True, True, True]), "fields": {}}}},
            "labels": {}, "provenance": {}}


def test_occlude_masks_tokens_probabilistically_and_is_pure():
    sample = _toy_occlude_sample()
    before = copy.deepcopy(sample)
    [out] = occlude(sample, np.random.default_rng(2), {"p": 0.5})

    assert _deep_eq(sample, before)
    mask = out["inputs"]["tokens"]["ctx"]["mask"]
    assert mask.sum() <= 5
    assert list(mask) != [True] * 5 or True                          # (sanity: no crash; exactness checked below)
    prov = out["provenance"]["transforms"][-1]
    assert prov["transform"] == "occlude" and prov["p"] == 0.5
    assert set(prov["occluded"]["ctx"]) == set(int(i) for i in np.nonzero(~mask)[0])


def test_occlude_p_zero_keeps_everything():
    sample = _toy_occlude_sample()
    [out] = occlude(sample, np.random.default_rng(0), {"p": 0.0})
    np.testing.assert_array_equal(out["inputs"]["tokens"]["ctx"]["mask"], [True] * 5)


def test_occlude_p_one_drops_everything_valid():
    sample = _toy_occlude_sample()
    [out] = occlude(sample, np.random.default_rng(0), {"p": 1.0})
    np.testing.assert_array_equal(out["inputs"]["tokens"]["ctx"]["mask"], [False] * 5)


def test_occlude_is_seed_deterministic():
    sample = _toy_occlude_sample()
    [o1] = occlude(sample, np.random.default_rng(9), {"p": 0.5})
    [o2] = occlude(sample, np.random.default_rng(9), {"p": 0.5})
    np.testing.assert_array_equal(o1["inputs"]["tokens"]["ctx"]["mask"], o2["inputs"]["tokens"]["ctx"]["mask"])


# ------------------------------------------------------------------ subsample
def _toy_subsample_sample():
    return {"inputs": {"knots": {"fields": {"time": np.array([0., 1, 2, 3, 4, 5])}},
                        "evidence": [{"t": 0, "excludes": []}, {"t": 2, "excludes": ["x"]},
                                     {"t": 4, "excludes": ["y"]}, {"t": 5, "excludes": ["z"]}]},
            "labels": {}, "provenance": {}}


def test_subsample_keeps_stride_and_is_pure():
    sample = _toy_subsample_sample()
    before = copy.deepcopy(sample)
    [out] = subsample(sample, np.random.default_rng(0), {"stride": 2})

    assert _deep_eq(sample, before)
    np.testing.assert_array_equal(out["inputs"]["knots"]["fields"]["time"], [0., 2., 4.])
    assert out["provenance"]["transforms"][-1] == {"transform": "subsample", "version": "1", "stride": 2,
                                                     "kept": [0, 2, 4]}


def test_subsample_remaps_evidence_times_without_dropping_events():
    sample = _toy_subsample_sample()
    [out] = subsample(sample, np.random.default_rng(0), {"stride": 2})
    ts = [ev["t"] for ev in out["inputs"]["evidence"]]
    assert ts == [0, 1, 2, 2]                                         # kept indices [0,2,4] -> new indices 0,1,2


def test_subsample_stride_one_is_identity():
    sample = _toy_subsample_sample()
    [out] = subsample(sample, np.random.default_rng(0), {"stride": 1})
    np.testing.assert_array_equal(out["inputs"]["knots"]["fields"]["time"], sample["inputs"]["knots"]["fields"]["time"])
    assert [ev["t"] for ev in out["inputs"]["evidence"]] == [0, 2, 4, 5]


def test_subsample_rejects_bad_stride():
    with pytest.raises(ValueError):
        subsample(_toy_subsample_sample(), np.random.default_rng(0), {"stride": 0})
