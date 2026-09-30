"""Unit R20 (docs/relations.md section 10, row R20): UI factors. `catalog.py` section `ui` (`ui.label_for`,
`ui.contains`, `ui.focus_next`, `ui.above`, `ui.drag_to`), `envs.computerworld`'s public UI fields
(`ui_public_fields`: `parent_id`, `zlayer`, `focus_rank`) and `ui-rel-v1` edges (`ui_edges`), and
`harness/data/relgen/ui.py`'s privileged labels of the same five names.

No `computerworld` wheel needed anywhere in this file: every scene here is the same hand-built inline dict
`test_computerworld.py` / `test_state_view_envs.py` use (`scene_widgets`, `cw_state_view` are pure)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rrp.envs.computerworld import (ScreenFrame, SlotRegistry, UI_REL_VOCAB, cw_state_view, scene_widgets, ui_edges,
                                    ui_public_fields)
from rrp.harness.data.relgen import TokenIndex
from rrp.harness.data.relgen.transforms import cf_swap, reveal, surprise
from rrp.harness.data.relgen.ui import (above_fn, contains_fn, dragged_widget_id, drag_to_fn, focus_next_fn,
                                        label_for_fn, label_for_sample)
import rrp.policies.relations.catalog  # noqa: F401  (registers FACTORS / FIELDS / PRESETS on import)
from rrp.policies.relations.base import FACTORS, FIELDS, PRESETS, FactorSpec, PrivilegedInput, RelCtx, TokenSet, \
    EdgeSet, assert_deployable, get_factor, resolve
from rrp.policies.relations.ops import OPS, FactorSite

I = {"a": 1024, "b": 0, "c": 0, "d": 1024, "tx": 0, "ty": 0}


def _node(i, x, y, w, h, z, sem=None, inter=None, tx=0, ty=0, kind="box"):
    return {"id": i, "bounds": {"x": x, "y": y, "width": w, "height": h}, "primitive": {"kind": kind},
            "semantic": sem, "interaction": inter, "transform": dict(I, tx=tx, ty=ty), "z": z, "opacity": 255,
            "clip": None}


def _lbl(text):
    return {"role": "label", "label": text, "value": None, "disabled": False, "focusable": False}


def _tb(text, disabled=False):
    return {"role": "textbox", "label": text, "value": "", "disabled": disabled, "focusable": True}


# window 0: a "Name" label immediately followed by its (focused) textbox, both z-layer 100; window 1: an "Email"
# label + a DISABLED textbox (never a focus_next candidate), z-layer 200 (renders above window 0).
SCENE = {"focus": {"interaction": "window:0:content:name"}, "nodes": [
    _node(1, 20, 30, 40, 20, 100, sem=_lbl("Name"), inter="window:0:content:name_label", tx=100, ty=100),
    _node(2, 20, 60, 40, 20, 100, sem=_tb("Name"), inter="window:0:content:name", tx=100, ty=100),
    _node(3, 20, 90, 40, 20, 200, sem=_lbl("Email"), inter="window:1:content:email_label", tx=200, ty=200),
    _node(4, 20, 120, 40, 20, 200, sem=_tb("Email", disabled=True), inter="window:1:content:email", tx=200, ty=200),
]}
NAME_LABEL, NAME_BOX, EMAIL_LABEL, EMAIL_BOX = 0, 1, 2, 3


def _table():
    return SlotRegistry().assign(scene_widgets(SCENE))


def _state_view():
    frame, reg = ScreenFrame(640, 480), SlotRegistry()
    return cw_state_view(SCENE, frame, reg, t=0.0), reg


# ================================================================== envs.computerworld: public fields
def test_ui_public_fields_on_the_fixture_scene():
    f = ui_public_fields(_table())
    np.testing.assert_array_equal(f["parent_id"], [0, 0, 1, 1])
    np.testing.assert_array_equal(f["zlayer"], [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(f["focus_rank"], [-1, 0, -1, -1])       # only the Name textbox is focused


def test_ui_public_fields_null_slot_gets_neutral_defaults():
    reg = SlotRegistry()
    table = reg.assign(scene_widgets(SCENE))
    smaller = {"focus": {}, "nodes": [SCENE["nodes"][0]]}                 # every other widget vanishes
    table2 = reg.assign(scene_widgets(smaller))
    f = ui_public_fields(table2)
    assert table2[NAME_BOX] is None
    assert f["parent_id"][NAME_BOX] == -1 and f["zlayer"][NAME_BOX] == 0.0 and f["focus_rank"][NAME_BOX] == -1


# ================================================================== envs.computerworld: ui-rel-v1 edges
def test_ui_edges_vocab_order():
    assert UI_REL_VOCAB == ("label_for", "contains", "focus_next", "above")


def test_ui_edges_label_for_points_from_label_to_its_control():
    e = ui_edges(_table())
    i = UI_REL_VOCAB.index("label_for")
    assert e[NAME_LABEL, NAME_BOX, i] and e[EMAIL_LABEL, EMAIL_BOX, i]
    assert not e[NAME_BOX, NAME_LABEL, i]                                 # directed: control -> label never holds
    assert e[..., i].sum() == 2                                           # no spurious cross-window pairs


def test_ui_edges_contains_is_same_window_reflexive_symmetric():
    e = ui_edges(_table())
    i = UI_REL_VOCAB.index("contains")
    assert e[NAME_LABEL, NAME_BOX, i] and e[NAME_BOX, NAME_LABEL, i] and e[NAME_LABEL, NAME_LABEL, i]
    assert not e[NAME_LABEL, EMAIL_LABEL, i] and not e[NAME_BOX, EMAIL_BOX, i]


def test_ui_edges_focus_next_only_links_focusable_enabled_boxed_widgets_cyclically():
    e = ui_edges(_table())
    i = UI_REL_VOCAB.index("focus_next")
    # labels are never focusable; the Email textbox is disabled -> the Name textbox is the ONLY tab stop, so it
    # cycles to itself
    assert e[NAME_BOX, NAME_BOX, i]
    assert e[..., i].sum() == 1


def test_ui_edges_above_is_the_higher_zlayer_widget():
    e = ui_edges(_table())
    i = UI_REL_VOCAB.index("above")
    assert e[NAME_LABEL, EMAIL_LABEL, i] and e[NAME_BOX, EMAIL_BOX, i]
    assert not e[EMAIL_LABEL, NAME_LABEL, i] and not e[NAME_LABEL, NAME_BOX, i]    # same layer: never above itself


def test_ui_edges_null_slot_row_and_column_are_all_false():
    reg = SlotRegistry()
    table = reg.assign(scene_widgets(SCENE))
    smaller = {"focus": {}, "nodes": [SCENE["nodes"][0]]}
    table2 = reg.assign(scene_widgets(smaller))
    e = ui_edges(table2)
    assert not e[NAME_BOX].any() and not e[:, NAME_BOX].any()


# ================================================================== harness/data/relgen/ui.py: labels match `ui_edges` (StateView side, R14-style cross-check)
def test_ui_relgen_labels_match_ui_edges_on_the_fixture():
    sv, reg = _state_view()
    table = reg.assign(scene_widgets(SCENE))
    e = ui_edges(table)
    ids = [sv.token_entity("widgets", i) for i in range(len(table))]
    idx = TokenIndex(sets={"ctx": ids})
    for name, fn in (("label_for", label_for_fn), ("contains", contains_fn), ("focus_next", focus_next_fn),
                     ("above", above_fn)):
        lab = fn(sv, idx)
        col = UI_REL_VOCAB.index(name)
        np.testing.assert_array_equal(lab.value[..., 0] > 0, e[..., col])
        assert lab.valid.all()                                           # every slot here is a real widget


def test_dragged_widget_id_is_the_currently_focused_widget():
    sv, _ = _state_view()
    ids = [n["id"] for n in sv.ui_tree() if n["role"] == "textbox" and n["label"] == "Name"]
    assert dragged_widget_id(sv) == ids[0]


def test_drag_to_fn_targets_the_nearest_other_widget_to_the_dragged_one():
    sv, reg = _state_view()
    table = reg.assign(scene_widgets(SCENE))
    ids = [sv.token_entity("widgets", i) for i in range(len(table))]
    idx = TokenIndex(sets={"ctx": ids})
    lab = drag_to_fn(sv, idx)
    assert lab.value[NAME_BOX, NAME_LABEL, 0] == 1.0                     # the Name label sits right above its box
    assert lab.value[NAME_BOX].sum() == 1.0                              # exactly one drop-target candidate wins
    assert lab.value[NAME_LABEL].sum() == 0.0                            # only the dragged (focused) row is ever true


def test_drag_to_fn_no_focused_widget_is_all_zero():
    scene = dict(SCENE, focus={})
    reg = SlotRegistry()
    sv = cw_state_view(scene, ScreenFrame(640, 480), reg, t=0.0)
    table = reg.assign(scene_widgets(scene))
    ids = [sv.token_entity("widgets", i) for i in range(len(table))]
    lab = drag_to_fn(sv, TokenIndex(sets={"ctx": ids}))
    assert lab.value.sum() == 0.0


def test_labels_registered():
    from rrp.harness.data.relgen import LABELS
    assert {"label_for", "contains", "focus_next", "above", "drag_to"} <= set(LABELS)
    assert LABELS["drag_to"].needs == frozenset({"ui_tree", "poses"})
    assert LABELS["contains"].needs == frozenset({"ui_tree"})


# ================================================================== catalog.py: ui.*
def test_catalog_ui_fields_registered():
    assert FIELDS["zlayer"].kind == "scalar" and FIELDS["zlayer"].prov == "public"
    assert FIELDS["focus_rank"].kind == "scalar" and FIELDS["focus_rank"].prov == "public"
    assert FIELDS["parent_id"].prov == "public"                          # foundation field, reused here


@pytest.mark.parametrize("name", ["ui.label_for", "ui.contains", "ui.focus_next", "ui.above"])
def test_catalog_ui_edge_factors_are_edge_bias_on_ui_rel_v1(name):
    d = get_factor(name)
    assert d.field == "edges:ui-rel-v1" and d.op == "edge" and d.form == "bias"
    assert d.sources[0] == "given" and "gt" in d.sources
    assert d.label == name.split(".", 1)[1]
    assert d.form in OPS[d.op].forms


def test_catalog_ui_contains_is_symmetric_others_are_directed():
    assert get_factor("ui.contains").algebra.direction == "symmetric"
    for name in ("ui.label_for", "ui.focus_next", "ui.above"):
        assert get_factor(name).algebra.direction == "directed"


def test_catalog_ui_label_for_carries_reveal_and_surprise_gen():
    assert get_factor("ui.label_for").gen == ("reveal", "surprise")
    for name in ("ui.contains", "ui.focus_next", "ui.above"):
        assert get_factor(name).gen == ()


def test_catalog_ui_drag_to_is_a_bilinear_pair_probe():
    d = get_factor("ui.drag_to")
    assert d.op == "bilinear" and d.form == "aug" and d.label == "drag_to"
    assert d.sources[0] == "probe" and "gt" in d.sources
    assert d.readout is not None and d.readout.address == "pair" and d.readout.label == "drag_to"
    assert d.form in OPS[d.op].forms


def test_ui_rel_v1_vocab_registered_in_catalog():
    from rrp.policies.relations.catalog import VOCABS
    assert VOCABS["ui-rel-v1"] == UI_REL_VOCAB


def test_catalog_ui_preset():
    specs = resolve(["preset:ui"])
    assert {s.name for s in specs} == {"ui.label_for", "ui.contains", "ui.focus_next", "ui.above", "ui.drag_to"}


def test_catalog_ui_factors_resolve_and_default_deploy_safe():
    assert_deployable(resolve(["preset:ui"]))                            # default sources: given / probe, both safe


def test_catalog_ui_gt_source_is_blocked_in_deploy_mode():
    for name in ("ui.contains", "ui.drag_to"):
        with pytest.raises(PrivilegedInput):
            assert_deployable(resolve([FactorSpec(name=name, source="gt")]))


# ================================================================== the pointer FactorSite: resolves + runs one tick on the fixture
def _ui_relctx(table):
    n = len(table)
    fields = ui_public_fields(table)
    edges = ui_edges(table)
    mask = torch.tensor([w is not None for w in table])[None]
    ts = TokenSet("ctx", mask, fields={
        "zlayer": torch.as_tensor(fields["zlayer"])[None, :, None],
        "focus_rank": torch.as_tensor(fields["focus_rank"])[None, :, None],
        "parent_id": torch.as_tensor(fields["parent_id"])[None, :, None],
    })
    rc = RelCtx(sets={"ctx": ts}, edges={"ctx>ctx": EdgeSet(UI_REL_VOCAB, torch.from_numpy(edges)[None])})
    return rc, n


def test_ui_preset_resolves_on_the_pointer_body_ctx_ctx_site():
    """The pointer body's own `ctx` token set (`policies.pointer.UICtx`) is exactly this shape: NW widget tokens
    self-attending at one `ctx>ctx` site (docs/relations.md section 2's pointer/CW row). This unit does not own
    `policies/pointer.py` (R6's `RelBlock`s there still carry the empty preset `none`, by design -- catalog.py's
    own R6 comment), so the wiring is exercised here, against the SAME `FactorSite` class the real net's
    `RelBlock`s use, on data built by this unit's own `envs.computerworld` functions."""
    rc, n = _ui_relctx(_table())
    specs = resolve(["preset:ui"])
    site = FactorSite(heads=2, dim=8, site="ctx>ctx", specs=specs, carries=("edges:ui-rel-v1", "hidden"))
    assert {s.name for s in site.specs} == {"ui.label_for", "ui.contains", "ui.focus_next", "ui.above", "ui.drag_to"}


def test_pointer_policy_with_preset_ui_runs_one_rollout_tick_on_the_fixture():
    """One tick, wheel-free: the CW fixture scene stands in for one `ComputerWorldEnv.observe()` (`scene_widgets` /
    `SlotRegistry` are the exact pure functions the real env's `observe()` / `state_view()` both build on), and one
    `FactorSite.bias` / `.augment` forward stands in for one attention layer of a rollout tick's forward pass."""
    torch.manual_seed(0)
    rc, n = _ui_relctx(_table())
    specs = resolve(["preset:ui"])
    site = FactorSite(heads=2, dim=8, site="ctx>ctx", specs=specs, carries=("edges:ui-rel-v1", "hidden"))
    for p in site.parameters():                                          # off zero-init: every factor is a
        torch.nn.init.normal_(p, std=0.1)                                # genuine no-op at step 0 (docs 3.2)

    b = site.bias(rc)
    assert b is not None and b.shape == (1, 2, n, n) and torch.isfinite(b).all()

    x = torch.randn(1, n, 8)
    qa, ka = site.augment(rc, x, x)
    assert qa.shape == (1, 2, n, 8) and ka.shape == (1, 2, n, 8)
    assert torch.isfinite(qa).all() and torch.isfinite(ka).all()

    logits = b + qa @ ka.transpose(-1, -2)                                # a real attention-logit combination
    weights = logits.softmax(-1)
    assert torch.allclose(weights.sum(-1), torch.ones(1, 2, n), atol=1e-5)


@pytest.mark.computerworld
def test_pointer_body_rollout_tick_with_the_real_computerworld_wheel():
    """The same forward, against a REAL `ComputerWorldEnv(body="cw_pointer")` tick's scene (skips without the
    optional `computerworld` extra, `tests/conftest.py`; the merge gate's own `pytest tests/unit` command does not
    put the wheel on `PYTHONPATH`, so this is a bonus check, not the acceptance path -- see
    `test_pointer_policy_with_preset_ui_runs_one_rollout_tick_on_the_fixture` above for the wheel-free version)."""
    from rrp.envs.base import make_env
    env = make_env("computerworld", task="cw/fill_form", body="cw_pointer", seed=0)
    env.reset(seed=0)
    env.step({})
    table = SlotRegistry().assign(scene_widgets(env.scene()))
    rc, n = _ui_relctx(table)
    specs = resolve(["preset:ui"])
    site = FactorSite(heads=2, dim=8, site="ctx>ctx", specs=specs, carries=("edges:ui-rel-v1", "hidden"))
    b = site.bias(rc)
    assert b is not None and b.shape == (1, 2, n, n)
    env.close()


# ================================================================== R9 transforms on UI data (docs 5.4's own worked example: "UI label changes")
def test_cf_swap_on_a_ui_label_field_swaps_widget_identity_and_its_pairwise_labels_consistently():
    sv, reg = _state_view()
    table = reg.assign(scene_widgets(SCENE))
    ids = [sv.token_entity("widgets", i) for i in range(len(table))]
    lab = contains_fn(sv, TokenIndex(sets={"ctx": ids}))
    char_codes = np.array([[1, 2, 0], [3, 4, 0], [5, 6, 0], [7, 8, 0]])   # a stand-in widget "label" text field
    sample = {"inputs": {"tokens": {"ctx": {"fields": {"label": char_codes}}}},
             "labels": {"contains": {"set": "ctx", "arity": 2, "value": lab.value, "valid": lab.valid}}}
    rng = np.random.default_rng(0)
    [out] = cf_swap(sample, rng, {"field": "label", "pair": (NAME_LABEL, EMAIL_LABEL)})

    swapped = out["inputs"]["tokens"]["ctx"]["fields"]["label"]
    np.testing.assert_array_equal(swapped[NAME_LABEL], char_codes[EMAIL_LABEL])
    np.testing.assert_array_equal(swapped[EMAIL_LABEL], char_codes[NAME_LABEL])
    np.testing.assert_array_equal(swapped[NAME_BOX], char_codes[NAME_BOX])         # untouched slot: unchanged

    before, after = lab.value[..., 0], out["labels"]["contains"]["value"][..., 0]
    # the swap is applied to BOTH axes of the pairwise "contains" label (docs 5.2: everything that references the
    # swapped slots moves together) -- Name-window membership now shows up at the Email-label row/column and vice
    # versa
    assert after[EMAIL_LABEL, NAME_BOX] == before[NAME_LABEL, NAME_BOX] == 1.0
    assert after[NAME_LABEL, NAME_BOX] == before[EMAIL_LABEL, NAME_BOX] == 0.0
    assert out["provenance"]["transforms"][-1]["transform"] == "cf_swap"


def test_label_for_sample_reveal_collapses_onto_the_true_binding():
    sv, _ = _state_view()
    control = next(n["id"] for n in sv.ui_tree() if n["role"] == "textbox" and n["label"] == "Name")
    name_label = next(n["id"] for n in sv.ui_tree() if n["role"] == "label" and n["label"] == "Name")
    email_label = next(n["id"] for n in sv.ui_tree() if n["role"] == "label" and n["label"] == "Email")
    sample = label_for_sample(control, [name_label, email_label], sv)
    assert sample["inputs"]["evidence"] == [{"t": 0, "excludes": [email_label]}]

    [out] = reveal(sample, np.random.default_rng(0), {})
    q = out["labels"]["reveal"]["value"]
    assert q.shape == (1, 2)
    np.testing.assert_allclose(q[0], [1.0, 0.0])                         # fully collapsed onto the true label


def test_label_for_sample_surprise_flips_after_the_bound_label_changes():
    """docs 5.4's own worked example: "UI label changes" -- the collapsed candidate is later contradicted (the
    control's true label-for association flips), and `surprise` records the switch."""
    sv, _ = _state_view()
    control = next(n["id"] for n in sv.ui_tree() if n["role"] == "textbox" and n["label"] == "Name")
    name_label = next(n["id"] for n in sv.ui_tree() if n["role"] == "label" and n["label"] == "Name")
    email_label = next(n["id"] for n in sv.ui_tree() if n["role"] == "label" and n["label"] == "Email")
    sample = label_for_sample(control, [name_label, email_label], sv,
                              evidence=[{"t": 0, "excludes": [email_label]}])
    sample["inputs"]["evidence"] = sample["inputs"]["evidence"] + [{"t": 1, "excludes": []}]

    [out] = surprise(sample, np.random.default_rng(0), {"rate": 1.0, "after": 1, "collapse": 0.5,
                                                         "schedule": [0, 1]})
    lab = out["labels"]["surprise"]
    np.testing.assert_allclose(lab["value"][0], [1.0, 0.0])              # t=0: collapsed onto the true label
    assert lab["switch_at"] == 1                                          # t=1: the grace period elapses, flips
    np.testing.assert_allclose(lab["value"][1], [0.0, 1.0])              # reopened onto the OTHER candidate
