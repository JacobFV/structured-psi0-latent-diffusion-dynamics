"""Radar builder (rrp.viz.export.radar): selectors, floor-to-reference normalization, gaps and spreads on tiny docs."""
from rrp.viz.export.radar import build_radar_from, resolve_value

DOCS = {
    "results": {"rows": [
        {"source_file": "a/cmp.json", "metric": "succ", "key_path": ["semfix, seed 1", "b1", "succ"], "k": 6, "n": 10, "decision": "D-1"},
        {"source_file": "a/cmp.json", "metric": "succ", "key_path": ["semfix, seed 2", "b1", "succ"], "k": 8, "n": 10, "decision": "D-1"},
        {"source_file": "b/cmp.json", "metric": "succ", "key_path": ["semfix, seed 2", "b1", "succ"], "k": 8, "n": 10},  # other file
        {"source_file": "a/cmp.json", "metric": "succ", "key_path": ["BC", "b1", "succ"], "k": 9, "n": 10},
        {"source_file": "a/cmp.json", "metric": "succ", "key_path": ["BC", "b1", "succ"], "k": 9, "n": 10},  # duplicate copy
    ]},
    "edits": {"rows": [{"source_file": "e.json", "edit": "halt", "variant": "semfix", "seed": None, "effect": -0.3, "ci": [-0.4, -0.2]}]},
    "robustness": {"reports": [{"robot": "r", "route": "bc", "nominal": {"motion": {"step": 3.0}}},
                               {"robot": "r", "route": "semfix", "nominal": {"motion": {"step": 2.0}}}]},
}
SEL = lambda lin: {"doc": "results", "where": {"source_file": "^a/cmp.json$"}, "key_path": [lin, "^b1$", "^succ$"], "agg": "sum_kn", "group": 0}


def test_sum_kn_groups_spread_and_dedupe():
    v = resolve_value(SEL("^semfix"), DOCS)
    assert (v["k"], v["n"]) == (14, 20) and v["spread"]["x"] == [0.6, 0.8]
    bc = resolve_value(SEL("^BC$"), DOCS)
    assert (bc["k"], bc["n"]) == (9, 10)  # the duplicate copy counts once
    assert resolve_value(SEL("^nosem"), DOCS) == {"missing": "no matching row in the curated evidence"}
    assert resolve_value({"missing": "why"}, DOCS) == {"missing": "why"}


def test_normalization_gaps_and_min_direction():
    config = {"normalization": {"clamp": [0, 1.5]}, "series": [{"id": "semfix"}, {"id": "bc"}, {"id": "nosem"}],
              "axes": [
                  {"id": "a", "label": "a", "metric": "m", "direction": "max", "protocol": "p", "decision": "D-1",
                   "floor": {"value": 0.0}, "reference": {"series": "bc"},
                   "series": {"semfix": SEL("^semfix"), "bc": SEL("^BC$"), "nosem": {"missing": "not run"}}},
                  {"id": "h", "label": "h", "metric": "m", "direction": "max", "protocol": "p", "decision": "D-2",
                   "floor": {"value": 0.0}, "reference": {"value": 0.25},
                   "series": {"semfix": {"doc": "edits", "where": {"edit": "^halt$", "seed": "^$"}, "agg": "effect", "negate": True}}},
                  {"id": "s", "label": "s", "metric": "m", "direction": "min", "protocol": "p", "decision": "D-3",
                   "floor": {"value": 6.0}, "reference": {"series": "bc"},
                   "series": {r: {"doc": "robustness", "list": "reports", "where": {"route": f"^{r}$"}, "agg": "path", "value": ["nominal", "motion", "step"]} for r in ("semfix", "bc")}},
              ]}
    out = build_radar_from(config, DOCS)
    a, h, s = out["axes"]
    assert a["series"]["bc"]["r"] == 1.0 and abs(a["series"]["semfix"]["r"] - 0.7 / 0.9) < 1e-3
    assert "drawn" not in a["series"]["nosem"] and a["series"]["nosem"]["missing"] == "not run"
    assert h["series"]["semfix"]["value"] == 0.3 and h["series"]["semfix"]["r"] == 1.2          # negated effect / 0.25
    assert h["series"]["semfix"]["spread"]["x"] == [0.2, 0.4]
    assert s["series"]["bc"]["r"] == 1.0 and abs(s["series"]["semfix"]["r"] - 4 / 3) < 1e-3       # lower is better
    assert out["n_values"] == 5


def test_file_selector_and_reference_multiple_floor(tmp_path):
    import json as _j
    (tmp_path / "lat.json").write_text(_j.dumps({"a": {"p95": 28.7}, "b": {"p95": 28.3}}))
    config = {"normalization": {"clamp": [0, 1.5]}, "series": [{"id": "bc"}, {"id": "lat"}],
              "axes": [{"id": "l", "label": "l", "metric": "ms", "direction": "min", "protocol": "p", "decision": "D-058",
                        "floor": {"reference_multiple": 1.25}, "reference": {"series": "bc"},
                        "series": {"bc": {"file": "lat.json", "value": ["b", "p95"]}, "lat": {"file": "lat.json", "value": ["a", "p95"]}}}]}
    ax = build_radar_from(config, {"_repo": str(tmp_path)})["axes"][0]
    assert abs(ax["floor"]["value"] - 35.375) < 1e-6 and ax["series"]["bc"]["r"] == 1.0
    assert abs(ax["series"]["lat"]["r"] - (35.375 - 28.7) / (35.375 - 28.3)) < 1e-3
