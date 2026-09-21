import pytest
from rrp.tasks.receipts import ReceiptStore, Receipt
from rrp.contracts.errors import ProvenanceError


def test_same_type_foreign_frame_is_not_accepted():
    store = ReceiptStore()
    store.put(Receipt(event_id="other", attempt=0, output_name="hole_frame",
                      type="frame_estimate", version=1, value={"x": 0.0}))
    with pytest.raises(ProvenanceError) as e:
        store.require(event_id="locate", attempt=0, output_name="hole_frame",
                      type="frame_estimate", version=1)
    assert e.value.code == "foreign_same_type"


def test_wrong_type_stale_and_old_valid():
    s = ReceiptStore()
    s.put(Receipt("locate", 0, "hole_frame", "frame_estimate", 0, {"x": 0.0}))
    with pytest.raises(ProvenanceError) as e:
        s.require(event_id="locate", attempt=0, output_name="hole_frame", type="contact_anchor")
    assert e.value.code == "wrong_type"
    # old-but-valid: explicitly requested older version, still valid -> accepted
    s.put(Receipt("locate", 0, "hole_frame", "frame_estimate", 1, {"x": 0.1}))
    assert s.require(event_id="locate", attempt=0, output_name="hole_frame", type="frame_estimate",
                     version=0).value == {"x": 0.0}
    assert s.require(event_id="locate", attempt=0, output_name="hole_frame", type="frame_estimate").value == {"x": 0.1}
    # stale: invalidated same-event older result
    s.invalidate("locate", 0, "hole_frame", "superseded")
    with pytest.raises(ProvenanceError) as e:
        s.require(event_id="locate", attempt=0, output_name="hole_frame", type="frame_estimate", version=0)
    assert e.value.code == "stale"
    # a different attempt of the same event is a different provenance
    with pytest.raises(ProvenanceError) as e:
        s.require(event_id="locate", attempt=1, output_name="hole_frame", type="frame_estimate")
    assert e.value.code in ("foreign_same_type", "missing")


def test_versions_monotonic():
    s = ReceiptStore()
    s.put(Receipt("a", 0, "o", "frame_estimate", 3, {}))
    with pytest.raises(ProvenanceError):
        s.put(Receipt("a", 0, "o", "frame_estimate", 3, {}))
