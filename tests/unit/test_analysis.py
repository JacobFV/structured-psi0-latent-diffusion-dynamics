from rrp.evaluation.statistics import success_counts, summarize_attempts, wilson


def test_analysis_recomputes_counts_from_episode_rows():
    rows = [{"success": True}, {"success": False}, {"success": False}]
    assert success_counts(rows) == (1, 3)


def test_timeouts_and_refusals_stay_in_denominator():
    result = summarize_attempts(["success", "timeout", "refused", "failure"])
    assert result.attempted == 4
    assert result.successes == 1
    assert result.success_rate == 0.25


def test_wilson_bounds():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.35
    assert wilson(0, 0) == (None, None)
