from rrp.evaluation.adaptation import nested_budget_indices, first_sustained_crossing


def test_adaptation_budgets_are_nested():
    selected = nested_budget_indices(n=120, budgets=[5, 20, 100], seed=3)
    assert set(selected[5]) <= set(selected[20]) <= set(selected[100])
    assert len(selected[100]) == 100


def test_unreached_success_is_censored():
    r = first_sustained_crossing(budgets=[0, 10, 50], success=[0.1, 0.3, 0.7], threshold=0.8)
    assert r.censored and r.budget is None
    r = first_sustained_crossing(budgets=[0, 10, 50], success=[0.1, 0.9, 0.7], threshold=0.8)
    assert r.censored
    r = first_sustained_crossing(budgets=[0, 10, 50], success=[0.1, 0.3, 0.9], threshold=0.8)
    assert r.budget == 50 and r.provisional
