"""Statistics computed from raw episode rows only."""
from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def success_counts(rows) -> tuple[int, int]:
    rows = list(rows)
    return sum(1 for r in rows if r["success"]), len(rows)


def summarize_attempts(outcomes):
    from dataclasses import dataclass

    @dataclass
    class S:
        attempted: int
        successes: int
        success_rate: float

    outs = list(outcomes)
    k = sum(1 for o in outs if o == "success")
    return S(len(outs), k, k / len(outs) if outs else float("nan"))


# ------------------------------------------------------------------------------------------------------------------
# W11 (stable core API): interval estimates and tests used by the comparison scripts, in one place. numpy only.
# Conventions: 95% intervals; bootstrap = percentile bootstrap with a fixed seed (reproducible); None when undefined.

def newcombe_diff(k1: int, n1: int, k2: int, n2: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    """CI for p1 - p2 of two INDEPENDENT proportions (Newcombe 1998 hybrid score, method 10; Wilson limits)."""
    if n1 == 0 or n2 == 0:
        return None, None
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    return (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def _finite(x):
    import numpy as np
    return np.asarray([v for v in x if v is not None and np.isfinite(v)], float)


def boot_ci(x, n: int = 4000, seed: int = 0) -> dict:
    """Mean of x with a percentile-bootstrap 95% CI (None/NaN entries dropped). Same as latent_causal.boot_ci."""
    import numpy as np
    x = _finite(x)
    if len(x) == 0:
        return dict(mean=None, lo=None, hi=None, n=0)
    g = np.random.default_rng(seed)
    bs = x[g.integers(0, len(x), (n, len(x)))].mean(1) if len(x) > 1 else np.array([x[0]])
    return dict(mean=float(x.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                n=int(len(x)))


def _paired(a, b):
    import numpy as np
    a, b = np.asarray(list(a), float), np.asarray(list(b), float)
    if a.shape != b.shape:
        raise ValueError(f"paired samples need equal length ({a.shape} vs {b.shape})")
    d = a - b
    return d[np.isfinite(d)]


def paired_bootstrap_ci(a, b, n: int = 4000, seed: int = 0) -> dict:
    """mean(a_i - b_i) over PAIRED units (same seed/scene; pairs with a missing side dropped) with a bootstrap CI."""
    return boot_ci(_paired(a, b), n=n, seed=seed)


def boot_diff(a, b, n: int = 4000, seed: int = 0) -> dict | None:
    """mean(a) - mean(b) for INDEPENDENT samples, percentile bootstrap (resampling each side)."""
    import numpy as np
    a, b = _finite(a), _finite(b)
    if len(a) == 0 or len(b) == 0:
        return None
    r = np.random.default_rng(seed)
    d = [r.choice(a, len(a)).mean() - r.choice(b, len(b)).mean() for _ in range(n)]
    return dict(mean=float(a.mean() - b.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                n_a=len(a), n_b=len(b))


def _pvalue(null, obs, alternative: str) -> float:
    import numpy as np
    eps = 1e-12 * max(1.0, abs(obs))
    if alternative == "two-sided":
        return float(np.mean(np.abs(null) >= abs(obs) - eps))
    if alternative == "greater":
        return float(np.mean(null >= obs - eps))
    if alternative == "less":
        return float(np.mean(null <= obs + eps))
    raise ValueError(alternative)


def paired_permutation_test(a, b, n: int = 10000, seed: int = 0, alternative: str = "two-sided",
                            exact_max: int = 16) -> dict:
    """Sign-flip permutation test of mean(a_i - b_i) = 0 for paired units. Exact enumeration of all 2^m sign
    patterns when m <= exact_max, else Monte Carlo with the observed pattern included ((k+1)/(n+1))."""
    import itertools
    import numpy as np
    d = _paired(a, b)
    m = len(d)
    if m == 0:
        return dict(mean_diff=None, p=None, n=0, exact=None)
    obs = float(d.mean())
    if m <= exact_max:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=m)))
        return dict(mean_diff=obs, p=_pvalue((signs * d).mean(1), obs, alternative), n=m, exact=True)
    g = np.random.default_rng(seed)
    null = (g.choice((1.0, -1.0), size=(n, m)) * d).mean(1)
    k = _pvalue(null, obs, alternative) * n
    return dict(mean_diff=obs, p=float((k + 1) / (n + 1)), n=m, exact=False)


def permutation_test(a, b, n: int = 10000, seed: int = 0, alternative: str = "two-sided") -> dict:
    """Two-sample permutation test of mean(a) - mean(b) = 0 (labels shuffled; Monte Carlo, (k+1)/(n+1))."""
    import numpy as np
    a, b = _finite(a), _finite(b)
    if len(a) == 0 or len(b) == 0:
        return dict(mean_diff=None, p=None, n_a=len(a), n_b=len(b))
    obs = float(a.mean() - b.mean())
    x = np.concatenate([a, b])
    g = np.random.default_rng(seed)
    null = np.empty(n)
    for i in range(n):
        p = g.permutation(x)
        null[i] = p[:len(a)].mean() - p[len(a):].mean()
    k = _pvalue(null, obs, alternative) * n
    return dict(mean_diff=obs, p=float((k + 1) / (n + 1)), n_a=len(a), n_b=len(b))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test for paired binary outcomes: b = pairs (A succeeds, B fails), c = the reverse.
    p = min(1, 2 * P[Binomial(b + c, 1/2) <= min(b, c)])."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m
    return min(1.0, 2 * tail)
