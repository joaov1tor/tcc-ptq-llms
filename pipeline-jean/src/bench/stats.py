"""Nonparametric statistics for the Stage 6 analysis of the PTQ benchmark.

Stdlib only — no scipy/numpy — so the venv stays small and the analysis runs
anywhere Python 3.12+ runs. Implements the tests the TCC asks for (§6.6):
Kruskal-Wallis across precisions, Mann-Whitney U pairwise against FP16,
Holm-Bonferroni correction, Cliff's δ and r=|Z|/√N as effect sizes, plus a
simple Pareto front for the trade-off matrix. Reference formulas come from
Lehmann (Nonparametrics, §2.5) and Siegel & Castellan (Nonparametric Statistics
for the Behavioral Sciences, §6).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _normal_sf(z: float) -> float:
    """P(Z >= z) for standard normal; math.erf-based so no scipy needed."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _normal_two_sided_p(z: float) -> float:
    return 2.0 * _normal_sf(abs(z))


def _chi2_sf_wilson_hilferty(x: float, df: int) -> float:
    """Survival function P(χ² > x | df), via the Wilson-Hilferty cube-root
    normal approximation. Error < 0.3% for df >= 2 and x in the usual range.
    """
    if df <= 0:
        raise ValueError("df must be >= 1")
    if x <= 0:
        return 1.0
    # (χ²/df)^(1/3) is approximately N(1 - 2/(9df), 2/(9df))
    h = 2.0 / (9.0 * df)
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - h)) / math.sqrt(h)
    return _normal_sf(z)


def _rank_with_ties(values: Sequence[float]) -> tuple[list[float], float]:
    """Return midranks (ties get the average rank) and the tie-correction sum
    Σ(t³ − t), where t is the size of each tie group.
    """
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    tie_sum = 0.0
    i = 0
    n = len(indexed)
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        midrank = (i + j) / 2.0 + 1.0  # 1-based average rank
        t = j - i + 1
        if t > 1:
            tie_sum += t ** 3 - t
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = midrank
        i = j + 1
    return ranks, tie_sum


def mann_whitney_u(x: Sequence[float], y: Sequence[float]) -> dict:
    """Two-sided Mann-Whitney U with tie correction and normal approximation.

    Returns {U, Z, p, n1, n2, n, ties_correction}. For n1+n2 < 10 the normal
    approximation is less reliable; the caller should check `n` before quoting
    the p-value verbatim.
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        raise ValueError("both groups must be non-empty")
    combined = list(x) + list(y)
    ranks, tie_sum = _rank_with_ties(combined)
    R1 = sum(ranks[:n1])
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)
    N = n1 + n2
    mu = n1 * n2 / 2.0
    # Lehmann §2.5: ties correct σ² by multiplying by (1 - T/(N³-N)).
    tie_factor = 1.0 - tie_sum / (N ** 3 - N) if N > 1 else 1.0
    sigma = math.sqrt(n1 * n2 / 12.0 * (N + 1) * tie_factor)
    # Continuity correction brings the discrete distribution closer to normal.
    if sigma == 0.0:
        z = 0.0
    else:
        numerator = U1 - mu
        correction = 0.5 if numerator > 0 else -0.5 if numerator < 0 else 0.0
        z = (numerator - correction) / sigma
    p = _normal_two_sided_p(z)
    return {"U": U, "U1": U1, "U2": U2, "Z": z, "p": p, "n1": n1, "n2": n2, "n": N, "ties_correction": tie_sum}


def kruskal_wallis(groups: Sequence[Sequence[float]]) -> dict:
    """Kruskal-Wallis H test with tie correction.

    Returns {H, df, p, N, k}. If fewer than 2 non-empty groups are given,
    returns H=0 and p=1.
    """
    non_empty = [list(g) for g in groups if len(g) > 0]
    k = len(non_empty)
    if k < 2:
        return {"H": 0.0, "df": 0, "p": 1.0, "N": sum(len(g) for g in non_empty), "k": k}
    combined: list[float] = []
    for g in non_empty:
        combined.extend(g)
    ranks, tie_sum = _rank_with_ties(combined)
    N = len(combined)
    pos = 0
    rank_sum_sq_over_n = 0.0
    for g in non_empty:
        n_i = len(g)
        R_i = sum(ranks[pos:pos + n_i])
        rank_sum_sq_over_n += R_i ** 2 / n_i
        pos += n_i
    H = (12.0 / (N * (N + 1))) * rank_sum_sq_over_n - 3.0 * (N + 1)
    # Siegel & Castellan §8.1: divide H by C = 1 - T/(N³-N) when there are ties.
    C = 1.0 - tie_sum / (N ** 3 - N) if N > 1 else 1.0
    if C > 0:
        H /= C
    df = k - 1
    p = _chi2_sf_wilson_hilferty(H, df)
    return {"H": H, "df": df, "p": p, "N": N, "k": k, "ties_correction": tie_sum}


def holm_bonferroni(pvalues: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the same order as `pvalues`.

    Controls the family-wise error rate; equivalent to multiplying each p by
    (m − i) where i is its rank among the sorted p-values, then enforcing
    monotonicity so adjusted p never decreases with rank.
    """
    m = len(pvalues)
    if m == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        candidate = min(1.0, p * (m - rank))
        running_max = max(running_max, candidate)
        adjusted[orig_idx] = running_max
    return adjusted


def cliff_delta(x: Sequence[float], y: Sequence[float]) -> dict:
    """Cliff's δ = (#(x>y) − #(x<y)) / (n1·n2). O(n log n) via sort+binary search."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        raise ValueError("both groups must be non-empty")
    y_sorted = sorted(y)
    greater = 0
    lesser = 0
    for xi in x:
        # count y strictly < xi and y strictly > xi
        lo, hi = 0, n2
        while lo < hi:
            mid = (lo + hi) // 2
            if y_sorted[mid] < xi:
                lo = mid + 1
            else:
                hi = mid
        less_than_xi = lo
        lo, hi = 0, n2
        while lo < hi:
            mid = (lo + hi) // 2
            if y_sorted[mid] <= xi:
                lo = mid + 1
            else:
                hi = mid
        less_or_equal_xi = lo
        greater += less_than_xi  # y < xi -> x > y
        lesser += n2 - less_or_equal_xi  # y > xi -> x < y
    delta = (greater - lesser) / (n1 * n2)
    return {"delta": delta, "interp": cliff_interp(delta), "n1": n1, "n2": n2}


def cliff_interp(delta: float) -> str:
    """Romano et al. 2006 thresholds for Cliff's δ magnitude."""
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    if a < 0.330:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def rank_r_effect(z: float, n: int) -> float:
    """Mann-Whitney effect size r = |Z| / √N (Cohen 1988: 0.1 small, 0.3 medium, 0.5 large)."""
    if n <= 0:
        return 0.0
    return abs(z) / math.sqrt(n)


def pareto_front(
    items: Iterable[dict],
    *,
    key: str = "name",
    minimize: Sequence[str] = ("ppl", "memory"),
    maximize: Sequence[str] = ("tps",),
) -> dict[str, dict]:
    """Classify each item as 'pareto' or list the items that dominate it.

    Returns {name: {"pareto": bool, "dominated_by": [names]}}. Comparison uses
    strict dominance: a dominates b iff a is at least as good as b in every
    axis AND strictly better in at least one.
    """
    items_list = [dict(it) for it in items]
    result: dict[str, dict] = {}
    for a in items_list:
        dominated_by: list[str] = []
        for b in items_list:
            if a is b:
                continue
            at_least_as_good = all(b[m] <= a[m] for m in minimize) and all(b[m] >= a[m] for m in maximize)
            strictly_better = any(b[m] < a[m] for m in minimize) or any(b[m] > a[m] for m in maximize)
            if at_least_as_good and strictly_better:
                dominated_by.append(b[key])
        result[a[key]] = {"pareto": not dominated_by, "dominated_by": dominated_by}
    return result
