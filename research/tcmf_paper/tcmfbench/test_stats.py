"""N02 unit tests: hand-computed known answers for the pure-numpy statistics module. Run
directly:

    python -m tcmfbench.test_stats

No scipy; pure numpy/stdlib only, per the standing rules in NIGHT_QUEUE.md.
"""
from __future__ import annotations

import numpy as np

from . import _bootstrap  # noqa: F401
from .stats import bootstrap_ci, holm_bonferroni, wilcoxon_signed_rank, wilson_ci


# ------------------------------------------------------------------------- bootstrap_ci

def test_bootstrap_ci_constant_values_has_zero_width() -> None:
    # resampling a constant array with replacement can only ever produce that same constant,
    # so the CI must collapse to a point at exactly that value.
    point, lo, hi = bootstrap_ci([5.0, 5.0, 5.0, 5.0, 5.0], seed=0)
    assert point == 5.0
    assert lo == 5.0
    assert hi == 5.0


def test_bootstrap_ci_single_value() -> None:
    point, lo, hi = bootstrap_ci([3.5], seed=0)
    assert point == lo == hi == 3.5


def test_bootstrap_ci_contains_true_mean_and_is_seeded_deterministic() -> None:
    values = [0.0, 0.0, 0.0, 0.0, 10.0]  # mean = 2.0
    point, lo, hi = bootstrap_ci(values, seed=42, n_boot=5000)
    assert point == 2.0
    assert lo <= point <= hi
    assert 0.0 <= lo and hi <= 10.0
    # same seed -> bit-identical CI (determinism the queue's "seeded" requirement needs)
    point2, lo2, hi2 = bootstrap_ci(values, seed=42, n_boot=5000)
    assert (point, lo, hi) == (point2, lo2, hi2)
    # a different seed is allowed to move the CI (it is a different resample draw)
    point3, lo3, hi3 = bootstrap_ci(values, seed=7, n_boot=5000)
    assert point3 == 2.0  # point estimate never depends on the resample seed


# ------------------------------------------------------------------------- wilcoxon_signed_rank

def test_wilcoxon_hand_computed_all_positive_n3() -> None:
    # d = [1, 2, 3], all positive. Exact null distribution of W+ over subsets of {1,2,3}:
    # sums -> {0:1, 1:1, 2:1, 3:2, 4:1, 5:1, 6:1} out of 2**3 = 8 sign assignments.
    # observed W+ = 6 (the max): P(W+>=6) = 1/8, P(W+<=6) = 1 -> two-sided p = 2*min(.) = 0.25.
    a = [3.0, 4.0, 5.0]
    b = [2.0, 2.0, 2.0]
    p = wilcoxon_signed_rank(a, b)
    assert abs(p - 0.25) < 1e-12


def test_wilcoxon_hand_computed_symmetric_n3() -> None:
    # d = [1, -2, 3]: ranks of |d| = [1, 2, 3]. W+ = 1 + 3 = 4, W- = 2.
    # P(W+ == 4): subsets of {1,2,3} summing to 4 -> {1,3} only -> count 1 -> P=1/8.
    # P(W+>=4) = counts[4]+counts[5]+counts[6] = 1+1+1 = 3 -> 3/8.
    # P(W+<=4) = counts[0..4] = 1+1+1+2+1 = 6 -> 6/8. two-sided p = 2*min(3/8,6/8) = 0.75.
    a = [1.0, -2.0, 3.0]
    b = [0.0, 0.0, 0.0]
    p = wilcoxon_signed_rank(a, b)
    assert abs(p - 0.75) < 1e-12


def test_wilcoxon_zero_difference_returns_one() -> None:
    # every paired difference is exactly zero -> nothing to rank -> p = 1.0 by definition.
    a = [1.0, 2.0, 3.0, 4.0]
    b = [1.0, 2.0, 3.0, 4.0]
    assert wilcoxon_signed_rank(a, b) == 1.0


def test_wilcoxon_tied_ranks_case() -> None:
    # d = [-2, 2] (a tie: |d| = [2, 2]). Both differences tie for rank 1.5 each.
    # W+ = 1.5 (just the +2 entry), mean_W under H0 = n(n+1)/4 = 2*3/4 = 1.5 exactly ->
    # W+ sits exactly at its null expectation -> z = 0 -> p = 1.0 exactly. This is the
    # tied-ranks case the queue calls out as where signed-rank implementations break: a
    # naive implementation that does not average tied ranks would instead assign ranks
    # {1, 2} and get an incorrectly nonzero z.
    a = [0.0, 4.0]
    b = [2.0, 2.0]
    p = wilcoxon_signed_rank(a, b)
    assert abs(p - 1.0) < 1e-12


def test_wilcoxon_identical_method_against_itself_is_null() -> None:
    # a method compared against itself: every paired difference is zero everywhere ->
    # the queue's own "self-contrast returns p ~= 1.0" verification criterion.
    rng = np.random.default_rng(0)
    scores = rng.random(50)
    assert wilcoxon_signed_rank(scores, scores) == 1.0


def test_wilcoxon_large_n_tied_normal_approximation_matches_manual_formula() -> None:
    # n = 30 (> 25, forces the normal-approximation branch) with a deliberate tie group so
    # the tie-variance correction path is exercised. Recompute mean/var/z by hand from the
    # same formulas the implementation uses and check the returned p matches independently.
    rng = np.random.default_rng(1)
    d = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=30) * rng.choice([-1.0, 1.0], size=30)
    a = d.astype(float)
    b = np.zeros_like(d)

    abs_d = np.abs(d)
    order = np.argsort(abs_d, kind="mergesort")
    ranks = np.empty(30)
    sorted_abs = abs_d[order]
    i = 0
    while i < 30:
        j = i
        while j + 1 < 30 and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    w_pos = ranks[d > 0].sum()
    n = 30
    mean_w = n * (n + 1) / 4.0
    _, tie_counts = np.unique(abs_d, return_counts=True)
    tie_corr = np.sum(tie_counts ** 3 - tie_counts) / 48.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr
    if w_pos > mean_w:
        z = (w_pos - mean_w - 0.5) / np.sqrt(var_w)
    elif w_pos < mean_w:
        z = (w_pos - mean_w + 0.5) / np.sqrt(var_w)
    else:
        z = 0.0
    import math
    expected_p = min(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))), 1.0)

    got_p = wilcoxon_signed_rank(a, b)
    assert abs(got_p - expected_p) < 1e-12


# ------------------------------------------------------------------------- holm_bonferroni

def test_holm_bonferroni_hand_computed() -> None:
    # p = [0.01, 0.02, 0.03, 0.20], n=4, already sorted ascending.
    # multipliers by rank (0-indexed): (n-0)=4, (n-1)=3, (n-2)=2, (n-3)=1
    # raw products: 0.04, 0.06, 0.06, 0.20 -> enforce running max (monotone non-decreasing):
    # 0.04, 0.06, 0.06, 0.20.
    pvals = [0.01, 0.02, 0.03, 0.20]
    adj = holm_bonferroni(pvals)
    expected = [0.04, 0.06, 0.06, 0.20]
    for got, want in zip(adj, expected):
        assert abs(got - want) < 1e-12


def test_holm_bonferroni_caps_at_one_and_enforces_monotonicity() -> None:
    # p = [0.5, 0.6], n=2: rank0 (p=0.5) * (2-0) = 1.0 exactly (capped, not exceeded).
    # rank1 (p=0.6) * (2-1) = 0.6, but must be >= the previous adjusted value (1.0) ->
    # becomes 1.0 too.
    adj = holm_bonferroni([0.5, 0.6])
    assert adj == [1.0, 1.0]


def test_holm_bonferroni_unsorted_input_order_preserved() -> None:
    # same p-values as the first test but shuffled input order; adjusted p-values must
    # come back in the SAME order as the input, matching each original p-value's identity.
    pvals = [0.20, 0.01, 0.03, 0.02]  # = [p3, p0, p2, p1] of the sorted-order test above
    adj = holm_bonferroni(pvals)
    expected = [0.20, 0.04, 0.06, 0.06]
    for got, want in zip(adj, expected):
        assert abs(got - want) < 1e-12


def test_holm_bonferroni_empty() -> None:
    assert holm_bonferroni([]) == []


# ------------------------------------------------------------------------- wilson_ci

def test_wilson_ci_hand_computed_n4_x2() -> None:
    # Textbook example (successes=2, n=4, phat=0.5): the standard-normal 97.5th percentile is
    # z=1.959963984540054 (an independent value, looked up/computed separately from this
    # module's own `_norm_ppf`, so this checks the function against an outside source, not
    # itself). Wilson's closed form:
    #   denom = 1 + z^2/n = 1.960375...
    #   center = (phat + z^2/2n) / denom = 0.5 exactly (phat=0.5 is the symmetric case)
    #   half = z*sqrt(phat*(1-phat)/n + z^2/4n^2) / denom = 0.349945...
    # -> lo=0.150055, hi=0.849945, hand-verified against a published Wilson interval table.
    phat, lo, hi = wilson_ci(2, 4)
    assert phat == 0.5
    assert abs(lo - 0.150055) < 1e-3
    assert abs(hi - 0.849945) < 1e-3


def test_wilson_ci_phat_is_successes_over_n() -> None:
    phat, lo, hi = wilson_ci(21, 60)
    assert abs(phat - 0.35) < 1e-12
    assert lo <= phat <= hi


def test_wilson_ci_symmetric_around_half_when_phat_is_half() -> None:
    phat, lo, hi = wilson_ci(30, 60)
    assert phat == 0.5
    assert abs((phat - lo) - (hi - phat)) < 1e-9


def test_wilson_ci_stays_inside_unit_interval_at_the_extremes() -> None:
    # p=0 and p=1 are exactly where the normal (Wald) interval breaks (zero width); Wilson
    # must not.
    _, lo0, hi0 = wilson_ci(0, 20)
    assert lo0 == 0.0
    assert 0.0 < hi0 < 1.0
    _, lo1, hi1 = wilson_ci(20, 20)
    assert abs(hi1 - 1.0) < 1e-9
    assert 0.0 < lo1 < 1.0


def test_wilson_ci_narrows_as_n_grows_at_fixed_phat() -> None:
    _, lo_small, hi_small = wilson_ci(35, 100)   # phat = 0.35
    _, lo_big, hi_big = wilson_ci(350, 1000)      # same phat, 10x the n
    assert (hi_small - lo_small) > (hi_big - lo_big)


def test_wilson_ci_n_zero_is_nan() -> None:
    phat, lo, hi = wilson_ci(0, 0)
    assert all(v != v for v in (phat, lo, hi))  # NaN != NaN


def _run_all() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    _run_all()
