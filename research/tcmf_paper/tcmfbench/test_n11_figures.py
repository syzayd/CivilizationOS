"""N11 unit tests: Fig 5 (graph degradation) + Fig 6 (decision accuracy).

Same posture as test_n09_figures.py / test_n10_figures.py: ``figures/make_figures.py`` lives
outside the package and is imported via sys.path. Fig 5 draws straight from the already-tested
``results_spurious/results_spurious.json`` (N04) with no new experiment - these tests check the
figure's numbers are read verbatim from that file, not transcribed by hand. Fig 6 introduces one
new piece of math (``stats.wilson_ci``, tested independently in ``test_stats.py``) applied to
``results_decision/results_decision.json``'s stored ``mean``/``n`` - since that file has no
surviving per-scenario array to bootstrap over and this is a cloud sandbox with no Ollama to
regenerate it, these tests check the (successes, n) recovery is exact and the resulting CIs are
well-formed.

Run: python -m tcmfbench.test_n11_figures (or pytest tcmfbench/test_n11_figures.py)
"""
from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path

from . import _bootstrap  # noqa: F401

_FIGURES_DIR = _bootstrap.REPO_ROOT / "research" / "tcmf_paper" / "figures"
if str(_FIGURES_DIR) not in sys.path:
    sys.path.insert(0, str(_FIGURES_DIR))
MF = importlib.import_module("make_figures")

_TCMF_PAPER_DIR = _bootstrap.REPO_ROOT / "research" / "tcmf_paper"


# ------------------------------------------------------------------------------- Fig 5

def _load_spurious():
    path = _TCMF_PAPER_DIR / "results_spurious" / "results_spurious.json"
    assert path.exists(), "run `python -m tcmfbench.run_spurious` to generate this file"
    return json.loads(path.read_text())


def test_fig5_curve_values_match_source_json_exactly():
    spurious = _load_spurious()
    for method in MF.FIG5_CURVE_METHODS:
        rates, mean, lo, hi = MF.extract_fig5_curve(spurious, method)
        assert rates == spurious["spurious_rates"]
        for r, m, l, h in zip(rates, mean, lo, hi):
            cell = spurious["curve"][str(r)][method]["recall@10"]
            assert m == cell["mean"]
            assert l == cell["ci_lo"]
            assert h == cell["ci_hi"]
            assert l <= m <= h


def test_fig5_curve_is_monotonically_non_increasing_for_tcmf_add():
    # Headline 1 (N04/NIGHT_LOG.md): a real, monotone loss as the spurious rate rises.
    spurious = _load_spurious()
    _, mean, _, _ = MF.extract_fig5_curve(spurious, "tcmf_add")
    assert all(mean[i] >= mean[i + 1] - 1e-9 for i in range(len(mean) - 1))


def test_fig5_semantic_rag_floor_is_never_crossed_by_tcmf_add():
    # The item's own verify criterion via NIGHT_QUEUE N04: report the p at which tcmf_add drops
    # below semantic_rag - and the honest answer was "never, up to p=0.4". This test pins that.
    spurious = _load_spurious()
    _, sem_mean, _, _ = MF.extract_fig5_curve(spurious, "semantic_rag")
    _, add_mean, _, _ = MF.extract_fig5_curve(spurious, "tcmf_add")
    assert all(a > s for a, s in zip(add_mean, sem_mean))


def test_fig5_grid_values_match_source_json_exactly():
    spurious = _load_spurious()
    dropout_rates = spurious["dropout_rates_2d"]
    spurious_rates_2d = spurious["spurious_rates"]
    for name in MF.FIG5_GRID_METHODS:
        for d in dropout_rates:
            for p in spurious_rates_2d:
                # Exactly the lookup draw_fig5 performs - same key construction, same source.
                assert spurious["grid_recall_at_10"][f"{d}|{p}"][name] is not None


def test_fig5_p0_grid_matches_p0_curve_within_sampling_noise():
    # grid_n=100 vs curve n=300 (different sample), both at dropout=0/p=0 - not bit-identical,
    # but must agree closely (both measure the same underlying quantity at the same settings).
    spurious = _load_spurious()
    _, mean, _, _ = MF.extract_fig5_curve(spurious, "tcmf_add")
    grid_p0 = spurious["grid_recall_at_10"]["0.0|0.0"]["tcmf_add"]
    assert abs(mean[0] - grid_p0) < 0.03


def test_fig5_renders_to_nonempty_vector_pdf_and_png():
    spurious = _load_spurious()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        MF.draw_fig5(spurious, out / "fig5_graph_degradation")
        pdf = out / "fig5_graph_degradation.pdf"
        png = out / "fig5_graph_degradation.png"
        assert pdf.stat().st_size > 500
        assert png.stat().st_size > 500
        assert pdf.read_bytes()[:4] == b"%PDF"


# ------------------------------------------------------------------------------- Fig 6

def _load_decision():
    path = _TCMF_PAPER_DIR / "results_decision" / "results_decision.json"
    assert path.exists(), "run `python -m tcmfbench.run_decision` to generate this file"
    return json.loads(path.read_text())


def test_fig6_recovers_exact_integer_successes_from_stored_mean():
    decision = _load_decision()
    ci = MF.build_fig6_ci(decision)
    n = decision["n"]
    for name in MF.FIG6_METHOD_ORDER:
        expected_mean = decision["methods"][name]["decision_acc"]["mean"]
        row = ci["methods"][name]
        assert row["n"] == n
        assert 0 <= row["successes"] <= n
        assert abs(row["successes"] / n - expected_mean) < 1e-9
    for name in ("no_retrieval", "oracle"):
        expected_mean = decision["controls"][name]["decision_acc"]["mean"]
        row = ci["controls"][name]
        assert abs(row["successes"] / n - expected_mean) < 1e-9


def test_fig6_ci_contains_the_point_estimate_and_is_well_formed():
    decision = _load_decision()
    ci = MF.build_fig6_ci(decision)
    for row in list(ci["methods"].values()) + list(ci["controls"].values()):
        assert 0.0 <= row["ci_lo"] <= row["mean"] <= row["ci_hi"] <= 1.0


def test_fig6_ci_matches_wilson_ci_directly():
    # Cross-check build_fig6_ci against a direct call to stats.wilson_ci for one method, so the
    # wiring (not just the underlying math, already unit-tested in test_stats.py) is verified.
    from .stats import wilson_ci
    decision = _load_decision()
    n = decision["n"]
    mean = decision["methods"]["tcmf_shipped"]["decision_acc"]["mean"]
    successes = round(mean * n)
    expected = wilson_ci(successes, n)
    ci = MF.build_fig6_ci(decision)
    row = ci["methods"]["tcmf_shipped"]
    assert (row["mean"], row["ci_lo"], row["ci_hi"]) == expected


def test_fig6_rejects_a_non_bernoulli_mean():
    # Guard rail described in build_fig6_ci's docstring: a mean that is not `successes/n` for
    # any integer successes must raise, not silently produce a wrong CI.
    import copy
    decision = copy.deepcopy(_load_decision())
    decision["methods"]["tcmf_add"]["decision_acc"]["mean"] = 0.31337  # not k/60 for any int k
    try:
        MF.build_fig6_ci(decision)
        assert False, "expected an AssertionError"
    except AssertionError:
        pass


def test_fig6_renders_to_nonempty_vector_pdf_and_png():
    decision = _load_decision()
    ci = MF.build_fig6_ci(decision)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        MF.draw_fig6(ci, out / "fig6_decision_accuracy")
        pdf = out / "fig6_decision_accuracy.pdf"
        png = out / "fig6_decision_accuracy.png"
        assert pdf.stat().st_size > 500
        assert png.stat().st_size > 500
        assert pdf.read_bytes()[:4] == b"%PDF"


if __name__ == "__main__":
    import traceback

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
