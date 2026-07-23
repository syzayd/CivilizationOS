"""Mixed-regime experiment: show that additive TCMF beats BOTH the causal-only oracle and
semantic RAG, because neither signal alone recovers both kinds of gold.

    python -m tcmfbench.run_mixed --n 300 --out results_mixed
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
from pathlib import Path

from . import _bootstrap  # noqa: F401
from .mixed import MixedConfig, generate_many_mixed
from . import methods as M
from . import metrics as MT

KS = (3, 5, 10)


def _score(ranked, mat) -> dict[str, float]:
    out = {f"recall@{k}": MT.recall_at_k(ranked, mat.gold_ids, k) for k in KS}
    out["causal@5"] = MT.recall_at_k(ranked, mat.gold_causal, 5)
    out["semantic@5"] = MT.recall_at_k(ranked, mat.gold_semantic, 5)
    out["root_mrr"] = MT.reciprocal_rank(ranked, mat.root_id)
    out["root_rank"] = MT.rank_of(ranked, mat.root_id) or (len(ranked) + 1)
    return out


def _agg(rows):
    return {k: (st.mean(r[k] for r in rows),
                st.pstdev([r[k] for r in rows]) if len(rows) > 1 else 0.0)
            for k in rows[0]}


async def _order(fn, mat):
    r = fn(mat)
    return await r if hasattr(r, "__await__") else r


async def _eval_rows(mats, method_fns):
    """Per-scenario score rows, unaggregated - lets callers pool rows across seeds
    before computing mean/std, instead of averaging per-seed averages."""
    per = {n: [] for n in method_fns}
    for mat in mats:
        for name, fn in method_fns.items():
            per[name].append(_score(await _order(fn, mat), mat))
    return per


async def _eval(mats, method_fns):
    per = await _eval_rows(mats, method_fns)
    return {n: _agg(rows) for n, rows in per.items()}


def _analytic_random_recall(gold_count: int, pool_size: int, ks) -> dict[str, float]:
    """Expected recall@k of a uniform-random ranking: E[recall@k] = k / pool_size
    (hypergeometric mean), capped at 1."""
    return {f"recall@{k}": min(1.0, k / pool_size) for k in ks}


_COLS = [f"recall@{k}" for k in KS] + ["causal@5", "semantic@5", "root_mrr", "root_rank"]


def _table(title, results, order=None):
    names = order or list(results)
    lines = [f"### {title}", "", "| method | " + " | ".join(_COLS) + " |",
             "|" + "---|" * (len(_COLS) + 1)]
    for n in names:
        a = results[n]
        cells = [f"{a[c][0]:.2f}±{a[c][1]:.2f}" if c != "root_rank" else f"{a[c][0]:.1f}"
                 for c in _COLS]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _methods():
    return {
        "random":       lambda m: M.rank_random(m, seed=1234),
        "semantic_rag": M.rank_semantic,
        "episodic":     M.rank_episodic,
        "causal_only":  lambda m: M.rank_causal_only(m, clean=True),
        "graph_ppr":    M.rank_graph_ppr,
        "tcmf_mult":    M.rank_tcmf_multiplicative,                       # OLD operator (pre-fix)
        "tcmf_add":     lambda m: M.rank_tcmf_additive(m, lam=4.0, clean=True),
        "tcmf_shipped": lambda m: M.rank_tcmf(m, lam=2.0),               # REAL retriever (fixed)
        "tcmf_rrf":     lambda m: M.rank_tcmf_rrf(m, clean=True),
    }


ORDER = ["random", "semantic_rag", "episodic", "causal_only", "graph_ppr",
         "tcmf_mult", "tcmf_add", "tcmf_shipped", "tcmf_rrf"]


def _mats(cfg, n, seed):
    return [M.materialize(sc, cfg.max_mem_per_citizen)
            for sc in generate_many_mixed(n, cfg, base_seed=seed)]


async def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = args.n
    cfg_overrides = {}
    if args.n_distractors is not None:
        cfg_overrides["n_distractors"] = args.n_distractors
    if args.n_noise is not None:
        cfg_overrides["n_noise"] = args.n_noise
    base = MixedConfig(**cfg_overrides)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    seed = seeds[0]  # used below for the (single-seed) ablations

    # ---- main comparison, pooled across all seeds (see run_eval.py for rationale) ----
    per_seed_rows = {}
    for s in seeds:
        per_seed_rows[s] = await _eval_rows(_mats(base, n, s), _methods())

    pooled_rows = {name: [] for name in _methods()}
    for s in seeds:
        for name, rows in per_seed_rows[s].items():
            pooled_rows[name].extend(rows)
    main = {name: _agg(rows) for name, rows in pooled_rows.items()}
    per_seed_agg = {s: {name: _agg(rows) for name, rows in per_seed_rows[s].items()}
                     for s in seeds}

    pool_size = base.total_gold() + base.n_distractors + base.n_noise
    analytic_random = _analytic_random_recall(base.total_gold(), pool_size, KS)

    mats = _mats(base, n, seed)  # single-seed pool for the remaining ablations

    # lambda tradeoff: causal weight vs which gold type is recovered
    lam_ab = await _eval(mats, {
        f"additive l={lam}": (lambda m, lam=lam: M.rank_tcmf_additive(m, lam=lam, clean=True))
        for lam in (0.5, 1, 2, 3, 4)
    })

    # edge-dropout robustness curve (overall recall@10)
    drop_methods = {
        "semantic_rag": M.rank_semantic,
        "causal_only":  lambda m: M.rank_causal_only(m, clean=True),
        "tcmf_add":     lambda m: M.rank_tcmf_additive(m, lam=4.0, clean=True),
    }
    dropouts = [0.0, 0.25, 0.5, 0.75, 1.0]
    curve = {}
    for p in dropouts:
        res = await _eval(_mats(MixedConfig(edge_dropout=p), n, seed), drop_methods)
        curve[p] = {m: res[m]["recall@10"][0] for m in drop_methods}

    ctbl = ["### Edge-dropout robustness (overall recall@10 vs fraction of causal edges missing)",
            "", "| method | " + " | ".join(f"drop={p}" for p in dropouts) + " |",
            "|" + "---|" * (len(dropouts) + 1)]
    for m in drop_methods:
        ctbl.append(f"| {m} | " + " | ".join(f"{curve[p][m]:.2f}" for p in dropouts) + " |")

    seed_tbl = ["### Main comparison recall@10, per seed (stability check)", "",
                "| method | " + " | ".join(f"seed={s}" for s in seeds) + " | pooled |",
                "|" + "---|" * (len(seeds) + 2)]
    for name in ORDER:
        cells = [f"{per_seed_agg[s][name]['recall@10'][0]:.2f}" for s in seeds]
        seed_tbl.append(f"| {name} | " + " | ".join(cells) + f" | {main[name]['recall@10'][0]:.2f} |")

    rand_tbl = ["### Random-baseline sanity check (analytic vs measured)", "",
                f"pool size = {base.total_gold()} gold + {base.n_distractors} distractors + "
                f"{base.n_noise} noise = {pool_size}", "",
                "| k | analytic E[recall@k] = k/pool | measured (random, pooled) |",
                "|---|---|---|"]
    for k in KS:
        rand_tbl.append(
            f"| {k} | {analytic_random[f'recall@{k}']:.3f} | "
            f"{main['random'][f'recall@{k}'][0]:.3f} |"
        )

    md = [
        "# TCMF Benchmark: Mixed Regime",
        "",
        f"Scenarios: {n} per seed | seeds: {seeds} ({len(seeds)}x{n} = {n * len(seeds)} total) | "
        f"chain_len: {base.chain_len} | "
        f"semantic_gold: {base.n_semantic_gold} | distractors: {base.n_distractors} | "
        f"noise: {base.n_noise} | pool size: {pool_size} | total gold: {base.total_gold()} "
        f"({base.chain_len - 1} causal + {base.n_semantic_gold} semantic)",
        "",
        "Neither signal alone recovers both gold types: `causal@5` = recall over causal-gold "
        "(graph-findable), `semantic@5` = recall over semantic-gold (similarity-findable). "
        "Additive TCMF should dominate both single-signal baselines on overall recall. Main "
        "comparison pools scenarios across all seeds before computing mean/std; the lambda "
        "tradeoff and dropout curve below use seed[0] only.",
        "",
        _table("Main comparison (mixed regime)", main, ORDER),
        "",
        "\n".join(seed_tbl),
        "",
        "\n".join(rand_tbl),
        "",
        _table("Additive lambda tradeoff (causal@5 vs semantic@5)", lam_ab),
        "",
        "\n".join(ctbl),
        "",
    ]
    (out / "RESULTS_MIXED.md").write_text("\n".join(md), encoding="utf-8")

    def _ser(d):
        return {nm: {k: {"mean": v[0], "std": v[1]} for k, v in a.items()} for nm, a in d.items()}
    (out / "results_mixed.json").write_text(json.dumps({
        "config": vars(base), "n": n, "seeds": seeds, "pool_size": pool_size,
        "analytic_random_recall": analytic_random,
        "main": _ser(main),
        "per_seed": {str(s): _ser(a) for s, a in per_seed_agg.items()},
        "lambda_tradeoff": _ser(lam_ab),
        "dropout_curve": {str(p): curve[p] for p in dropouts},
    }, indent=2), encoding="utf-8")

    print(_table("Main comparison (mixed regime)", main, ORDER).replace("### ", "== "))
    print("\n" + "\n".join(seed_tbl).replace("### ", "== "))
    print("\n" + "\n".join(rand_tbl).replace("### ", "== "))
    print("\n" + "\n".join(ctbl).replace("### ", "== "))
    print(f"\nWrote {out/'RESULTS_MIXED.md'} and {out/'results_mixed.json'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=str, default=None,
                    help="comma-separated base seeds, e.g. '0,1,2,3,4'; overrides --seed and "
                         "pools the main comparison's per-scenario rows across all of them")
    p.add_argument("--n-distractors", type=int, default=None,
                    help="override MixedConfig.n_distractors (default 6)")
    p.add_argument("--n-noise", type=int, default=None,
                    help="override MixedConfig.n_noise (default 8)")
    p.add_argument("--out", type=str, default="results_mixed")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
