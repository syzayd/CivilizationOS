# Reproducing every table and figure

Every command below is offline and deterministic (fixed seeds), except the real-text and
decision tiers, which need a local Ollama. All commands run from `research/tcmf_paper/`.
Runtimes are wall-clock on the machine this was last regenerated on (Windows, CPU-only,
Ollama serving `qwen2.5:3b-instruct` + `nomic-embed-text` locally).

Every number cited in `FINDINGS.md` and the paper draft traces to one of the committed
`results_*/*.json` files below by exact command. If you find a number that does not, that is
a bug per the standing rule in `NIGHT_QUEUE.md` - find its source or delete the claim.

## Synthetic tier (fully offline, no Ollama)

| Result dir | Command | ~Runtime | Cache |
|---|---|---|---|
| `results_main` | `python -m tcmfbench.run_eval --n 300 --out results_main` | seconds | none needed (synthetic embeddings, computed not cached) |
| `results_mixed` | `python -m tcmfbench.run_mixed --n 300 --out results_mixed` | seconds | none |
| `results_main_pool80` | `python -m tcmfbench.run_eval --n 300 --seeds 0,1,2,3,4 --n-distractors 20 --n-noise 55 --out results_main_pool80` | ~5 min | none |
| `results_mixed_pool80` | `python -m tcmfbench.run_mixed --n 300 --seeds 0,1,2,3,4 --n-distractors 20 --n-noise 55 --out results_mixed_pool80` | ~3.5 min | none |
| `results_main_scale` | `python -m tcmfbench.run_eval --n 300 --seeds 0,1,2,3,4 --n-distractors 20 --n-noise 55 --out results_main_scale` | ~5 min | none |
| `results_mixed_scale` | `python -m tcmfbench.run_mixed --n 300 --seeds 0,1,2,3,4 --n-distractors 20 --n-noise 55 --out results_mixed_scale` | ~3.5 min | none |
| `results_main_tuned` | `python -m tcmfbench.run_tuned --regime pure --n 300 --out results_main_tuned` | ~1 min | none |
| `results_mixed_tuned` | `python -m tcmfbench.run_tuned --regime mixed --n 300 --out results_mixed_tuned` | ~1 min | none |
| `results_spurious` | `python -m tcmfbench.run_spurious --n 300 --grid-n 100 --out results_spurious` | a few min | none |
| `results_theory` | `python -m tcmfbench.run_theory --out results_theory` | seconds | none |
| `results_lambda_sweep` | `python -m tcmfbench.run_lambda_sweep --n 300 --n-seeds 5 --out results_lambda_sweep` | ~2 min | none - the script itself asserts its own lambda=0.6/8 (mult) and lambda=4 (additive) points reproduce `results_main_scale/results.json` to machine precision before writing output |
| `figures/fig1_*`, `figures/fig2_*`, `figures/fig3_*`, `figures/fig4_*` | `python figures/make_figures.py --out figures` (needs `pip install -r requirements-bench.txt`; Fig 4 needs `results_lambda_sweep/` to already exist, else it is skipped with a printed message) | seconds | `figures/fig1_scenario.json`, `figures/fig3_pairs.json` (both committed; regenerated fresh each run - deterministic modulo the memory-id-string caveat in `tcmfbench/test_n10_figures.py::_strip_ids`, so every *numeric* field reproduces bit-for-bit even though `root_id`/`distractor_id` strings can shift between process invocations) |
| `figures/fig5_*`, `figures/fig6_*` | same `python figures/make_figures.py --out figures` command (N11); Fig 5 needs `results_spurious/` to already exist (see below), Fig 6 needs `results_decision/` to already exist (needs Ollama, see below) - each is skipped with a printed message if its input is missing | seconds | `figures/fig6_decision_ci.json` (committed; the Wilson-CI derivation of Fig 6's error bars from `results_decision.json`'s stored mean/n - see the caveat under "Decision-quality tier" below) |

**Checked, not a bug, but not bit-identical either:** running the documented command fresh
against the current codebase reproduces `results_main_scale`/`results_mixed_scale` exactly
(verified on `tcmf_add`, `graph_ppr`, `tcmf_shipped`, `causal_only`), but does **not** reproduce
`results_main_pool80`/`results_mixed_pool80` - e.g. mixed-regime `tcmf_add` recall@10 = 0.7983
(scale, and fresh) vs 0.7875 (pool80). This is a known, already-investigated discrepancy, not a
new finding: `results_*_pool80` came from an independently-written earlier harness (see
`FINDINGS.md`'s "Addendum - independent N01 replication" and `NIGHT_LOG.md`), deliberately left
unreconciled at the time as a noise estimate between two implementations of the same experiment,
pending N03's held-out tune split. N03 has since landed, and the paper's Table 8 (`sec:pool-scale`)
already cites the reproducible `scale` number (0.80), not the stale `pool80` number (0.79) - so
no claim in the paper is broken. The `_pool80` directories are superseded artifacts kept for
provenance (the addendum's own point is the ~0.01 cross-implementation noise estimate); `_scale`
is the one the paper actually draws from.

## Real-text tier (needs Ollama: `nomic-embed-text`)

| Result dir | Command | ~Runtime | Cache |
|---|---|---|---|
| `results_realtext` | `python -m tcmfbench.run_realtext --n 120 --out results_realtext` | first run: several min (encoder-bound); reruns: seconds | `results_realtext/emb_cache.json` (committed) |
| `results_n06` | `python -m tcmfbench.run_n06_domains --out results_n06` | first run: tens of min (200 scenarios to embed + ~1,200 local LLM calls for the decision tier); reruns: seconds | extends `results_realtext/emb_cache.json` and `results_decision/llm_cache.json` (both committed) |

**Discovered 2026-08-06 (N11), not fixed, "reruns: seconds" above is currently FALSE for a
fresh process:** `realtext.generate_realtext`'s domain pick is `rng.integers(len(DOMAINS))`
when `domain_idx` is not pinned (`run_realtext.py` and `run_decision.py` both call it that way;
only `run_n06_domains.py` pins `domain_idx` per domain and is unaffected). N05 (2026-08-04) grew
`DOMAINS` from 6 to 8 entries. A same-seeded `numpy.random.Generator.integers` call with a
different upper bound draws a different value from its very first call (confirmed directly:
`np.random.default_rng(0).integers(6)` and `np.random.default_rng(0).integers(8)` diverge), so
every scenario `run_realtext`/`run_decision` would generate today at any given seed differs from
what generated the currently-committed `results_realtext`/`results_decision` (both predate N05).
Concretely: attempting `python -m tcmfbench.run_decision --n 60 --out results_decision` in a
cloud sandbox with only the committed caches now fails with `RuntimeError: ... text not cached`
partway through, because it requests embeddings for scenarios the committed
`results_realtext/emb_cache.json` never saw. This does not invalidate the committed results
(each is still internally consistent and was itself produced by the exact command in its own
result directory, before N05 landed) - it means the command as written cannot be blindly re-run
today and expected to reproduce or extend them without either (a) pinning `domain_idx` to a
round-robin over 8 domains and accepting a *different*, not-bit-compatible committed artifact
after a fresh Ollama-backed regeneration, or (b) restricting the rerun to the original 6 domains
explicitly. Not fixed tonight: fixing it well is a judgment call on which domain set future runs
should use, and any regeneration needs Ollama (LOCAL-ONLY).

## Decision-quality tier (needs Ollama: `qwen2.5:3b-instruct`)

| Result dir | Command | ~Runtime | Cache |
|---|---|---|---|
| `results_decision` | `python -m tcmfbench.run_decision --n 60 --out results_decision` | first run: a few min (60 x 10 = 600 LLM calls); reruns: seconds, EXCEPT see the domain-determinism caveat directly above - a fresh rerun currently cannot complete offline | `results_decision/llm_cache.json` (committed) |

**Note for Fig 6 (N11):** `results_decision.json` stores only `mean`/`std` per method, not a raw
per-scenario correctness array, so there is nothing for `stats.bootstrap_ci` to resample. Fig 6's
error bars are Wilson score intervals (`stats.wilson_ci`, new in N11) computed from
`(round(mean * n), n)` instead - exact given the known n, pure numpy/`math.erf`, no scipy, and
not the same procedure as every other figure/table's bootstrap CI. `figures/fig6_decision_ci.json`
records this explicitly per method.

## LoCoMo public-benchmark check (N18, needs Ollama for embedding)

| Result dir | Command | Notes |
|---|---|---|
| `results_locomo` | `python -m tcmfbench.run_locomo_regime --data <path to locomo10.json>` | `locomo10.json` itself is not committed (third-party dataset, download separately) - `results_locomo/results_locomo.json` and the embedding cache are, so the finding is inspectable without re-downloading. |

## Not yet regenerable (open NIGHT_QUEUE.md items as of this writing)

N12 (ablation), N13 (second encoder + latency), N16 (scale/multi-crisis stress), N17
(TCMFBench spinoff) have no committed result artifact yet - their rows will be added to this
file the day each lands, not before. Do not cite a number for any of these; none exists.

## Structural validation (LaTeX draft, private repo)

From `paper/`: `.\build.ps1` runs `validate.py` (braces, environments, cross-references,
citations) then a full `pdflatex`/`bibtex`/`pdflatex`/`pdflatex` cycle, failing on any undefined
reference or citation (these otherwise compile silently as `??`). TinyTeX must be on PATH
(`%APPDATA%\TinyTeX\bin\windows`).
