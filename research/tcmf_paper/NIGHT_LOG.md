# TCMF Night Log

Append-only, one entry per night, newest last. Written by the nightly hardening routine
(see `NIGHT_QUEUE.md`).

Each entry must record: which queue item, why it was chosen, **what the numbers actually
said** (including when they hurt the paper), what was verified vs assumed, and the LaTeX
delta the private paper repo needs.

---

## 2026-07-23 (setup, not a night task)

- **Task:** Built the 14-night hardening queue (`NIGHT_QUEUE.md`) and registered the nightly
  cloud routine that works it.
- **Why:** The paper's evidence base, not its prose, is what stands between it and a
  reviewer's "this is a small handcrafted synthetic benchmark." Two weeks of one-step-per-
  night on scale, domains, baselines, ablations, uncertainty, and figures.
- **Result:** Queue N01-N14 written, ordered so the riskiest item (N01, larger candidate
  pool) runs first - it is the one that could invalidate every later night. N06 and N14 are
  flagged LOCAL-ONLY (Ollama); the cloud agent skips them and takes the next CLOUD-OK item.
- **Verified:** benchmark inventory confirmed against the real tree - `GenConfig` already
  carries `n_distractors` / `n_noise` (so N01 is a threading job, not a rewrite), `run_eval`
  and `run_mixed` currently expose only `--n/--seed/--out`, no scipy and no matplotlib are
  installed locally (hence pure-numpy stats in N02 and a `requirements-bench.txt` in N09).
- **Next:** N01.

---

## 2026-07-23 - N01: Larger candidate pool + multi-seed harness

- **Item chosen:** N01, the lowest-numbered OPEN item and CLOUD-OK (this is a cloud sandbox:
  no Ollama, no access to Zaid's machine, so any LOCAL-ONLY item - none were reached tonight
  since N01 is first - would have been skipped for the next CLOUD-OK one).
- **What was built:** Added `--n-distractors`, `--n-noise`, `--seeds` (comma list) to both
  `tcmfbench/run_eval.py` and `tcmfbench/run_mixed.py`, threaded into `GenConfig`/`MixedConfig`
  (both dataclasses already had the fields, confirming the setup night's inventory - this was a
  threading job, not a rewrite). Multi-seed support pools per-scenario rows across all seeds
  before computing mean/std (not an average of per-seed averages), and reports a per-seed
  recall@10 stability table so a one-off seed can't hide behind an aggregate. Also added an
  `_analytic_random_recall()` closed-form check (`E[recall@k] = k/pool`, hypergeometric mean)
  to both runners, and a `random` baseline to `run_mixed.py`'s method set (it was missing
  there, which meant the mixed regime had no sanity check on the harness at all).
  Backward compatibility checked: default single-seed, no-override invocation reproduces the
  original small-pool numbers in `FINDINGS.md` bit for bit.
- **Verified, not assumed:** Traced `materialize(sc, max_mem_per_citizen=8)` and confirmed by
  direct measurement (not just code-reading) that the per-citizen split does NOT silently
  re-cap the enlarged pool: `n_citizens` scales with pool size
  (`ceil(len(memories)/max_mem_per_citizen)`), and every downstream `retrieve()` call passes
  `k=10_000`, far above any realistic pool. At pool=78, `_episodic_scores()` returned scores
  for all 78 memories, zero truncation, for 5 independently-generated scenarios. Added this as
  a permanent regression test, plus a test that the analytic random-baseline formula matches a
  hand-computed known case (gold=3, pool=10) and the empirical measured baseline within 0.03
  over 200 scenarios: `tcmfbench/tests/test_pool_scaling.py`, 4/4 passing.
- **Experiment:** reran both regimes at pool~80 (20 distractors, 55 noise, chain_len=4
  unchanged) across 5 seeds, n=300 each (1500 scenarios pooled per regime):
  `python -m tcmfbench.run_eval  --n 300 --seeds 0,1,2,3,4 --n-distractors 20 --n-noise 55 --out results_main_pool80`
  `python -m tcmfbench.run_mixed --n 300 --seeds 0,1,2,3,4 --n-distractors 20 --n-noise 55 --out results_mixed_pool80`
  All 5 seeds agreed to 2 decimal places on every method's recall@10 in both regimes - not a
  one-off. The random baseline's measured recall@10 (0.139 pure / 0.134 mixed) matched the
  analytic k/pool prediction (0.128 / 0.125) closely, confirming the harness measures what it
  claims to.
- **What the numbers actually said (this is the important part):**
  - **Pure regime: the margin survives.** `tcmf_add` still ties the causal oracle exactly at
    recall@10 = 1.00 and both crush every non-causal baseline (semantic_rag/episodic = 0.00,
    tcmf_mult = 0.01, graph_ppr = 0.33, tcmf_rrf = 0.97). One real degradation: the REAL
    shipped retriever `tcmf_shipped`'s recall@10 falls from 1.00 (old ~17-candidate pool) to
    **0.80** at pool~78. Root-cause placement is untouched (root_mrr = 1.00, root_rank = 1.0,
    unchanged) - it still nails which memory is the root cause, it just doesn't pull every
    other causal-gold witness into the top 10 as reliably at this scale.
  - **Mixed regime: the margin against `graph_ppr` does NOT survive - this is the honest
    negative result.** At the old pool, `tcmf_add` (0.98) and `tcmf_shipped` clearly beat
    `graph_ppr` (0.80) on overall recall@10. At pool~80, `graph_ppr`'s recall@10 is
    **unchanged at 0.80**, while `tcmf_add` falls to **0.79** and `tcmf_shipped` falls to
    **0.73** - `graph_ppr` now edges out both. Stable across all 5 seeds, not sampling noise.
    Cause, from the `causal@5`/`semantic@5` breakdown: `tcmf_add` still perfectly recovers
    causal-gold (causal@5 = 1.00, unchanged) but its semantic-gold recovery collapses from
    0.38 (old pool) to 0.18 at the larger pool - `lambda=4` (picked by eye at the old, small
    pool, never revisited) now overweights the causal term against a much bigger competing
    episodic pool and crowds out the semantic-gold memories it used to also retrieve.
    `graph_ppr` doesn't have this problem: it scores by proximity to graph *events*, not by
    competing in a normalized episodic pool against 55 extra noise memories, so its number
    doesn't move with pool size.
  - I did not retune, reseed, or reframe to make this look better. It is written down plainly
    here and in `FINDINGS.md`'s new N01 section.
- **Assumed, not verified this night:** the fusion-operator, lambda, threshold, depth-weighting,
  and difficulty ablations in `run_eval.py`/`run_mixed.py` still run on a single seed
  (`seeds[0]`) at whatever pool the caller passes - only the *main comparison* table was
  multi-seeded and rerun at pool~80 tonight, per the item's explicit ask. Re-running every
  ablation multi-seed at pool~80 is out of scope for one night and is not blocking; flagging so
  a later night doesn't assume it already happened.
- **LaTeX delta for the private paper repo:** could not reach `syzayd/tcmf-paper` - no `gh` CLI
  in this sandbox (`gh: command not found`); continued without it per the standing instructions
  and recorded the delta as prose (also in `FINDINGS.md`'s N01 section):
  1. Any claim shaped like "TCMF beats every baseline at recall@10 in both regimes" must become
     regime-specific: holds in the pure regime at pool~80; does NOT hold in the mixed regime
     against `graph_ppr` until N03's held-out retune is checked.
  2. Add a pool-size ablation table/figure: recall@10 at pool~19 vs pool~80, both regimes, all
     methods - sourced from `results_main_pool80/results.json` and
     `results_mixed_pool80/results_mixed.json` (committed, regenerable).
  3. Note the benchmark's synthetic tiers are now validated up to pool~80 across 5 independent
     seeds (1500 scenarios/regime), with a closed-form random-baseline sanity check, superseding
     any prose that still says "~17-19 candidates" as the only scale tested.
- **Files touched:** `tcmfbench/run_eval.py`, `tcmfbench/run_mixed.py` (CLI + pooling +
  analytic check), `tcmfbench/tests/test_pool_scaling.py` (new), `FINDINGS.md` (new N01
  section), `NIGHT_QUEUE.md` (N01 -> DONE), committed result dirs `results_main_pool80/`,
  `results_mixed_pool80/`.
- **Next:** N02 (bootstrap CIs + paired significance) is now higher priority than its queue
  position suggests, precisely because tonight found a real margin difference (graph_ppr vs
  tcmf_add/tcmf_shipped in the mixed regime) that needs a significance test, not just point
  estimates, before anyone decides whether it is real or within noise (5-seed agreement to 2
  decimals suggests real, but N02's proper machinery should confirm). N03's held-out lambda
  retune at this pool size is also now directly motivated by tonight's finding, not just
  "tuned on test" hygiene.
