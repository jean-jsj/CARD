"""Participant-facing evaluation harness for the causal-demand benchmark.

Scores submissions against a benchmark cell directory across the evaluation layers:

* **Layer 1** — demand prediction (revenue-weighted WMAPE / WMPE) on the held-out weeks; runs on both the synthetic and actual-data arms.
* **Layer 2** — elasticity estimation (J×J matrix: sign / F1 / NDCG / WMAPE / RMSE / WMPE, substitute-complement-unrelated stratification).
* **Layer 3** — counterfactual demand response (headline): a PAIR of pooled numbers on the flagship +10% scenario — signed own-price WMPE (identification bias; the leaderboard ranks by |WMPE| ascending) and unsigned substitution WAPE on the competitor Δq (both category-netted, micro-averaged; see SUBMISSION_FORMAT.md).
* **Layer 4** — label-free causal-coherence validity checks; the scored layer of the actual-data arm (real data has no counterfactual truth).

The deliverable is a score per model and a leaderboard — no statistical tests, no across-model comparison machinery. The metric mathematics lives in the standalone `causal_demand_metrics` package (single source of truth); this package adds the scoring interface:

* `metrics.evaluate_submission` — CLI scoring one model's predictions against one cell, with optional placement among shipped reference scores.
* `metrics.evaluate_all` — score every cell you have predictions for.
* `metrics.leaderboard` — CLI assembling scores into a (README-ready) table.

See SUBMISSION_FORMAT.md for the file contracts.
"""
