# metrics/ — scoring harness

The **participant-facing scorer**: it grades a submission against a benchmark cell and assembles the leaderboard, built on `causal_demand_metrics` (the metric math; single source of truth). No statistical tests — the deliverable is a score per model and a leaderboard.

## What you submit

Up to three CSVs per cell on the synthetic arm, two on the actual-data arm (see [`SUBMISSION_FORMAT.md`](SUBMISSION_FORMAT.md)):

| File | Contents |
|---|---|
| `layer1_demand_predictions.csv` | Forecast units for the holdout weeks. |
| `layer2_elasticities.csv` | Your J×J own/cross elasticity matrix. |
| `layer3_counterfactual_deltas.csv` | Predicted demand change per product per intervention. |
| `layer1_actual_predictions.csv` | (actual arm) Forecast units on the real POS panel. |
| `layer4_actual_deltas.csv` | (actual arm) Predicted Δq under the public own-price sweep. |

Any layer you omit scores `not_submitted`; a malformed file scores `invalid_format` (the others still score). The CSV formats are stable across metric versions.

## What you get back

- **Headline (counterfactual prediction):** two pooled numbers on the flagship +10% scenario, both category-netted and micro-averaged — **own-price bias** (signed WMPE) on the focal product's Δq, and **substitution error** (unsigned WAPE) on the competitor Δq. The leaderboard ranks by **|own-price bias| ascending** (closest-to-zero first); the substitution error rides alongside. Both are also reported per scenario across the full 16-scenario sweep.
- **Additional outputs:** a forecasting + elasticity-recovery diagnostics CSV (which capability is missing) and the cell × 16-scenario counterfactual matrix (which scenarios hurt you).
- **Actual-data arm:** sales forecasting (same forecast error/bias) plus the validity checks — label-free causal-coherence checks (own-price sign, substitution sign, own-elasticity band coverage, sign-flip monotonicity), each with a bootstrap CI. Elasticity recovery and counterfactual prediction are not scored on real data (no counterfactual truth exists there).

## Release model (dev / eval split)

- **Dev seed 1** ships with the scoring truth → score locally, instantly, offline.
- **Eval seeds** ship public-only; truth stays with the maintainer. The README leaderboard column = headline averaged over eval seeds ± spread.
- `hidden/` is also a **data-access rule**: models consume `public/` files only. The truth files exist for scoring, never as model input.

## Quickstart

```bash
# score one dev cell
python3 -m metrics.evaluate_submission \
    --cell-dir benchmark/dev/complex_log_log_endogenous_seed001 \
    --submission-dir my_model/complex_log_log_endogenous_seed001/ \
    --submission-name my_model \
    --out scores/complex_log_log_endogenous.json

# or score every dev cell you have predictions for
python3 -m metrics.evaluate_all --cells-root benchmark/dev/ \
    --submissions-root my_model/ --submission-name my_model --out-dir scores/

# insight CSVs
python3 -m metrics.diagnostics scores/*.json --out diag.csv --layer3-out ivs.csv
```

## CLI map

| Command | Does |
|---|---|
| `evaluate_submission` | Score one submission against one cell. |
| `evaluate_all` | Score every dev cell you have predictions for. |
| `leaderboard` | Assemble/rank scores (maintainer: `--aggregate-seeds` for the README table). |
| `diagnostics` | The two insight CSVs. |
