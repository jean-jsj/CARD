# baselines/ — the reference estimator grid

The four reference models that anchor the paper's results: for each demand
family, a 2x2 grid crossing **instrument use** (off/on) with **product-text
use** (off/on). Every estimator consumes only a cell's `public/` files and
writes the three submission CSVs of
[`docs/SUBMISSION_FORMAT.md`](SUBMISSION_FORMAT.md), so each
variant directory is a complete, scoreable submission.

| Variant | Instruments | Product text |
|---|---|---|
| `no_iv_no_text` | — | — |
| `iv_no_text` | `supply_cost_proxy` | — |
| `no_iv_text` | — | TF-IDF distances |
| `iv_text` | `supply_cost_proxy` | TF-IDF distances |

Each variant implements the cell's own demand system, fit from public files
alone, so the grid measures the value of each input with the functional form
held correct:

- **log-log cells** ([`loglog_grid.py`](../card_metrics/baselines/loglog_grid.py)): per-product Poisson
  (PPML) count models with store + week fixed effects; the IV corners add a
  control-function residual whose first stage regresses log price on
  `supply_cost_proxy`; the text corners weight a neighbor log-price index by
  TF-IDF text distances ([`text_distance.py`](../card_metrics/baselines/text_distance.py)) instead of
  brand membership.
- **discrete-choice cells** ([`probit_simulation.py`](../card_metrics/baselines/probit_simulation.py)):
  the covariance-probit mechanics re-simulated from public data — share
  inversion under a text-kernel error covariance, a price-utility scale
  calibrated to the reduced-form share regression
  ([`probit_shares.py`](../card_metrics/baselines/probit_shares.py); OLS naive / IV instrumented), and
  counterfactuals by re-simulating choices with common random numbers.

## Run

Needs the `[baselines]` extras (`pip install -e ".[baselines]"` adds
statsmodels + scikit-learn).

```bash
# all four corners of one cell (or drop --cells to run every downloaded cell)
python -m card_metrics.baselines.run_reference_grid \
    --cells-root benchmark/dev \
    --cells complex_log_log_endogenous_seed001 \
    --out-root reference_out

# score a corner
python -m card_metrics.evaluate_submission \
    --cell-dir benchmark/dev/complex_log_log_endogenous_seed001 \
    --submission-dir reference_out/iv_text/complex_log_log_endogenous_seed001 \
    --submission-name iv_text --out scores/iv_text.json
```

Rerunning `iv_text` on the full `complex_log_log_endogenous_seed001` cell
reproduces the shipped score in
[`submissions/reference_iv_text/`](../submissions/reference_iv_text/) (own-price
bias −0.001, substitution error 0.455, forecast error 0.461). The log-log
corners take minutes per cell; the probit corners are slower (share inversion +
simulation, tens of minutes per cell).

The four `reference_*` entries under [`submissions/`](../submissions/) hold the
shipped full-cell scores, and [docs/reference_results.html](reference_results.html)
is a browsable explorer. The reference models are not leaderboard entries.
