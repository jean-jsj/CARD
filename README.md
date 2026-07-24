# CARD — Causal Recovery of Demand

[![CI](https://github.com/jean-jsj/CARD/actions/workflows/ci.yml/badge.svg)](https://github.com/jean-jsj/CARD/actions/workflows/ci.yml)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CARD-yellow)](https://huggingface.co/datasets/jean-jsj/CARD)
[![DOI](https://img.shields.io/badge/DOI-10.57967%2Fhf%2F9681-blue)](https://doi.org/10.57967/hf/9681)
[![Code License](https://img.shields.io/badge/code-Apache--2.0-green)](LICENSE)
[![Data License](https://img.shields.io/badge/data-CC%20BY%204.0-green)](https://huggingface.co/datasets/jean-jsj/CARD)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-blue)](pyproject.toml)

**Can a model that fits observed demand well still recover causal price response, substitution, and counterfactual outcomes when prices and promotions are endogenous?**

This benchmark pairs synthetic retail scanner panels with a multimodal product surface (marketing-copy product descriptions that carry the true substitution geometry). Demand is simulated from a known data-generating process; prices and promotions respond to hidden demand shocks in half the cells, so estimators that ignore endogeneity fit the observed data well and still get the counterfactuals wrong. True elasticities and counterfactual outcomes are hidden and used only for scoring.

## News

- **2026-07**: example-notebook ladder ([`examples/`](examples/README.md)), the reference-grid estimator code ([`baselines/`](baselines/README.md)), ~18 MB `dev_mini` starter slices, a [data dictionary](docs/DATA.md), and a [FAQ](docs/FAQ.md).
- **2026-07-22**: data release v0.1 on Hugging Face, archived under DOI [10.57967/hf/9681](https://doi.org/10.57967/hf/9681); leaderboard open for submissions.

## Try it in 3 minutes

```bash
git clone https://github.com/jean-jsj/CARD && cd CARD
pip install -e .                       # scoring needs numpy + pandas only
pip install huggingface_hub

python examples/download_data.py --cell complex_log_log_endogenous_seed001 --mini
python examples/quickstart.py --cell-dir benchmark/dev_mini/complex_log_log_endogenous_seed001
```

`quickstart.py` builds a deliberately naive submission from the public files and scores it against the cell's hidden truth. A correct run ends with a score JSON whose headline block reads:

```
"counterfactual_prediction": { "headline": {
    "scenario": "sweep_single_share_highest_plus10",
    "own_price":    { "own_price_wmpe": 1.0, ... },      # predicted no change: fully biased
    "substitution": { "substitution_wape": 1.0, ... } }} # predicted no change: full error
"sales_forecasting": { "demand_wmape": 0.544, ... }
```

You should beat this easily. The `--mini` flag fetches a 10-store starter slice; drop it for the full ~1 GB cell (leaderboard entries are scored on full cells). Then continue with the notebooks — each runs on the mini slice, locally or on Colab:

| | Notebook | |
|---|---|---|
| 1 | [Explore the data](examples/01_explore_data.ipynb) — panel, product text, instruments, scenarios | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jean-jsj/CARD/blob/main/examples/01_explore_data.ipynb) |
| 2 | [Build and score a submission](examples/02_score_a_submission.ipynb) — the full loop, end to end | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jean-jsj/CARD/blob/main/examples/02_score_a_submission.ipynb) |
| 3 | [Endogeneity and instruments, live](examples/03_endogeneity_and_instruments.ipynb) — the core phenomenon on the paired cells | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jean-jsj/CARD/blob/main/examples/03_endogeneity_and_instruments.ipynb) |

## The 2×2 grid

| Axis | Values |
|---|---|
| Demand family | **log-log** demand system (distance-disciplined cross-elasticities) / **structured random-coefficients discrete choice** (embedding-distance error covariance) |
| Endogeneity | **off** (control) / **on** (promotion *depth* responds to a hidden demand shock; cost-based instruments stay valid) |

Every cell is the full market: **40 products, 731 stores, 156 weeks** — 140 public training weeks plus 16 holdout-context weeks whose prices/promotions are public but whose sales are withheld. The two released instruments are `supply_cost_proxy` (primary) and `promo_cost` (secondary). Each product carries a `product_text` marketing-copy description (the substitution geometry's carrier). Store markets and brand codes are pseudonymized. The two cells of an endogeneity on/off pair draw byte-identical random sequences — the toggle changes exactly one coupling constant, so any change in a method's error between the pair is attributable to the confounding itself.

## Data

Hosted on Hugging Face: [`jean-jsj/CARD`](https://huggingface.co/datasets/jean-jsj/CARD). Column-by-column schema: [docs/DATA.md](docs/DATA.md).

The four cells are `complex_{log_log,covariance_probit}_{exogenous,endogenous}_seed001` (the `complex_` prefix is part of the frozen cell identifiers). Each cell directory contains:

- `public/` — everything your model may consume: `transactions_train_public.csv`, `transactions_holdout_context_public.csv`, `products_public.csv`, `stores_public.csv`, and `counterfactual_sweep_context_public.csv` (the 16 scored scenarios).
- `hidden/` — **dev seed only**: the scoring truth (holdout sales, elasticity matrix, counterfactual Δq), so you can score locally and instantly.
- `release/` — per-cell manifest (SHA-256 per file), release notes, datasheet.

A `dev_mini/` tree holds ~18 MB 10-store slices of the log-log pair for quickstarts and notebooks.

<details>
<summary><b>The data-access rule (what <code>hidden/</code> is and is not)</b></summary>

The dev seed ships its own answer key so you can iterate offline with instant scoring. `hidden/` files are scoring truth, **never model input** — entries whose models consumed them are disqualified. Dev-seed numbers are self-reported; leaderboard entries are verified by the maintainer against private eval-seed truth. Eval seeds ship `public/` only.

</details>

## What is scored

| Task | Question | Metrics |
|---|---|---|
| **Sales forecasting** | Do you forecast held-out sales? | **forecast error** (revenue-weighted WMAPE, lower = better) and **forecast bias** (signed WMPE, 0 = best) |
| **Elasticity recovery** | Do you recover the J×J price-response matrix? | **sign accuracy** and **substitute/complement F1** (higher = better), **cross-effect ranking** (NDCG, higher = better), **magnitude error** (WMAPE/RMSE, lower = better), **elasticity bias** (WMPE, 0 = best) |
| **Counterfactual prediction** (headline) | Do you predict the demand change under a +10% price move? | **own-price bias** (signed WMPE, 0 = unbiased) and **substitution error** (WAPE; 1.0 = predicting no change, lower = better) |
| **Validity checks** (actual arm) | Are your real-data predictions causally coherent? | label-free sign/band/monotonicity checks → PASS / WARN / FAIL |

The headline is counterfactual prediction, read off one flagship scenario: the leaderboard ranks by |own-price bias| (0 = unbiased), and the substitution error is reported alongside (competitor-redistribution error; predicting "no change" scores the full mass). Both are category-netted so a category-wide shift can't masquerade as substitution. The interesting comparison is *endogeneity-on vs -off*: a purely predictive model can win on forecast error and still carry a large own-price bias in the endogeneity-on cells.

The benchmark also runs an **actual-data arm** (sales forecasting + validity checks) on a real point-of-sale panel: the **Dominick's Finer Foods** scanner data published by the [Kilts Center for Marketing](https://www.chicagobooth.edu/research/kilts/research-data/dominicks) (Chicago Booth), **Bathroom Tissues** category — the closest real analog to the synthetic facial-tissue category. The data are free for academic research (attribution to the Kilts Center required) and are not redistributed here: download the category files from the Kilts Center and point `--actual-data-root` at them; the loader ([`metrics/actual_data.py`](metrics/actual_data.py)) is deterministic, so every participant reconstructs the identical panel. Real data has no counterfactual truth, so elasticity recovery and counterfactual prediction are synthetic-arm only — and because the Dominick's files are public, the actual-arm forecast is an honor-system diagnostic rather than an adversarially-hidden target.

## Baselines

[`baselines/`](baselines/README.md) ships the four reference estimators behind the paper's results — a 2×2 grid crossing **instrument use** with **product-text use**, each implementing the cell's own demand system from public files alone (`pip install -e ".[baselines]"`). Their full-cell scores are committed under [`submissions/`](submissions/) and browsable in [docs/reference_results.html](docs/reference_results.html); rerunning a corner reproduces its shipped score. They anchor the two axes any submission navigates: without the instruments the endogeneity-on cells bias the own-price response; without the text the substitution structure is mostly out of reach.

## Leaderboard

Entries are ranked by **|own-price bias|** in each family's endogeneity-on cell (0 = unbiased; the sign stays visible for direction). The two family columns rank independently — the same entry can hold different ranks in each. The sales forecast error is reported alongside and never enters the rank: the benchmark's point is that the two can diverge. The plots pair each arena cell with its endogeneity-off control.

![log-log leaderboard](leaderboard/leaderboard_log_log.svg)

![discrete-choice leaderboard](leaderboard/leaderboard_covariance_probit.svg)

<!-- LEADERBOARD:START -->
| Model | log-log: own-price bias (rank) | forecast error | discrete-choice: own-price bias (rank) | forecast error |
|---|---|---|---|---|
| *no verified entries yet* | | | | |
<!-- LEADERBOARD:END -->

**Metric names.** **Own-price bias** = signed weighted mean percentage error (WMPE) of the predicted demand change for the price-changed product, headline +10% scenario; 0 = unbiased, the sign shows over- vs under-shoot. **Sales forecast error** = revenue-weighted mean absolute percentage error (WMAPE) of predicted units over the 16 withheld weeks; lower is better. The **substitution error** (WAPE on competitor demand changes; 1.0 = predicting no change) and the full elasticity-recovery scorecard are reported per cell in `scores/` and in the diagnostics CSVs, not ranked here.

The four reference models are not leaderboard entries; their results are reported in the paper and browsable in [`docs/reference_results.html`](docs/reference_results.html) (open locally — an interactive explorer with a per-family toggle). Their per-cell scores live in [`submissions/`](submissions/) (`reference_*`), and their predictions are hosted with the dataset (`reference/` on Hugging Face).

## Submit your model

1. Build on `public/` files only; write the three CSVs per cell ([format](metrics/SUBMISSION_FORMAT.md)).
2. Score locally on the dev seed: `python -m metrics.evaluate_all --cells-root benchmark/dev --submissions-root my_model/ --submission-name my_model --out-dir scores/`.
3. Open a PR adding `submissions/<your-model-name>/` with your predictions and an `entry.md` ([CONTRIBUTING.md](CONTRIBUTING.md)).
4. The maintainer scores your predictions on the private eval seeds and updates the leaderboard above.

## The generator is withheld — verifiably

The DGP code and calibrated parameters are not published while the evaluation phase runs (publishing them would let anyone regenerate the hidden truth). The construction is documented at the equation level in the paper appendix, and a SHA-256 commitment to the exact frozen generator source is published in [`GENERATOR_COMMITMENT.md`](GENERATOR_COMMITMENT.md); the archive will be released to match that hash after the evaluation phase.

## Repository layout

| Path | What |
|---|---|
| [`causal_demand_metrics/`](causal_demand_metrics/README.md) | Scoring math (pip-installable; numpy + pandas only). |
| [`metrics/`](metrics/README.md) | Scoring harness: `evaluate_submission`, `evaluate_all`, `leaderboard`, `diagnostics`. |
| [`baselines/`](baselines/README.md) | The four reference estimators (instruments × text grid). |
| [`examples/`](examples/README.md) | Notebook ladder, data download, naive-baseline quickstart. |
| [`docs/`](docs/) | [Data dictionary](docs/DATA.md), [FAQ](docs/FAQ.md), [reference-results explorer](docs/reference_results.html). |
| [`submissions/`](submissions/) | Reference-model entries and scores; verified leaderboard entries land here by PR. |
| [`DATASHEET.md`](DATASHEET.md) | Datasheet for the dataset (Gebru et al. format). |
| `tests/` | Test suite (self-contained; no data download needed). |

## License & citation

Code: **Apache-2.0** ([LICENSE](LICENSE)). Data: **CC BY 4.0** (declared on the Hugging Face dataset card). The panels are synthetic, calibrated to moments of the IRI academic scanner dataset; no real transactions are included.

```bibtex
@misc{hong2026card,
  author    = {Hong, Juwon and Hwang, Minha and Shankar, Venkatesh},
  title     = {CARD: Causal Recovery of Demand},
  year      = {2026},
  doi       = {10.57967/hf/9681},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/jean-jsj/CARD}
}
```

(Also in [CITATION.cff](CITATION.cff) and [CITATION.bib](CITATION.bib); the paper reference will replace this upon publication.)
