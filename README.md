# CARD — Causal Recovery of Demand

[![CI](https://github.com/jean-jsj/CARD/actions/workflows/ci.yml/badge.svg)](https://github.com/jean-jsj/CARD/actions/workflows/ci.yml)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-CARD-yellow)](https://huggingface.co/datasets/jean-jsj/CARD)
[![DOI](https://img.shields.io/badge/DOI-10.57967%2Fhf%2F9681-blue)](https://doi.org/10.57967/hf/9681)
[![License](https://img.shields.io/badge/code%20Apache--2.0%20%7C%20data%20CC--BY--4.0-green)](LICENSE)

CARD is a benchmark that scores whether a demand model recovers the **causal** effect of price on sales — not just forecast accuracy. It ships synthetic retail scanner panels from a known data-generating process whose true elasticities and counterfactual outcomes are withheld for scoring.

- **Controlled confounding**: in half the cells, discount depth responds to a hidden demand shock, so models that fit observed sales well still get the counterfactuals wrong; two valid cost-side instruments are released.
- **Text-encoded market structure**: each product's marketing copy carries which products compete closely; the transactions alone carry only a weak trace.
- **Paired cells**: endogeneity on/off pairs share identical random draws, so failures are attributable.

## Installation

```bash
git clone https://github.com/jean-jsj/CARD && cd CARD
pip install -e ".[data]"
```

## Quickstart

```bash
card download --cell complex_log_log_endogenous_seed001 --mini   # ~18 MB starter slice
python examples/quickstart.py --cell-dir benchmark/dev_mini/complex_log_log_endogenous_seed001
# own-price bias  +1.00   (ranked headline; 0 = unbiased)
# forecast error   0.54   (displayed, never ranked)
```

That builds a deliberately naive submission — three CSVs ([format](docs/SUBMISSION_FORMAT.md)) — and scores it against the cell's hidden truth. Drop `--mini` for the full ~1 GB cell; leaderboard entries are scored on full cells.

Two notebooks, runnable locally or on Colab:

- [01_quickstart.ipynb](examples/01_quickstart.ipynb) — the data and the scoring loop, end to end [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jean-jsj/CARD/blob/main/examples/01_quickstart.ipynb)
- [02_endogeneity.ipynb](examples/02_endogeneity.ipynb) — the confounding appearing, and the instrument removing it [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jean-jsj/CARD/blob/main/examples/02_endogeneity.ipynb)

## Data

Four cells on Hugging Face ([`jean-jsj/CARD`](https://huggingface.co/datasets/jean-jsj/CARD)): {log-log, discrete-choice} demand × endogeneity {off, on}. Every cell is 40 products × 731 stores × 156 weeks (140 training weeks public; the final 16 weeks' sales withheld). Each cell has `public/` (panel, product texts, stores, the 16 counterfactual scenarios) and, on the dev seed, `hidden/` (scoring truth — **never model input**; eval-seed truth stays with the maintainer). Column-by-column schema: [docs/DATA.md](docs/DATA.md); questions: [docs/FAQ.md](docs/FAQ.md).

## What is scored

| Task | Metric |
|---|---|
| Sales forecasting | forecast error (WMAPE) and bias, on the 16 withheld weeks |
| Elasticity recovery | sign, F1, ranking (NDCG), magnitude error and bias on the 40×40 matrix |
| **Counterfactual prediction** | **own-price bias** (signed error of the predicted demand change under the flagship +10% scenario; 0 = unbiased) — **the ranked headline** — plus an unranked substitution error |
| Validity checks | label-free coherence checks on a real panel (Dominick's, [Kilts Center](https://www.chicagobooth.edu/research/kilts/research-data/dominicks)); PASS/WARN/FAIL |

The leaderboard ranks |own-price bias| per demand family and displays the forecast error beside it, never ranked — the benchmark's point is that the two diverge.

## Leaderboard

![log-log leaderboard](docs/leaderboard/leaderboard_log_log.svg)

<!-- LEADERBOARD:START -->
| Model | log-log: own-price bias (rank) | forecast error | discrete-choice: own-price bias (rank) | forecast error |
|---|---|---|---|---|
| *no verified entries yet* | | | | |
<!-- LEADERBOARD:END -->

To enter: score the full dev cells locally, then open a PR adding `submissions/<your-model>/` with your predictions ([CONTRIBUTING](.github/CONTRIBUTING.md)). The maintainer verifies on private eval seeds. The four reference models (instruments × text grid, code in [`card_metrics/baselines/`](card_metrics/baselines/)) are not entries; their results are in [docs/reference_results.html](docs/reference_results.html) and [`submissions/`](submissions/).

## Repository layout

```
card_metrics/    scoring math + harness + reference baselines (pip package; `card` CLI)
examples/        2 notebooks + quickstart.py
docs/            data dictionary, FAQ, submission format, datasheet, baselines, leaderboard assets
submissions/     reference scores; verified leaderboard entries land here by PR
scripts/         maintainer tooling (leaderboard render, mini-slice builder)
tests/           self-contained; no data download needed
```

The generator is withheld while the evaluation phase runs; its frozen source is committed to by SHA-256 in [docs/GENERATOR_COMMITMENT.md](docs/GENERATOR_COMMITMENT.md) and released when the phase closes.

## Citation

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
