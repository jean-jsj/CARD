# Reference model: `iv_no_text`

One corner of the benchmark's 2x2 reference grid (with/without the released
instruments x with/without the product text). This corner uses the two released instruments (supply_cost_proxy primary, promo_cost secondary) but not the product text.

- **Model.** Each cell's own demand family (the log-linear demand system or the
  discrete-choice system), fit from the `public/` files alone, with two-stage least squares in the price equation.
  The functional form is held correct by construction, so the grid isolates the
  value of each input rather than model misspecification.
- **Data used.** `public/` files only. Text corners read `product_text`.
- **Scores.** `scores/<cell>.json` — dev seed 1, scored with this repository's
  harness (`python -m metrics.evaluate_all`).
- **Predictions.** Full submission-format CSVs are hosted with the dataset
  (Hugging Face `jean-jsj/CARD`, `reference/iv_no_text/`) — the per-cell counterfactual
  prediction files are too large for a git repository.
- **Contact.** Maintainer (see `CITATION.cff`).
