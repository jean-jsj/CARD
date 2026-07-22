# Reference model: `no_iv_no_text`

One corner of the benchmark's 2x2 reference grid (with/without the released
instruments x with/without the product text). This corner uses neither the released instruments nor the product text.

- **Model.** Each cell's own demand family (the log-linear demand system or the
  discrete-choice system), fit from the `public/` files alone, with ordinary least squares on the public prices and promotions.
  The functional form is held correct by construction, so the grid isolates the
  value of each input rather than model misspecification.
- **Data used.** `public/` files only. Text corners read `product_text`;
  simple cells ship no text, so the text corners cover complex cells only.
- **Scores.** `scores/<cell>.json` — dev seed 1, scored with this repository's
  harness (`python -m metrics.evaluate_all`).
- **Predictions.** Full submission-format CSVs are hosted with the dataset
  (Hugging Face `jean-jsj/CARD`, `reference/no_iv_no_text/`) — the per-cell Layer-3
  files are too large for a git repository.
- **Contact.** Maintainer (see `CITATION.cff`).
