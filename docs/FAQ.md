# FAQ

**Do I need any license or real data to participate?**
No. The scored benchmark is entirely synthetic and downloads directly from
[Hugging Face](https://huggingface.co/datasets/jean-jsj/CARD). The licensed
inputs (IRI scanner data, TDLinx/Spectra store data) entered only as
calibration targets; no raw record from any licensed source is released. The
optional actual-data arm uses the public Dominick's data, downloaded from the
Kilts Center under its academic terms.

**How big is the download?**
Each full cell is roughly 1 GB (most of it the counterfactual-scenario context
and truth). Start with the ~18 MB `dev_mini` slices instead
(`card download --cell complex_log_log_endogenous_seed001 --mini`)
— every notebook runs on them. Score real entries on the full cells.

**Why does the dev cell ship its own answer key (`hidden/`)?**
So you can iterate locally with instant scoring. Dev-seed numbers are
self-reported; leaderboard entries are verified by the maintainer on private
eval seeds. Using `hidden/` files as model *input* disqualifies an entry — the
split is a data-access rule, not just a directory layout.

**Do I have to use the product text? Or the instruments?**
No. Every input is optional; the benchmark exists to measure what each one
buys. The reference grid quantifies both axes: ignoring the instruments costs
own-price bias in the endogeneity-on cells, and ignoring the text costs
substitution accuracy.

**Do I have to submit all files / all cells?**
No. Absent tasks are reported `not_submitted`; absent cells simply don't get
scores. But missing *rows* within a submitted file are penalized, not excluded
(omission cannot shrink your denominator).

**What is ranked, exactly?**
One number per demand family: |own-price bias| on the flagship +10% scenario
in the endogeneity-on cell (0 = unbiased). The sales forecast error is
displayed beside it and never ranked; the substitution error and the full
elasticity scorecard are reported unranked. See the README's leaderboard
section and [docs/SUBMISSION_FORMAT.md](SUBMISSION_FORMAT.md).

**Why is the generator withheld? How do I know the data won't change?**
Publishing the generator would let anyone regenerate the hidden truth for the
eval seeds. The frozen source is committed to by SHA-256 in
[GENERATOR_COMMITMENT.md](GENERATOR_COMMITMENT.md) and will be released to
match that hash after the evaluation phase; every released file carries its
own hash in the per-cell `release/MANIFEST.json`. The construction is
documented at the equation level in the paper appendix.

**Why are there no images, only text?**
The multimodal surface is marketing copy only. The substitution geometry is
carried in the meaning of the text — validated to be recoverable from meaning
and from nothing else (surface statistics, brand labels, and text length all
probe at chance).

**Can my model be text-blind and still recover substitution from sales?**
Only weakly: the transactions carry a trace of the ordering, but text-blind
reference methods recover far less of it than text-aware ones. That gap is a
designed finding, not an accident.

**Why do my elasticities use my own baseline in the denominator?**
Because you never observe the true baseline sales. Submit
`sum(delta_q) / (0.01 x sum(your_baseline_q))`; the denominator convention is
identical across submissions and absorbed into the magnitude metrics.

**What Python do I need?**
Python ≥ 3.9. `pip install -e ".[data]"` covers scoring and downloads; the
reference baselines add `".[baselines]"` (statsmodels + scikit-learn);
notebooks also use matplotlib.

**Something looks wrong in the data or scorer.**
Open a GitHub issue. Scoring math is frozen between minor versions — behavior
changes get a version bump and a changelog note so scores stay comparable.
