# Contributing

## Leaderboard entries (verified-entry flow)

1. Build your model on the `public/` files only — `hidden/` is scoring truth,
   never model input. Score locally on the dev seed with
   `python -m metrics.evaluate_all`.
2. Open a PR adding a directory under `submissions/<your-model-name>/`
   containing:
   - your prediction CSVs for each cell you enter (see
     [`metrics/SUBMISSION_FORMAT.md`](metrics/SUBMISSION_FORMAT.md)), or a
     download link if they exceed repo-friendly size;
   - `entry.md`: model description, data used (which public files, text used
     or not), compute, and a contact.
3. The maintainer runs your predictions against the private eval-seed truth
   and updates the README leaderboard (mean ± spread across eval seeds). Dev
   numbers are self-reported; eval numbers are maintainer-verified.

An entry can be withdrawn at any time by PR. Entries that consumed `hidden/`
files as model input are disqualified.

## Issues & fixes

Bug reports and scoring-harness fixes are welcome as issues/PRs. The scoring
math in `causal_demand_metrics/` is frozen between minor versions — behavior
changes require a version bump and a changelog note, so scores stay
comparable. The data themselves are versioned on the Hugging Face hub;
data issues are tracked here.

Run the tests with `python -m pytest tests/ -q` (no data download needed).
