# causal_demand_metrics/

**Pure scoring math** — the single source of truth for all evaluation tasks, shared by the construction pipeline and the participant harness (`metrics/`). numpy + pandas only: no file I/O, no DGP code, no hidden-truth generation. pip-installable.

## What it scores (four tasks)

**Sales forecasting.** Accuracy of held-out observed-sales forecasts, revenue-weighted across products:
- **Demand-WMAPE**
- **Demand-WMPE**

**Elasticity recovery.** How well the estimated own/cross price-elasticity matrix matches the truth:
- **own-price sign accuracy**
- **own-price WMAPE / WMPE**
- **cross-price NDCG**
- **cross-price F1** (substitute / complement / unrelated)
- **cross-price WMAPE / RMSE**

Magnitudes are scored against the **total** elasticity; the substitute/complement/unrelated labels are assigned on the **switching-only** part, so a category-wide change can't masquerade as substitution.

**Validity checks (no ground truth).** Causal-coherence sanity checks that read only the submission plus public price moves, so they score on **real POS** data with no hidden truth. Each rate carries a **bootstrap CI**:
- **own-price sign** — fraction of focal store-weeks where price↑⇒units↓ (↓⇒↑)
- **substitution sign** — competitor redistribution flowing the right way under a hike, reported both **|Δq|-weighted** and **unweighted per-competitor** (a big correct competitor can't hide many small wrong ones), category margin netted out; a per-product **complements** set flips the expected sign
- **own-elasticity range coverage** — fraction of own elasticities in a plausible reference band (sign-correct, not extreme). Defaults grounded in CPG elasticity meta-analyses (Tellis 1988; Bijmolt, van Heerde & Pieters 2005) and store-level scanner estimates (Hoch et al. 1995); `FACIAL_TISSUE_OWN_BAND` pins the category
- **cross-elasticity plausibility** — off-diagonal magnitude sanity: fraction extreme, and fraction whose |cross| exceeds the priced product's own |ε|
- **monotonicity** — focal response flips across a paired +x% / −x% sweep

`coherence_gate` folds these into a **PASS / WARN / FAIL** verdict — the validity checks are a *gate*, not a ranker; a model can pass every gate and still be wrong on magnitudes.

**Counterfactual prediction (the headline).** A PAIR of numbers from ONE scenario (`sweep_single_share_highest_plus10`, the flagship +X% hike) — **own-price bias** (signed WMPE) and **substitution error** (unsigned WAPE), both on the category-netted Δq. See *The counterfactual headline* below for what each measures.

## The counterfactual headline

Raising one product's price produces a vector of demand changes across the whole category — the product's own drop, plus the gains spread over its competitors. The headline reads **two separate numbers** off the SAME submitted Δq̂, both from the single scenario `sweep_single_share_highest_plus10`, each on the category-netted residual (each side netted by its own category shift `Δq − ΔM·share`, `ΔM = Σ Δq`, so a category-wide magnitude move can't masquerade as substitution):

- **own-price response — signed WMPE** — the size and direction of the price-changed (focal) product's own demand change, pooled over store-weeks: `Σ(Δq̂_f − Δq*_f) / Σ|Δq*_f|`. This is the identification/bias axis — an endogeneity-biased estimate gets it wrong; 0 = unbiased, the sign shows over- vs under-shoot.
- **substitution — unsigned WAPE** — honest magnitude accuracy of the competitor (all non-focal) Δq, pooled over store-weeks, on raw mass and **geometry-blind** (no `d_total` closeness weighting): `Σ_{k≠f}|Δq̂_k − Δq*_k| / Σ_{k≠f}|Δq*_k|`. Lower = closer to the true competitor redistribution.

Both numbers are micro-averaged: numerators and denominators are pooled across all store-weeks and divided once (no per-store-week ratio-then-average, no renormalization). The leaderboard ranks by **|own-price WMPE| ascending** (closest-to-zero bias first); the substitution WAPE rides alongside.

## Which tasks score on which data arm

| Task | Synthetic arm | Actual arm |
|---|---|---|
| Sales forecasting | ✓ scored | ✓ scored |
| Elasticity recovery | ✓ scored | — (no ground truth) |
| Counterfactual prediction (headline) | ✓ scored (ranked) | — (no ground truth) |
| Validity checks (label-free) | — (actual-arm task) | ✓ scored |

The actual-data results are a **DIAGNOSTIC PANEL**, reported in their own `(cell_type, data_arm)` leaderboard partition — they are NOT ranked into the synthetic own-price-bias headline and are NOT a headline number. The validity checks are **label-free** (real-data causal-coherence, no hidden truth): they are the sole actual-arm scoring task, while sales forecasting runs on both arms.

## Module map

| Module | Scores |
|---|---|
| `sales_forecasting` | Revenue-weighted Demand-WMAPE / WMPE on observed holdout sales. |
| `elasticity` | J×J elasticity matrix (sign / class-F1 / NDCG / magnitude); closed-form log-log truth. |
| `headline_decomposition` | The counterfactual headline (own-price bias, signed WMPE, + substitution error, unsigned competitor-Δq WAPE, both category-netted, pooled). |
| `validity_checks` | Label-free causal-coherence checks with bootstrap CIs: own-price sign, substitution sign (weighted + unweighted, complement-aware), own/cross elasticity plausibility, ± sweep monotonicity, PASS/WARN/FAIL gate. **No hidden truth — scores on real POS.** |

## Design constraints

- **Pure math.** No I/O, no DGP, no hidden-truth generation. The one truth generator that needs a hidden replay (covariance-probit ε*) lives in `benchmark_pipeline/`, not here.
- **Frozen definitions.** Semantic changes bump the version; `scores.json` carries `schema_version` + `benchmark_version` so scores from different definitions are not silently mixed.

## Status

`0.5.0.dev0` (pre-release). `pip install -e .`
