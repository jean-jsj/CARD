"""Layer 2 elasticity-estimation metrics (Proposal v2.2 §2).

All elasticities — model-side and ground-truth — are defined via the spec's
1% price-perturbation construction at observed test-window prices:

    ε_ij = [Σ_st q_i(p_j·1.01, p_−j) − Σ_st q_i(p)] / (0.01 · Σ_st q*_i(p))

yielding a J×J matrix (rows = affected product i, columns = priced product j;
diagonal = own-price). The denominator always uses the DGP's baseline
quantities q* (spec design choice: isolates price-response accuracy from
demand-level accuracy).

Ground truth per family:

* **log_log** — the DGP is constant-elasticity, so the perturbation response
  is exact in closed form: `q_i(p_j·1.01)/q_i(p) = exp(e_eff·ln 1.01)` with
  `e_eff(i,j) = own_i·1{i=j} + cross[j,i]`, giving
  `ε*_ij = (1.01^e_eff − 1)/0.01` independent of (s, t). A replay would add
  only integer-rounding noise.
* **covariance_probit** — no closed form; the truth is J replays of the
  consumer micro-simulation (one per perturbed product) over the evaluation
  window, using the family's common-RNG replay machinery so the perturbation
  response is exact for the realized consumer population. That generator
  depends on the benchmark's hidden DGP and lives pipeline-side
  (`benchmark_pipeline/metrics/elasticity_truth.py`), not in this package.

Scoring follows the spec's four dimensions: Direction (own sign accuracy;
cross F1 per substitute/complement/unrelated class), Ranking (cross NDCG),
Magnitude (WMAPE, RMSE), Bias (WMPE, mean signed error), with cross-price
metrics stratified by true relationship class. The unrelated-class boundary
is the bottom `unrelated_threshold_pct` (default 20%) percentile of the
classification basis |ε|, reported with the scores (spec note on class
boundaries).

Purchase-incidence amendment (re-ruled 2026-06-11). The scored truth ε* is
the **TOTAL** elasticity — q_i = M(p)·s_i(p), incidence margin included —
because participants model units and any correct units-based model estimates
the total effect. Since ln q_i = ln M + ln s_i, the decomposition
ε_total(i,j) = ε_M(j) + ε_cond(i,j) is exactly additive; the DGP emits the
conditional (fixed-M switching) matrix alongside the total. The
substitute/complement/unrelated **classification runs on the conditional
elasticity** when supplied (pure substitution semantics — under totals, the
common ε_M(j) < 0 shift would turn zero-switching pairs into apparent
complements and make the bottom-percentile band select knife-edge
cancellation pairs). Predictions are classed on ε̂_ij − ε*_M(j) (the true
incidence component netted out — truth-anchored boundaries, same as the
threshold itself); magnitude/bias/ranking are always scored on totals.

D-D1 / M-2.1 (2026-07-04, Option 2 — KEEP AS-IS): the own-price DIAGONAL is
INTENTIONALLY scored on TOTALS here, NOT netted to the conditional/switching
basis; cross classification LABELS use the conditional (incidence-netted)
basis while cross MAGNITUDE also stays on totals. This "category graded on no
axis" property is scoped (M-2.3) to the Layer-3 headline ARENA metric
(``headline_decomposition.py``), NOT to this elasticity-matrix diagnostic,
which is a documented "total-elasticity accuracy" measure. A fully
category-netted Layer 2 (own AND cross magnitude on the conditional basis) is
DEFERRED as its own future decision — ticket ``D-D1-L2-NET``, NOT done. See
decisions.md 2026-07-04 (M-2.2) and clarified §9 "D-D1 Phase-5 micro-decision
batch (2026-07-04)".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PERTURBATION = 0.01


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def elasticity_truth_log_log(
    dgp: pd.DataFrame,
    cross: np.ndarray,
    product_ids: list[str],
    store_sensitivity: np.ndarray | None = None,
) -> pd.DataFrame:
    """Closed-form ε* for the log_log family (rows = affected, cols = priced).

    `store_sensitivity` (D5, symmetry-unfreeze 2026-06-16): an optional length-S
    vector of per-store price-sensitivity multipliers `sens_s` (the
    `loglog_store_price_sensitivity_hidden` artifact). The DGP scales BOTH own and
    cross effects per store by the same `sens_s` in deviation-from-base form, so
    the per-store response is `(1.01^(sens_s·e_eff) − 1)/0.01`. Under that form the
    per-store baseline quantity q* is sens-invariant (the `target_units` anchor
    holds exactly across stores), so the quantity-weighted aggregate in the metric
    definition (`Σ_st q*_i` denominator) reduces to the unweighted store-average:

        ε*_ij = mean_s[ (1.01^(sens_s·e_eff_ij) − 1) / 0.01 ].

    When `None` (or all-ones), this is byte-identical to the single-`e_eff` closed
    form (backward compatible). The integrated matrix stays close to the base
    matrix: the asymptotic Jensen gap (E[sens]=1, 1% perturbation) is only
    ~0.2-0.3% relative, and the dominant deviation on a finite panel is the
    realized store-mean of sens differing from 1 (e.g. ~+2% over the 731-store
    seed-1 draw → ~2% relative on the steepest own entries; max abs ~0.04 on a
    -2.5 own elasticity). The aggregate own-elasticity therefore tracks the
    base ≈ -1.8 anchor (deviation form keeps q* sens-invariant).
    """
    own = dgp.set_index("product_id")["own_elasticity"].reindex(product_ids).to_numpy(dtype=float)
    j = len(product_ids)
    e_eff = np.asarray(cross, dtype=float).T.copy()  # cross[j_priced, i_affected] -> [i, j]
    e_eff[np.arange(j), np.arange(j)] += own
    sens = (
        None if store_sensitivity is None
        else np.asarray(store_sensitivity, dtype=float).reshape(-1)
    )
    if sens is None or sens.size == 0 or np.all(sens == 1.0):
        # Backward-compatible scalar closed form; the all-ones case routes here
        # too so the result is byte-identical to store_sensitivity=None.
        eps = (np.power(1.0 + PERTURBATION, e_eff) - 1.0) / PERTURBATION
    else:
        # Broadcast sens (S,) over e_eff (J,J) -> (S,J,J); store-average.
        scaled = sens[:, None, None] * e_eff[None, :, :]
        per_store = (np.power(1.0 + PERTURBATION, scaled) - 1.0) / PERTURBATION
        eps = per_store.mean(axis=0)
    return pd.DataFrame(eps, index=pd.Index(product_ids, name="affected_product_id"),
                        columns=pd.Index(product_ids, name="priced_product_id"))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _sign(values: np.ndarray) -> np.ndarray:
    return np.sign(values)


def _f1_per_class(true_labels: np.ndarray, pred_labels: np.ndarray, classes: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cls in classes:
        tp = int(np.sum((true_labels == cls) & (pred_labels == cls)))
        fp = int(np.sum((true_labels != cls) & (pred_labels == cls)))
        fn = int(np.sum((true_labels == cls) & (pred_labels != cls)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else (0.0 if (tp + fp + fn) > 0 else None)
        )
        out[cls] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "n_true": int(np.sum(true_labels == cls)),
        }
    return out


def _ndcg(eps_hat: pd.DataFrame, eps_star: pd.DataFrame, k: int) -> float | None:
    """Mean NDCG@k over focal products: rank others by |ε̂_ij|, gains |ε*_ij| (spec 2-2 B)."""
    scores: list[float] = []
    for i in eps_star.index:
        others = [j for j in eps_star.columns if j != i]
        if not others:
            continue
        true_gain = eps_star.loc[i, others].abs()
        pred_rank = eps_hat.loc[i, others].abs().sort_values(ascending=False, kind="mergesort")
        ideal_rank = true_gain.sort_values(ascending=False, kind="mergesort")
        kk = min(k, len(others))
        discounts = 1.0 / np.log2(np.arange(2, kk + 2))
        dcg = float((true_gain.reindex(pred_rank.index[:kk]).to_numpy(dtype=float) * discounts).sum())
        idcg = float((ideal_rank.iloc[:kk].to_numpy(dtype=float) * discounts).sum())
        if idcg > 0:
            scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else None


def _magnitude_bias_block(
    hat: np.ndarray, star: np.ndarray, weights: np.ndarray
) -> dict[str, Any]:
    """WMAPE / RMSE / WMPE / mean signed error over one set of (i, j) entries.

    `weights` carries the focal-product revenue weight w_i replicated per
    entry; the absolute-value denominator |ε*| follows the spec's note on
    signed elasticities.
    """
    if hat.size == 0:
        return {"wmape": None, "rmse": None, "wmpe": None, "mean_signed_error": None, "n_entries": 0}
    denominator = float(np.sum(weights * np.abs(star)))
    return {
        "wmape": (float(np.sum(weights * np.abs(hat - star)) / denominator) if denominator > 0 else None),
        "rmse": float(np.sqrt(np.mean((hat - star) ** 2))),
        "wmpe": (float(np.sum(weights * (hat - star)) / denominator) if denominator > 0 else None),
        "mean_signed_error": float(np.mean(hat - star)),
        "n_entries": int(hat.size),
    }


def elasticity_scores(
    eps_hat: pd.DataFrame,
    eps_star: pd.DataFrame,
    weights: pd.Series,
    *,
    eps_star_conditional: pd.DataFrame | None = None,
    unrelated_threshold_pct: float = 0.20,
    ndcg_k: int | None = None,
) -> dict[str, Any]:
    """Score a J×J estimated elasticity matrix against truth (spec §2).

    All frames are indexed affected_product_id × priced_product_id. `eps_star`
    is the scored truth (TOTAL elasticity under the incidence amendment).
    `eps_star_conditional`, when supplied, is the fixed-M switching matrix:
    class stratification then runs on conditional values, with predictions
    classed on ε̂ − ε*_M(j) (the true incidence component, column-constant,
    derived as total − conditional). Without it, classification falls back to
    the total values (pre-incidence behavior). The submission must cover the
    full matrix; missing entries are reported and treated as 0.0 (the
    no-information value) rather than dropped, so a partial submission cannot
    shrink its own denominator.
    """
    products = list(eps_star.index)
    aligned_hat = eps_hat.reindex(index=eps_star.index, columns=eps_star.columns)
    n_missing = int(aligned_hat.isna().to_numpy().sum())
    aligned_hat = aligned_hat.fillna(0.0)
    w = weights.reindex(eps_star.index).fillna(0.0).astype(float)

    # --- own-price (diagonal) ---
    own_star = np.diag(eps_star.to_numpy(dtype=float))
    own_hat = np.diag(aligned_hat.to_numpy(dtype=float))
    own_w = w.to_numpy(dtype=float)
    own_block = {
        "sign_accuracy": float(np.mean(_sign(own_hat) == _sign(own_star))),
        **_magnitude_bias_block(own_hat, own_star, own_w),
    }

    # --- cross-price (off-diagonal) ---
    j = len(products)
    off_mask = ~np.eye(j, dtype=bool)
    star_off = eps_star.to_numpy(dtype=float)[off_mask]
    hat_off = aligned_hat.to_numpy(dtype=float)[off_mask]
    w_off = np.repeat(own_w, j).reshape(j, j)[off_mask]  # w_i of the affected (focal) product

    if eps_star_conditional is not None:
        aligned_cond = eps_star_conditional.reindex(
            index=eps_star.index, columns=eps_star.columns
        ).to_numpy(dtype=float)
        cond_star_off = aligned_cond[off_mask]
        incidence_off = star_off - cond_star_off  # ε*_M(j), replicated per column
        basis_star = cond_star_off
        basis_hat = hat_off - incidence_off
        classification_basis = "conditional_switching_incidence_netted_out"
    else:
        basis_star = star_off
        basis_hat = hat_off
        classification_basis = "total"

    threshold = (
        float(np.quantile(np.abs(basis_star), unrelated_threshold_pct)) if basis_star.size else 0.0
    )

    def classify(values: np.ndarray) -> np.ndarray:
        labels = np.full(values.shape, "unrelated", dtype=object)
        labels[values > threshold] = "substitute"
        labels[values < -threshold] = "complement"
        return labels

    true_labels = classify(basis_star)
    pred_labels = classify(basis_hat)
    classes = ["substitute", "complement", "unrelated"]

    if ndcg_k is None:
        ndcg_k = j - 1
    cross_block: dict[str, Any] = {
        "f1_per_class": _f1_per_class(true_labels, pred_labels, classes),
        "ndcg": _ndcg(aligned_hat, eps_star, ndcg_k),
        "ndcg_k": int(ndcg_k),
        "ndcg_at_5": _ndcg(aligned_hat, eps_star, min(5, j - 1)) if j > 1 else None,
        "all_pairs": _magnitude_bias_block(hat_off, star_off, w_off),
        "by_true_class": {
            cls: _magnitude_bias_block(
                hat_off[true_labels == cls], star_off[true_labels == cls], w_off[true_labels == cls]
            )
            for cls in classes
        },
        "unrelated_abs_threshold": threshold,
        "unrelated_threshold_pct": unrelated_threshold_pct,
        "classification_basis": classification_basis,
    }
    return {
        "metric": "v2_2_layer2_elasticity_estimation",
        "truth_definition": "total_effect_incidence_plus_switching",
        "n_products": j,
        "n_matrix_entries_missing_in_submission": n_missing,
        "submission_complete": n_missing == 0,
        "own_price": own_block,
        "cross_price": cross_block,
        "weighting": "focal-product observed revenue share over the public training window",
    }
