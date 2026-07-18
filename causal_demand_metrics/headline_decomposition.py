"""Decomposed Layer-3 headline (metrics FINAL decisions, 2026-07-04; IDs M-1.*).

The headline is a PAIR of magnitude-oriented L1 numbers scoring HONEST
COUNTERFACTUAL ACCURACY (not structure recovery). Both are read from ONE +10%
scenario (`sweep_single_share_highest_plus10`, M-1.11) and computed on the
category-NETTED Δq (M-1.7), POOLED across store-weeks as a single
micro-average (M-1.6):

* **own = signed WMPE** on the category-netted FOCAL Δq (M-1.3). Signed because
  the own axis is an IDENTIFICATION / bias test — endogeneity bias is
  directional, and WMPE rewards the unbiasedness IV buys (WMAPE would punish
  IV's variance and could rank a biased low-variance model above an unbiased
  one). Discrimination (naive vs sophisticated vs oracle) is carried by THIS
  axis (M-1.8).

* **substitution = unsigned WAPE** (L1 / total variation) on the category-netted
  COMPETITOR Δq, over ALL non-focal products, on RAW un-normalized Δq mass,
  geometry-BLIND (M-1.4, M-1.5). No direction score, no proximity weights, no
  renormalization to sum-to-1. This axis is honest accuracy and is NOT required
  to separate naive from sophisticated (M-1.8).

Netting (M-1.7): each side's residual is `Δq − ΔM·s_base` with `ΔM = Σ_j Δq`
the row-sum of THAT side's own Δq and `s_base` the baseline-unit share. This
makes both axes independent of the category/incidence magnitude we do not grade
(a no-change or pure-proportional-contraction prediction nets to 0; a
right-substitution-wrong-category prediction nets to the true substitution). The
focal is INCLUDED in the ΔM row-sum.

The old direction-of-substitution build (geometry-weighted, per-store-week, then
averaged) is DELETED (M-1.2); the substitution-GEOMETRY / multimodal claim lives
in the SEPARATE E1 diversion panel (M-1.9, deferred), NOT here. This module now
has NO notion of geometry / proximity.

Pure math: the caller supplies the truth frame, the prediction deltas, and the
focal product. No file I/O.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def focal_from_context(context: pd.DataFrame, intervention_id: str) -> str | None:
    """The product whose price moves under `intervention_id` (public context)."""
    rows = context[context["intervention_id"].astype(str) == str(intervention_id)].copy()
    if rows.empty or "intervention_price" not in rows or "baseline_price" not in rows:
        return None
    moved = rows[
        ~np.isclose(
            rows["intervention_price"].astype(float), rows["baseline_price"].astype(float)
        )
    ]
    if moved.empty:
        return None
    return str(moved["product_id"].astype(str).iloc[0])


def decomposed_headline(frame: pd.DataFrame, focal: str) -> dict[str, Any]:
    """Pooled own-price WMPE + competitor substitution WAPE for one intervention.

    `frame` carries one intervention's rows with columns
    ``product_id, store_id, week, baseline_units, dq_true, dq_pred``
    (``dq_true`` = true_counterfactual − baseline; ``dq_pred`` = submitted
    predicted_delta_units, 0 where omitted).

    Returns a PAIR of magnitude-oriented L1 numbers (M-1.1, M-1.3, M-1.4):

    * ``own_price_wmpe`` — SIGNED WMPE on the category-netted focal Δq, pooled
      over store-weeks where the focal is present. ``None`` if the pooled
      denominator ``Σ_sw |rt_f|`` is 0 (M-1.3, M-1.6).
    * ``substitution_wape`` — UNSIGNED WAPE (raw-mass L1) on the category-netted
      competitor Δq, geometry-blind, pooled over all store-weeks. ``None`` if the
      pooled denominator ``Σ_sw Σ_{k≠f} |rt_k|`` is 0 (M-1.4, M-1.5, M-1.6).

    Both on category-netted residuals ``Δq − ΔM·s_base`` (M-1.7). Numerators and
    denominators are pooled and divided ONCE (micro-average, M-1.6); the four
    ``*_sum`` accumulators are emitted so cross-cell / cross-scenario re-pooling
    is a sum-of-numerators ÷ sum-of-denominators.
    """
    work = frame.copy()
    work["product_id"] = work["product_id"].astype(str)
    for col in ("baseline_units", "dq_true", "dq_pred"):
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    own_signed_sum = 0.0
    own_abs_true_sum = 0.0
    sub_abs_err_sum = 0.0
    sub_abs_true_sum = 0.0
    n_store_weeks_scored = 0
    n_focal_missing = 0

    for (sw_store, sw_week), group in work.groupby(["store_id", "week"], sort=True):
        base = group["baseline_units"].to_numpy(float)
        base_total = float(base.sum())
        share = base / base_total if base_total > 0 else np.zeros_like(base)
        dq_true_all = group["dq_true"].to_numpy(float)
        dq_pred_all = group["dq_pred"].to_numpy(float)
        # Category-netted residuals: Δq − ΔM·s_base, each side by its own ΔM
        # (= Σ_j Δq, focal INCLUDED). Independent of the category/incidence
        # magnitude we do not grade (M-1.7).
        resid_true = dq_true_all - float(dq_true_all.sum()) * share
        resid_pred = dq_pred_all - float(dq_pred_all.sum()) * share
        is_focal = (group["product_id"].to_numpy() == str(focal))

        # own axis — pool the SIGNED netted focal residual (M-1.3, M-1.6).
        if is_focal.any():
            rt_f = float(resid_true[is_focal][0])
            rp_f = float(resid_pred[is_focal][0])
            own_signed_sum += rp_f - rt_f
            own_abs_true_sum += abs(rt_f)
        else:
            n_focal_missing += 1

        # substitution axis — pool the ABSOLUTE netted competitor error, raw
        # mass, geometry-blind (M-1.4, M-1.5, M-1.6). No direction score, no
        # proximity weights, no per-store-week ratio, no renormalization.
        is_comp = ~is_focal
        if is_comp.any():
            err = np.abs(resid_pred[is_comp] - resid_true[is_comp])  # L1 / total variation
            sub_abs_err_sum += float(err.sum())
            sub_abs_true_sum += float(np.abs(resid_true[is_comp]).sum())
            n_store_weeks_scored += 1

    own_price_wmpe = (own_signed_sum / own_abs_true_sum) if own_abs_true_sum > 0 else None
    substitution_wape = (sub_abs_err_sum / sub_abs_true_sum) if sub_abs_true_sum > 0 else None

    return {
        "own_price_wmpe": (float(own_price_wmpe) if own_price_wmpe is not None else None),
        "substitution_wape": (float(substitution_wape) if substitution_wape is not None else None),
        "n_store_weeks_scored": int(n_store_weeks_scored),
        "n_store_weeks_focal_missing": int(n_focal_missing),
        "own_signed_sum": float(own_signed_sum),
        "own_abs_true_sum": float(own_abs_true_sum),
        "sub_abs_err_sum": float(sub_abs_err_sum),
        "sub_abs_true_sum": float(sub_abs_true_sum),
        "focal_product_id": str(focal),
    }
