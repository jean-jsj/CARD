"""causal-demand-metrics: scoring math for the causal-demand benchmark.

The pure metric definitions for all three evaluation layers, extracted as a
standalone package so the benchmark pipeline, the participant scoring
harness, and any third-party tooling share one source of truth:

- `layer1_demand`         — revenue-weighted WMAPE/WMPE on observed holdout
                            sales (Proposal v2.2 §1).
- `layer2_elasticity`     — J×J elasticity scoring (sign/F1/NDCG/WMAPE/WMPE,
                            class-stratified) + the closed-form log_log truth
                            (Proposal v2.2 §2).
- `headline_decomposition` — the Layer-3 HEADLINE (spec revision notes #9-#11):
                            own-price WMPE + d_total-weighted competitor-only
                            cosine, read off two scenarios (note #10) and
                            aggregated across store-weeks by true-substitution
                            norm (note #11). The full-vector cosine is retired.

- `layer4_validity`        — label-free causal-coherence checks (own-price sign,
                            substitution sign, own-elasticity range coverage,
                            ± sweep monotonicity). Reads only the submission +
                            public price moves, so it scores on REAL POS data
                            with no hidden truth.

The retired full-vector cosine metric was deleted in the Phase-3 re-freeze
(D-RA4, 2026-07-03); the live Layer-3 scoring path is `headline_decomposition`.

This package contains no data-generating code and no hidden-truth
generators; it is pure numpy/pandas math over frames the caller supplies.
Metric definitions are frozen by maintainer ruling (2026-06-11; re-ruled
under the purchase-incidence amendment — R1–R4 — and the headline
decomposition — R5/note #9, 2026-06-14) — semantic changes require a
benchmark version bump, not a patch release.
"""

from __future__ import annotations

# 0.5.0.dev0: note #11 degenerate-store-week weighting (2026-06-15) — the
# headline substitution cosine is averaged across store-weeks weighted by each
# week's true-substitution norm, so signal-free weeks no longer inject noise
# (oracle now scores exactly 1.0). Also retires the full-vector cosine from the
# public API (in-pipeline diagnostic only). (0.4.0.dev0 = note #10 scenario
# split; 0.3.0.dev0 = R5 decomposition; 0.2.0.dev0 = R1-R4 incidence cascade.)
# Still pre-release (0.x); the pip cut + downstream cross-workstream
# verification are tracked in docs/metrics_followups.md (non-blocking for the
# metric implementation itself).
__version__ = "0.5.0.dev0"

from causal_demand_metrics.headline_decomposition import (  # noqa: F401
    decomposed_headline,
    focal_from_context,
)
from causal_demand_metrics.layer1_demand import (  # noqa: F401
    build_demand_truth,
    demand_prediction_scores,
    revenue_weights,
)
from causal_demand_metrics.layer2_elasticity import (  # noqa: F401
    elasticity_scores,
    elasticity_truth_log_log,
)
from causal_demand_metrics.layer4_validity import (  # noqa: F401
    coherence_gate,
    cross_elasticity_plausibility,
    own_elasticity_range_coverage,
    own_price_sign_validity,
    price_direction_from_context,
    substitution_sign_validity,
    sweep_monotonicity,
    validity_scores,
)
