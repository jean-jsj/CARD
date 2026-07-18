"""causal-demand-metrics: pure scoring math for the causal-demand benchmark.

Metric definitions for every evaluation layer, packaged standalone so the benchmark pipeline, the participant scoring harness, and third-party tooling share one source of truth: `layer1_demand` (revenue-weighted WMAPE/WMPE on observed holdout sales), `layer2_elasticity` (J×J elasticity scoring plus the closed-form log_log truth), `headline_decomposition` (the Layer-3 headline: signed own-price WMPE plus unsigned competitor substitution WAPE on the category-netted Δq), and `layer4_validity` (label-free causal-coherence checks that read only the submission and public price moves, so they score on real POS data with no hidden truth). The package contains no data-generating code and no hidden-truth generators; it is pure numpy/pandas math over frames the caller supplies. Metric definitions are frozen: a semantic change requires a benchmark version bump, not a patch release.
"""

from __future__ import annotations

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
