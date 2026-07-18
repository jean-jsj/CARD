"""Assemble scored models into a leaderboard table.

Usage:
    python3 -m metrics.leaderboard scores_a.json scores_b.json ... \
        [--out leaderboard.csv] [--format table|markdown|csv] [--aggregate-seeds]

Each input is an `evaluate_submission` output. Rows are ranked (within each
cell-type) by `|own-price WMPE|` ASCENDING — closest-to-zero identification bias
first (the Layer-3 headline own axis). Own-price WMPE (signed, for direction) and
substitution WAPE are reported as the PAIR, both from the single scenario
`sweep_single_share_highest_plus10`; the Layer-1/2 headline numbers ride along as
columns. The README's main arena is the two complex endogeneity-on cell-types.

`--aggregate-seeds` groups scores of the same (model × cell-type) across seeds
and reports `mean ± spread` — the maintainer-side mode for building the README
table from the eval seeds. `--format markdown` emits the README-ready table.
The leaderboard is per cell-type; endogeneity-on cells are the benchmark's
main arena.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SEED_SUFFIX = re.compile(r"_seed\d+$")


def _cell_type(scores: dict[str, Any]) -> str:
    slug = scores.get("cell_slug") or Path(scores.get("cell_dir", "")).name
    return _SEED_SUFFIX.sub("", slug)


def leaderboard_rows(score_payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scores in score_payloads:
        layer3 = scores.get("layer3_counterfactual", {})
        # Redesigned headline (M-0.2, M-1.2): the own-price WMPE (signed) and the
        # substitution WAPE — both from the single scenario
        # `sweep_single_share_highest_plus10` — are reported as the PAIR. Read
        # directly from the new headline dict; the pre-redesign fallbacks are
        # gone.
        headline = layer3.get("headline") or {}
        own_block = headline.get("own_price") or {}
        sub_block = headline.get("substitution") or {}
        own_wmpe = own_block.get("own_price_wmpe")
        sub_wape = sub_block.get("substitution_wape")
        n_store_weeks = sub_block.get("n_store_weeks_scored")
        layer1 = scores.get("layer1_demand_prediction", {})
        layer2 = scores.get("layer2_elasticity_estimation", {})
        own = layer2.get("own_price") or {}
        # Actual-data arm (M-0.1, M-6.2). On synthetic rows the `data_arm` tag is
        # absent → defaults to "synthetic", and the `layer4_validity_actual`
        # block is absent → every `.get` chain below degrades to `None`. The four
        # L4 fractions are a DIAGNOSTIC PANEL reported alongside — NOT combined
        # into one scalar, NOT ranked (M-6.2). L4 leaf keys verified against the
        # FROZEN causal_demand_metrics/layer4_validity.py: `frac_correct_sign`
        # (own_price_sign_validity), `frac_redistribution_mass_correct`
        # (substitution_sign_validity), `frac_in_band` (own_elasticity_range_
        # coverage), `frac_consistent` (sweep_monotonicity).
        data_arm = scores.get("data_arm", "synthetic")
        l4 = scores.get("layer4_validity_actual") or {}
        # Layer 1 runs on BOTH arms (M-0.1 / M-3.1): keep the actual-arm L1 in a
        # SEPARATE column from the synthetic-arm L1 so the two arms never mix in
        # one column. `layer1_demand_prediction` is the same block name / leaf
        # keys on both arms; populate the actual columns ONLY for actual rows.
        is_actual = data_arm == "actual"
        rows.append(
            {
                "submission": scores.get("submission_name"),
                "cell_type": _cell_type(scores),
                "cell_slug": scores.get("cell_slug") or Path(scores.get("cell_dir", "")).name,
                "data_arm": data_arm,
                "layer3_substitution_wape": sub_wape,
                "layer3_own_price_wmpe": own_wmpe,
                "layer3_n_store_weeks": n_store_weeks,
                "layer1_demand_wmape": layer1.get("demand_wmape") if not is_actual else None,
                "layer1_demand_wmpe": layer1.get("demand_wmpe") if not is_actual else None,
                "layer2_own_sign_accuracy": own.get("sign_accuracy"),
                "layer2_own_wmape": own.get("wmape"),
                "layer2_own_wmpe": own.get("wmpe"),
                "layer4_own_sign_frac": (l4.get("own_price_sign") or {}).get("frac_correct_sign"),
                "layer4_substitution_frac": (l4.get("substitution_sign") or {}).get(
                    "frac_redistribution_mass_correct"
                ),
                "layer4_range_in_band": (l4.get("own_elasticity_range") or {}).get("frac_in_band"),
                "layer4_monotonicity_frac": (l4.get("monotonicity") or {}).get("frac_consistent"),
                # #86: the PASS/WARN/FAIL coherence verdict (actual-arm panel headline).
                "layer4_gate_verdict": (l4.get("gate") or {}).get("verdict"),
                "layer1_actual_wmape": layer1.get("demand_wmape") if is_actual else None,
                "layer1_actual_wmpe": layer1.get("demand_wmpe") if is_actual else None,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame):
        # Rank WITHIN each cell-type by |own-price WMPE| ASCENDING
        # (closest-to-zero identification bias first); the reported
        # `layer3_own_price_wmpe` column stays SIGNED (direction). The README's
        # main arena is the two complex endogeneity-on cell types.
        # Rank WITHIN each cell-type by |own-price WMPE| ASCENDING
        # (closest-to-zero identification bias first); the reported
        # `layer3_own_price_wmpe` column stays SIGNED (direction). The README's
        # main arena is the two complex endogeneity-on cell types. The sort
        # `by=`/`ascending=` are EXACTLY as spec 03 set them (M-6.2: do NOT add an
        # actual-arm sort key). Actual rows carry `None` own-WMPE → `na_position=
        # "last"` sinks them to the bottom of their cell-type block; the cumcount
        # groupby is extended to (cell_type, data_arm) so `rank` RESTARTS per arm,
        # keeping the actual arm in its own diagnostic partition — never
        # interleaved into, and never a headline ranking of, the synthetic rows.
        frame["_abs_own_wmpe"] = pd.to_numeric(frame["layer3_own_price_wmpe"], errors="coerce").abs()
        frame = frame.sort_values(
            ["cell_type", "_abs_own_wmpe"], ascending=[True, True], na_position="last"
        ).reset_index(drop=True).drop(columns="_abs_own_wmpe")
        frame.insert(0, "rank", frame.groupby(["cell_type", "data_arm"]).cumcount() + 1)
    return frame


def aggregate_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    """Group per (submission × cell_type) across seeds: mean ± cross-seed spread.

    The maintainer-side mode for the README table: the official score is the
    headline averaged over the eval seeds, with the spread as built-in
    seed-robustness evidence.
    """
    numeric = [
        "layer3_substitution_wape",
        "layer3_own_price_wmpe",
        "layer1_demand_wmape",
        "layer1_demand_wmpe",
        "layer2_own_sign_accuracy",
        "layer2_own_wmape",
        "layer2_own_wmpe",
        # Actual-data arm diagnostics (M-0.1, M-6.2): without these the
        # seed-aggregated table silently DROPS every actual-arm number.
        "layer4_own_sign_frac",
        "layer4_substitution_frac",
        "layer4_range_in_band",
        "layer4_monotonicity_frac",
        "layer1_actual_wmape",
        "layer1_actual_wmpe",
    ]
    # Carry `data_arm` through the grouping so each arm aggregates separately
    # (M-6.2: arms stay in separate partitions, both survive aggregation, M-0.1).
    if "data_arm" not in frame.columns:
        frame = frame.assign(data_arm="synthetic")
    grouped = frame.groupby(["submission", "cell_type", "data_arm"], dropna=False)
    rows = []
    for (submission, cell_type, data_arm), group in grouped:
        row: dict[str, Any] = {
            "submission": submission,
            "cell_type": cell_type,
            "data_arm": data_arm,
            "n_seeds": int(group["cell_slug"].nunique()),
        }
        for col in numeric:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            row[col] = float(values.mean()) if len(values) else None
            row[f"{col}_seed_sd"] = float(values.std(ddof=0)) if len(values) > 1 else None
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        # Same |own-WMPE| ASC direction as `leaderboard_rows` (spec 03); the
        # rank cumcount restarts per (cell_type, data_arm) so the actual arm is
        # its own partition, never interleaved (M-6.2).
        out["_abs_own_wmpe"] = pd.to_numeric(out["layer3_own_price_wmpe"], errors="coerce").abs()
        out = out.sort_values(
            ["cell_type", "_abs_own_wmpe"], ascending=[True, True], na_position="last"
        ).reset_index(drop=True).drop(columns="_abs_own_wmpe")
        out.insert(0, "rank", out.groupby(["cell_type", "data_arm"]).cumcount() + 1)
    return out


def to_markdown(frame: pd.DataFrame) -> str:
    """README-ready markdown: one table, headline first, ± spread when present."""
    display = frame.copy()
    for col in ("layer3_own_price_wmpe", "layer3_substitution_wape"):
        if f"{col}_seed_sd" in display.columns:
            display[col] = [
                (
                    f"{m:.3f} ± {s:.3f}"
                    if pd.notna(m) and pd.notna(s)
                    else (f"{m:.3f}" if pd.notna(m) else "")
                )
                for m, s in zip(display[col], display[f"{col}_seed_sd"])
            ]
    display = display[[c for c in display.columns if not c.endswith("_seed_sd")]]
    keep = [
        c
        for c in [
            "rank", "submission", "cell_type", "cell_slug", "data_arm", "n_seeds",
            "layer3_own_price_wmpe", "layer3_substitution_wape",
            "layer1_demand_wmape", "layer2_own_wmape", "layer2_own_wmpe",
            # Actual-arm diagnostic panel columns (M-0.1, M-6.2): rendered
            # alongside, NOT the sort key.
            "layer1_actual_wmape",
            "layer4_own_sign_frac", "layer4_substitution_frac",
            "layer4_range_in_band", "layer4_monotonicity_frac",
        ]
        if c in display.columns
    ]
    display = display[keep]
    for col in display.columns:
        if display[col].dtype == float:
            display[col] = display[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    header = "| " + " | ".join(display.columns) + " |"
    divider = "|" + "|".join(["---"] * len(display.columns)) + "|"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False))
    return "\n".join([header, divider, body]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--format", choices=["table", "markdown", "csv"], default="table")
    parser.add_argument("--aggregate-seeds", action="store_true")
    args = parser.parse_args()
    payloads = [json.loads(path.read_text()) for path in args.scores]
    table = leaderboard_rows(payloads)
    if args.aggregate_seeds:
        table = aggregate_seeds(table)
    if args.format == "markdown":
        rendered = to_markdown(table)
    elif args.format == "csv":
        rendered = table.to_csv(index=False)
    else:
        rendered = table.to_string(index=False)
    if args.out:
        args.out.write_text(rendered if args.format != "csv" else table.to_csv(index=False), encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
