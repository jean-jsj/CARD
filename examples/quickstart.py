"""Build and score a deliberately naive baseline submission for one cell.

Demonstrates the three submission files end to end (you should beat this easily):

* Layer 1 — per (product, store), predict the mean units over the last 8 training weeks, for every holdout (product, store, week). The training panel records positive-sales rows only, so this simple mean ignores zero-sales weeks — one of many things a real model should do better.
* Layer 2 — the all-zeros J x J elasticity matrix (the no-information value).
* Layer 3 — predicted_delta_units = 0 everywhere ("prices don't matter").

Usage:
    python examples/quickstart.py --cell-dir benchmark/dev/<cell_slug>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product as iproduct
from pathlib import Path

import pandas as pd

LAST_N_TRAIN_WEEKS = 8


def build_submission(cell_dir: Path, out_dir: Path) -> None:
    public = cell_dir / "public"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(public / "transactions_train_public.csv")
    holdout = pd.read_csv(public / "transactions_holdout_context_public.csv")
    products = pd.read_csv(public / "products_public.csv")

    # Layer 1: mean units per (product, store) over the last N training weeks.
    recent = train[train["week"] > train["week"].max() - LAST_N_TRAIN_WEEKS]
    mean_units = (
        recent.groupby(["product_id", "store_id"])["units"].mean().rename("predicted_units")
    )
    l1 = holdout[["product_id", "store_id", "week"]].merge(
        mean_units, on=["product_id", "store_id"], how="left"
    )
    l1["predicted_units"] = l1["predicted_units"].fillna(0.0)
    l1.to_csv(out_dir / "layer1_demand_predictions.csv", index=False)

    # Layer 2: all-zeros elasticity matrix (diagonal included).
    ids = products["product_id"].tolist()
    l2 = pd.DataFrame(
        [(j, i, 0.0) for j, i in iproduct(ids, ids)],
        columns=["priced_product_id", "affected_product_id", "elasticity"],
    )
    l2.to_csv(out_dir / "layer2_elasticities.csv", index=False)

    # Layer 3: zero demand change for every sweep-context row.
    sweep = pd.read_csv(
        public / "counterfactual_sweep_context_public.csv",
        usecols=["intervention_id", "product_id", "store_id", "week"],
    )
    l3 = sweep.drop_duplicates().assign(predicted_delta_units=0.0)
    l3.to_csv(out_dir / "layer3_counterfactual_deltas.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-dir", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="score JSON path")
    args = parser.parse_args()

    cell_dir = args.cell_dir
    if not (cell_dir / "public").is_dir():
        sys.exit(f"{cell_dir} has no public/ directory — run examples/download_data.py first.")
    if not (cell_dir / "hidden").is_dir():
        sys.exit(
            f"{cell_dir} has no hidden/ scoring truth — local scoring works on the dev seed only."
        )

    sub_dir = args.submission_dir or Path("submissions/naive_baseline") / cell_dir.name
    out = args.out or Path("scores") / f"{cell_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    build_submission(cell_dir, sub_dir)
    print(f"naive baseline written to {sub_dir}; scoring...")
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "metrics.evaluate_submission",
            "--cell-dir",
            str(cell_dir),
            "--submission-dir",
            str(sub_dir),
            "--submission-name",
            "naive_baseline",
            "--out",
            str(out),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
