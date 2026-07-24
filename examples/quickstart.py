"""Build and score a deliberately naive submission for one cell.

Forecast = recent average sales, elasticities = 0, demand changes = 0.
You should beat this easily.

Usage:
    python examples/quickstart.py --cell-dir benchmark/dev_mini/<cell_slug>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import product as iproduct
from pathlib import Path

import pandas as pd


def build_submission(cell_dir: Path, out_dir: Path) -> None:
    public = cell_dir / "public"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(public / "transactions_train_public.csv")
    holdout = pd.read_csv(public / "transactions_holdout_context_public.csv")
    products = pd.read_csv(public / "products_public.csv")

    recent = train[train["week"] > train["week"].max() - 8]
    mean_units = recent.groupby(["product_id", "store_id"])["units"].mean().rename("predicted_units")
    forecast = holdout[["product_id", "store_id", "week"]].merge(
        mean_units, on=["product_id", "store_id"], how="left"
    ).fillna({"predicted_units": 0.0})
    forecast.to_csv(out_dir / "forecast_predictions.csv", index=False)

    ids = products["product_id"].tolist()
    pd.DataFrame(
        [(j, i, 0.0) for j, i in iproduct(ids, ids)],
        columns=["priced_product_id", "affected_product_id", "elasticity"],
    ).to_csv(out_dir / "elasticity_matrix.csv", index=False)

    sweep = pd.read_csv(
        public / "counterfactual_sweep_context_public.csv",
        usecols=["intervention_id", "product_id", "store_id", "week"],
    )
    sweep.drop_duplicates().assign(predicted_delta_units=0.0).to_csv(
        out_dir / "counterfactual_deltas.csv", index=False
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-dir", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="score JSON path")
    args = parser.parse_args()

    cell_dir = args.cell_dir
    if not (cell_dir / "public").is_dir():
        sys.exit(f"{cell_dir} has no public/ directory — run `card download` first.")
    if not (cell_dir / "hidden").is_dir():
        sys.exit(f"{cell_dir} has no hidden/ truth — local scoring works on the dev seed only.")

    sub_dir = args.submission_dir or Path("submissions_local/naive") / cell_dir.name
    out = args.out or Path("scores") / f"{cell_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    build_submission(cell_dir, sub_dir)
    code = subprocess.call(
        [sys.executable, "-m", "card_metrics.evaluate_submission",
         "--cell-dir", str(cell_dir), "--submission-dir", str(sub_dir),
         "--submission-name", "naive", "--out", str(out)]
    )
    if code:
        return code

    score = json.loads(out.read_text())
    headline = score["counterfactual_prediction"]["headline"]
    print(f"own-price bias  {headline['own_price']['own_price_wmpe']:+.2f}   (ranked headline; 0 = unbiased)")
    print(f"forecast error   {score['sales_forecasting']['demand_wmape']:.2f}   (displayed, never ranked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
