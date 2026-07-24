"""Build the dev_mini/ slice of a dev cell: same market, a handful of stores.

The mini slice exists for fast onboarding (quickstart, notebooks, Colab): it
keeps every product, week, and intervention but only the top-N stores by
training-window volume, cutting a cell from ~1 GB to ~20 MB while remaining
fully scoreable with the standard harness. Store selection is computed on the
FIRST cell listed and reused for the others, so a paired endogeneity on/off
slice shares identical stores and the controlled-pair comparison survives.

Mini-slice numbers are for orientation only; leaderboard entries are scored
on the full cells.

Usage:
    python scripts/make_mini_cell.py \
        --cells-root benchmark/dev --out-root benchmark/dev_mini \
        --cells complex_log_log_endogenous_seed001 complex_log_log_exogenous_seed001 \
        --n-stores 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SLICED_BY_STORE = [
    ("public", "transactions_train_public.csv"),
    ("public", "transactions_holdout_context_public.csv"),
    ("public", "counterfactual_sweep_context_public.csv"),
    ("public", "stores_public.csv"),
    ("hidden", "transactions_full_hidden.csv"),
    ("hidden", "counterfactual_sweep_truth_hidden.csv"),
]
COPIED_WHOLE = [
    ("public", "products_public.csv"),
    ("hidden", "elasticity_truth_hidden.csv"),
    ("release", "scoring_params.json"),
]


def pick_stores(cell_dir: Path, n_stores: int) -> list[str]:
    train = pd.read_csv(cell_dir / "public" / "transactions_train_public.csv",
                        usecols=["store_id", "units"])
    return (
        train.groupby("store_id")["units"].sum().sort_values(ascending=False)
        .head(n_stores).index.tolist()
    )


def slice_cell(cell_dir: Path, out_dir: Path, stores: list[str]) -> None:
    manifest: dict[str, str] = {}
    keep = set(stores)
    for sub, name in SLICED_BY_STORE:
        frame = pd.read_csv(cell_dir / sub / name)
        frame = frame[frame["store_id"].isin(keep)]
        target = out_dir / sub / name
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
        manifest[f"{sub}/{name}"] = hashlib.sha256(target.read_bytes()).hexdigest()
    for sub, name in COPIED_WHOLE:
        source = cell_dir / sub / name
        target = out_dir / sub / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        manifest[f"{sub}/{name}"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (out_dir / "release" / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "release" / "release_notes.md").write_text(
        f"# Mini slice of {cell_dir.name}\n\n"
        f"Top {len(stores)} stores by training-window volume; every product, week, and\n"
        "intervention retained. Built by scripts/make_mini_cell.py from the full dev\n"
        "cell. For quickstarts and notebooks only — leaderboard scoring uses the full\n"
        "cell.\n\n"
        f"Stores: {', '.join(stores)}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--n-stores", type=int, default=10)
    args = parser.parse_args()

    stores = pick_stores(args.cells_root / args.cells[0], args.n_stores)
    print(f"stores (from {args.cells[0]}): {stores}")
    for cell in args.cells:
        out_dir = args.out_root / cell
        slice_cell(args.cells_root / cell, out_dir, stores)
        size_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
        print(f"{cell}: {size_mb:.1f} MB -> {out_dir}")


if __name__ == "__main__":
    main()
