"""Download one benchmark cell from the Hugging Face hub.

Usage:
    pip install huggingface_hub
    python examples/download_data.py --cell complex_log_log_endogenous_seed001

Downloads into benchmark/dev/<cell_slug>/ (the dev seed includes hidden/
scoring truth; eval seeds are public-only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "jean-jsj/CARD"

CELLS = [
    f"{complexity}_{family}_{endo}_seed001"
    for complexity in ("simple", "complex")
    for family in ("log_log", "covariance_probit")
    for endo in ("exogenous", "endogenous")
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument("--local-dir", type=Path, default=Path("benchmark"))
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"dev/{args.cell}/*"],
        local_dir=args.local_dir,
    )
    print(f"done: {args.local_dir / 'dev' / args.cell}")


if __name__ == "__main__":
    main()
