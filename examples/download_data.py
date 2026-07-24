"""Download one benchmark cell from the Hugging Face hub.

Usage:
    pip install huggingface_hub
    python examples/download_data.py --cell complex_log_log_endogenous_seed001

    # ~18 MB starter slice (10 stores) instead of the ~1 GB full cell:
    python examples/download_data.py --cell complex_log_log_endogenous_seed001 --mini

Downloads into benchmark/dev/<cell_slug>/ (or benchmark/dev_mini/<cell_slug>/
with --mini). The dev seed includes hidden/ scoring truth; eval seeds are
public-only. Mini slices exist for the log-log pair only and are for
quickstarts and notebooks — leaderboard scoring uses the full cells.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "jean-jsj/CARD"

CELLS = [
    f"complex_{family}_{endo}_seed001"
    for family in ("log_log", "covariance_probit")
    for endo in ("exogenous", "endogenous")
]
MINI_CELLS = [c for c in CELLS if "log_log" in c]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument(
        "--mini",
        action="store_true",
        help="download the ~18 MB 10-store starter slice instead of the full cell",
    )
    parser.add_argument("--local-dir", type=Path, default=Path("benchmark"))
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub")

    tree = "dev_mini" if args.mini else "dev"
    if args.mini and args.cell not in MINI_CELLS:
        raise SystemExit(f"mini slices exist for {MINI_CELLS} only")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"{tree}/{args.cell}/*"],
        local_dir=args.local_dir,
    )
    print(f"done: {args.local_dir / tree / args.cell}")


if __name__ == "__main__":
    main()
