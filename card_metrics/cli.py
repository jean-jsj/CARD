"""The `card` command: download data, score submissions, build the leaderboard.

    card download --cell complex_log_log_endogenous_seed001 [--mini]
    card score --cell-dir ... --submission-dir ... --submission-name ... --out ...
    card score-all --cells-root ... --submissions-root ... --submission-name ... --out-dir ...
    card leaderboard scores/*.json
    card diagnostics scores/*.json --out diag.csv

`score`, `score-all`, `leaderboard`, and `diagnostics` accept the same options
as their modules (`python -m card_metrics.<name> --help`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "jean-jsj/CARD"
CELLS = [
    f"complex_{family}_{endo}_seed001"
    for family in ("log_log", "covariance_probit")
    for endo in ("exogenous", "endogenous")
]
MINI_CELLS = [c for c in CELLS if "log_log" in c]


def download(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="card download",
                                     description="Fetch one benchmark cell from Hugging Face.")
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument("--mini", action="store_true",
                        help="~18 MB 10-store starter slice instead of the full ~1 GB cell")
    parser.add_argument("--local-dir", type=Path, default=Path("benchmark"))
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub")
    if args.mini and args.cell not in MINI_CELLS:
        raise SystemExit(f"mini slices exist for {MINI_CELLS} only")

    tree = "dev_mini" if args.mini else "dev"
    snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                      allow_patterns=[f"{tree}/{args.cell}/*"], local_dir=args.local_dir)
    print(f"done: {args.local_dir / tree / args.cell}")
    return 0


def main() -> int:
    commands = {"download": download}
    lazy = {"score": "evaluate_submission", "score-all": "evaluate_all",
            "leaderboard": "leaderboard", "diagnostics": "diagnostics"}
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd in commands:
        return commands[cmd](argv)
    if cmd in lazy:
        import importlib

        module = importlib.import_module(f"card_metrics.{lazy[cmd]}")
        sys.argv = [f"card {cmd}"] + argv
        return module.main() or 0
    print(f"unknown command {cmd!r}; run `card --help`", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
