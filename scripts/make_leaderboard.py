"""Regenerate the README leaderboard from submitted entries' score files.

Scans submissions/<model>/scores/<cell>.json for every non-reference entry and
rewrites:

  leaderboard/leaderboard_log_log.svg           (paired endo-on / endo-off scatter)
  leaderboard/leaderboard_covariance_probit.svg
  the README.md block between the LEADERBOARD markers (dual-rank table)

Ranking rule: within each demand family, entries are ranked by |own-price
WMPE| in the endogeneity-on cell (ascending; 0 = unbiased). The two family
columns rank independently; the same entry can hold different ranks. L1
forecast WMAPE is reported alongside and never enters the rank.

Participant report mode: --report <model> writes <model>_report.html — the
same two plots with every other entry in gray and <model> highlighted, plus
the table with <model>'s rows marked. All remaining metrics (including the
substitution WAPE and the per-scenario matrix) ship as CSVs via
`python -m metrics.diagnostics`.

Usage:
  python scripts/make_leaderboard.py                 # refresh README + SVGs
  python scripts/make_leaderboard.py --report mymodel --out report.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = [("log_log", "log-log"), ("covariance_probit", "discrete-choice")]
ENDO = {"on": "endogenous", "off": "exogenous"}

ACCENT = "#2a78d6"
GRAY = "#a5a39a"
INK = "#333333"
GRID = "#dddbd4"
START, END = "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->"


def load_entries(subs: Path) -> dict[str, dict]:
    """{model: {family: {endo: {own, l1}}}} for complete non-reference entries."""
    entries: dict[str, dict] = {}
    for d in sorted(subs.iterdir() if subs.is_dir() else []):
        if not d.is_dir() or d.name.startswith("reference_"):
            continue
        entry: dict = {}
        for fam, _ in FAMILIES:
            entry[fam] = {}
            for endo, suffix in ENDO.items():
                p = d / "scores" / f"complex_{fam}_{suffix}_seed001.json"
                if not p.exists():
                    continue
                s = json.loads(p.read_text())
                hl = s["layer3_counterfactual"]["headline"]
                entry[fam][endo] = {
                    "own": hl["own_price"]["own_price_wmpe"],
                    "l1": s["layer1_demand_prediction"]["demand_wmape"],
                }
        if any(entry[fam] for fam, _ in FAMILIES):
            entries[d.name] = entry
    return entries


def scatter_svg(entries: dict, fam: str, fam_label: str, highlight: str | None) -> str:
    """Paired endo-on (top) / endo-off (bottom) scatter for one family."""
    rows = {m: e[fam] for m, e in entries.items() if e.get(fam)}
    W, PH, L, R, T, B, GAP = 640, 240, 56, 16, 30, 40, 26
    H = 2 * PH + GAP

    xs = [v["l1"] for e in rows.values() for v in e.values()]
    ys = [abs(v["own"]) for e in rows.values() for v in e.values()]
    xmin = min(xs + [0.44]) - 0.005 if xs else 0.4
    xmax = max(xs + [0.5]) + 0.005 if xs else 0.55
    ymax = max(ys + [0.25]) * 1.12 if ys else 0.3

    def panel(endo: str, top: float, title: str) -> str:
        X = lambda v: L + (v - xmin) / (xmax - xmin) * (W - L - R)
        Y = lambda v: top + PH - B - v / ymax * (PH - T - B)
        s = f'<text x="{L}" y="{top + 14}" font-size="11" font-weight="600" fill="{INK}">{title}</text>'
        step = round((xmax - xmin) / 5, 2) or 0.01
        t = xmin
        while t <= xmax:
            s += (f'<line x1="{X(t):.0f}" y1="{top + T}" x2="{X(t):.0f}" y2="{top + PH - B}" stroke="{GRID}"/>'
                  f'<text x="{X(t):.0f}" y="{top + PH - B + 14}" font-size="9" text-anchor="middle" fill="{GRAY}">{t:.2f}</text>')
            t = round(t + step, 4)
        s += (f'<line x1="{L}" y1="{Y(0):.0f}" x2="{W - R}" y2="{Y(0):.0f}" '
              f'stroke="{INK}" stroke-width="1.2" stroke-dasharray="4 3"/>'
              f'<text x="{L + 4}" y="{Y(0) - 5:.0f}" font-size="9" fill="{GRAY}">0 = unbiased</text>')
        for m, e in sorted(rows.items()):
            if endo not in e:
                continue
            cx, cy = X(e[endo]["l1"]), Y(abs(e[endo]["own"]))
            mine = highlight is not None and m == highlight
            col = ACCENT if (highlight is None or mine) else GRAY
            r = 7 if mine else 5.5
            s += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{col}" stroke="#ffffff" stroke-width="1.5"/>'
            lab = html.escape(m) + (" (your model)" if mine else "")
            w = "600" if mine else "400"
            s += (f'<text x="{cx + 10:.1f}" y="{cy - 7:.1f}" font-size="10" '
                  f'font-weight="{w}" fill="{INK if (highlight is None or mine) else GRAY}">{lab}</text>')
        return s

    body = panel("on", 0, f"{fam_label} · endogeneity on (ranked)")
    body += panel("off", PH + GAP, f"{fam_label} · endogeneity off (control)")
    if not rows:
        body += (f'<text x="{W / 2}" y="{H / 2}" font-size="13" text-anchor="middle" '
                 f'fill="{GRAY}">no verified entries yet</text>')
    body += (f'<text x="{(L + W - R) / 2}" y="{H - 4}" font-size="10" text-anchor="middle" fill="{INK}">'
             f'L1 forecast WMAPE (lower = better fit); vertical axis: |own-price WMPE| (0 = unbiased)</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H + 16}" '
            f'font-family="Helvetica,Arial,sans-serif"><rect width="100%" height="100%" fill="#ffffff"/>'
            + body + "</svg>")


def rank_table(entries: dict, highlight: str | None) -> str:
    if not entries:
        return "| Model | log-log own WMPE (rank) | L1 WMAPE | discrete-choice own WMPE (rank) | L1 WMAPE |\n|---|---|---|---|---|\n| *no verified entries yet* | | | | |"
    ranks: dict[str, dict[str, int]] = {}
    for fam, _ in FAMILIES:
        scored = [(m, abs(e[fam]["on"]["own"])) for m, e in entries.items() if e.get(fam, {}).get("on")]
        for i, (m, _) in enumerate(sorted(scored, key=lambda t: t[1]), 1):
            ranks.setdefault(m, {})[fam] = i
    lines = ["| Model | log-log own WMPE (rank) | L1 WMAPE | discrete-choice own WMPE (rank) | L1 WMAPE |",
             "|---|---|---|---|---|"]
    order = sorted(entries, key=lambda m: ranks.get(m, {}).get("log_log", 99))
    for m in order:
        cells = [f"**{m}** ← your model" if m == highlight else m]
        for fam, _ in FAMILIES:
            on = entries[m].get(fam, {}).get("on")
            if on:
                cells += [f"{on['own']:+.3f} ({ranks[m][fam]})", f"{on['l1']:.3f}"]
            else:
                cells += ["not submitted", ""]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def update_readme(table: str) -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    if START not in text or END not in text:
        raise SystemExit("README leaderboard markers not found")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(head + START + "\n" + table + "\n" + END + tail)


def write_report(entries: dict, model: str, out: Path) -> None:
    if model not in entries:
        raise SystemExit(f"no scores found for submission {model!r}")
    svgs = "".join(
        f"<h2>{fl} family</h2>" + scatter_svg(entries, fam, fl, model)
        for fam, fl in FAMILIES
    )
    table = rank_table(entries, model)
    t_rows = table.splitlines()[2:]
    html_rows = ""
    for r in t_rows:
        cells = [c.strip() for c in r.strip("|").split("|")]
        mine = "your model" in cells[0]
        style = f' style="background:#e8f0fb"' if mine else ""
        html_rows += f"<tr{style}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    out.write_text(f"""<!doctype html><meta charset="utf-8">
<title>CARD report — {html.escape(model)}</title>
<style>body{{font:15px/1.5 Helvetica,Arial,sans-serif;color:#333;max-width:760px;margin:32px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left}}
h1{{font-size:22px}}h2{{font-size:15px;margin-top:28px}}svg{{max-width:100%}}</style>
<h1>CARD scoring report — {html.escape(model)}</h1>
<p>Ranked by |own-price WMPE| in each family's endogeneity-on cell (0 = unbiased);
the endogeneity-off panel is the control. Your model is highlighted; all other
entries are gray.</p>
{svgs}
<h2>Ranking</h2>
<table><tr><th>Model</th><th>log-log own WMPE (rank)</th><th>L1 WMAPE</th>
<th>discrete-choice own WMPE (rank)</th><th>L1 WMAPE</th></tr>{html_rows}</table>
<p>Every remaining metric, including the substitution WAPE, the full Layer-2
elasticity scorecard, and the per-scenario Layer-3 matrix, is in the CSVs
produced by <code>python -m metrics.diagnostics</code> over your scores.json
files.</p>
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", metavar="MODEL")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    entries = load_entries(ROOT / "submissions")
    if args.report:
        out = args.out or ROOT / f"{args.report}_report.html"
        write_report(entries, args.report, out)
        print(f"report -> {out}")
        return

    lb = ROOT / "leaderboard"
    lb.mkdir(exist_ok=True)
    for fam, fl in FAMILIES:
        (lb / f"leaderboard_{fam}.svg").write_text(scatter_svg(entries, fam, fl, None))
    update_readme(rank_table(entries, None))
    print(f"{len(entries)} entries -> leaderboard/ + README table")


if __name__ == "__main__":
    main()
