"""Render README figures from saved results.

    python -m qailab qec --distance 3 --p 0.03 --eval-p 0.01 0.02 0.05
    python scripts/make_figures.py

Writes light and dark PNGs to docs/figures/ (GitHub picks one per the viewer's theme).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926"),
}


def qec_figure(result: dict, theme: str, path: Path) -> None:
    t = THEMES[theme]
    evals = sorted(result["evaluations"], key=lambda e: e["p"])
    p = [100 * e["p"] for e in evals]
    series = {k: [100 * e[k]["logical_error_rate"] for e in evals] for k in ("no_decoding", "pymatching", "neural")}
    trained_p = 100 * result["config"]["p"]

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    # reference line: what you get with no correction at all
    ax.plot(p, series["no_decoding"], color=t["muted"], lw=1.5, ls=(0, (4, 3)), zorder=2, label="No decoding")
    # the two decoders nearly coincide: PyMatching as open rings drawn over the transformer's diamonds
    ax.plot(p, series["neural"], color=t["s2"], lw=2, marker="D", ms=6, mec=t["surface"], mew=1.5,
            zorder=3, label="Transformer (neural)")
    ax.plot(p, series["pymatching"], color=t["s1"], lw=2, marker="o", ms=11, mfc="none", mec=t["s1"], mew=2,
            zorder=4, label="PyMatching (matching)")

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xticks(p)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xlim(min(p) - 0.3, max(p) + 1.15)
    ax.set_ylim(0.7, 45)

    ax.grid(axis="y", color=t["grid"], lw=0.8, zorder=0)
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    ax.tick_params(which="minor", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.set_xlabel("Physical error rate per qubit per round", color=t["ink2"], fontsize=9.5, labelpad=8)
    ax.set_ylabel("Logical error rate (log scale)", color=t["ink2"], fontsize=9.5, labelpad=8)

    # mark the training point
    ax.axvline(trained_p, color=t["axis"], lw=1, zorder=1)
    ax.text(trained_p + 0.06, 0.78, "trained here", color=t["muted"], fontsize=8.5, va="bottom")

    # direct labels just past the line ends; the two decoders end together, so split them vertically
    xe = p[-1]
    for key, text, dy in (("no_decoding", "No decoding", 0), ("pymatching", "PyMatching", 6),
                          ("neural", "Transformer", -6)):
        ax.annotate(text, (xe, series[key][-1]), xytext=(12, dy), textcoords="offset points",
                    ha="left", va="center", color=t["ink2"], fontsize=9)

    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(n) for n in ("No decoding", "PyMatching (matching)", "Transformer (neural)")]
    leg = ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", frameon=False,
                    fontsize=8.5, labelcolor=t["ink2"], handlelength=2.2)
    for h in leg.legend_handles:
        h.set_alpha(1)

    d, r = result["config"]["distance"], result["rounds"]
    fig.text(0.075, 0.955, "A neural decoder matches matching-based decoding", color=t["ink"],
             fontsize=12.5, fontweight="semibold", ha="left")
    fig.text(0.075, 0.905, f"Surface code, distance {d}, {r} rounds, phenomenological noise. "
             f"{evals[0]['shots']:,} test shots per point; lower is better.",
             color=t["ink2"], fontsize=9, ha="left")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.84, bottom=0.14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


# Severity uses the status palette (never series colors); every bar is also labeled in words.
RISK_ROWS = [
    ("critical", "Critical", "#d03b3b"),
    ("high", "High", "#ec835a"),
    ("medium", "Medium", "#fab219"),
    ("low", "Low", "#898781"),
    ("info", "Already post-quantum", "#0ca30c"),
]


def _risk_examples(findings: list[dict]) -> dict[str, str]:
    """A few words on what sits in each risk level, from the findings themselves."""
    out = {}
    for level, *_ in RISK_ROWS:
        fams = []
        for f in findings:
            if f["risk"] != level:
                continue
            name = f["family"] + (f"-{f['key_size']}" if level == "critical" and f["key_size"] else "")
            name = {"PrivateKey": "committed private key", "TLS-classical": "classical TLS",
                    "PQC": "ML-KEM"}.get(name, name)
            if f["rule_id"] == "generic.jwt-alg":
                name = "JWT signing (RS256)"
            if name not in fams:
                fams.append(name)
        out[level] = ", ".join(fams[:5])
    return out


def pqc_figure(scan: dict, theme: str, path: Path) -> None:
    t = THEMES[theme]
    fs = scan["findings"]
    counts = {lvl: sum(f["risk"] == lvl for f in fs) for lvl, *_ in RISK_ROWS}
    examples = _risk_examples(fs)
    vulnerable = sum(f["quantum_vulnerable"] for f in fs)

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ys = list(range(len(RISK_ROWS)))[::-1]
    for y, (lvl, label, color) in zip(ys, RISK_ROWS):
        n = counts[lvl]
        ax.barh(y, n, height=0.62, color=color, edgecolor=t["surface"], linewidth=2, zorder=2)
        ax.text(n + 0.25, y, f"{n}", va="center", ha="left", color=t["ink"], fontsize=10, fontweight="bold")
        ax.text(n + 0.25 + 0.28 * len(str(n)) + 0.35, y, examples[lvl], va="center", ha="left",
                color=t["ink2"], fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label, _ in RISK_ROWS], color=t["ink2"], fontsize=9.5)
    ax.set_xlim(0, max(counts.values()) * 2.05)
    ax.set_xticks([])
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(t["axis"])

    fig.text(0.03, 0.93, f"{vulnerable} of {len(fs)} crypto uses in the demo app are quantum-vulnerable",
             color=t["ink"], fontsize=12.5, fontweight="bold", ha="left")
    fig.text(0.03, 0.865, "Findings by risk level from the post-quantum scanner, with what each level holds",
             color=t["ink2"], fontsize=9, ha="left")
    fig.subplots_adjust(left=0.25, right=0.98, top=0.8, bottom=0.05)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def cover_figure(path: Path) -> None:
    """1920x1080 cover for a LinkedIn article."""
    t = THEMES["dark"]
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"]})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("#0d0d0d")
    fig.text(0.07, 0.80, "QUBIT SENTINEL", color=t["s1"], fontsize=30, fontweight="bold")
    fig.text(0.07, 0.66, "Three AI experiments\nfor the quantum era", color=t["ink"], fontsize=64,
             fontweight="bold", va="top", linespacing=1.1)
    cards = [
        ("01", "Error correction", "A transformer decoder matches\nPyMatching on the surface code"),
        ("02", "Noise prediction", "Where a simple rule of thumb\nties machine learning"),
        ("03", "Crypto scanner", "13 of 18 crypto uses in a\ndemo app are quantum-vulnerable"),
    ]
    for i, (num, head, body) in enumerate(cards):
        x = 0.07 + i * 0.295
        fig.patches.append(plt.Rectangle((x, 0.10), 0.27, 0.22, transform=fig.transFigure,
                                         facecolor=t["surface"], edgecolor="#2c2c2a", lw=2))
        fig.text(x + 0.02, 0.275, num, color=t["s1"], fontsize=22, fontweight="bold", va="top")
        fig.text(x + 0.065, 0.275, head, color=t["ink"], fontsize=24, fontweight="bold", va="top")
        fig.text(x + 0.02, 0.205, body, color=t["ink2"], fontsize=19, va="top", linespacing=1.35)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    runs = sorted((ROOT / "results" / "qec").glob("*.json"), key=lambda q: q.stat().st_mtime)
    if not runs:
        sys.exit("No QEC results yet: run `python -m qailab qec ...` first.")
    result = json.loads(runs[-1].read_text())
    for theme in THEMES:
        out = OUT / f"qec_decoder_{theme}.png"
        qec_figure(result, theme, out)
        print(f"wrote {out}")
    scan_path = ROOT / "results" / "pqc" / "findings.json"
    if scan_path.exists():
        scan = json.loads(scan_path.read_text())
        for theme in THEMES:
            out = OUT / f"pqc_risk_{theme}.png"
            pqc_figure(scan, theme, out)
            print(f"wrote {out}")
    cover_figure(OUT / "cover.png")
    print(f"wrote {OUT / 'cover.png'}")


if __name__ == "__main__":
    main()
