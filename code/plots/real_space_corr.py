"""Regenerate the real-space correlation figures S12-S21 from data/.

Each figure shows C_s, 50 C_c and 100 C_p along r = (r, 0) and r = (0, r); markers outside the
fixed y window of each t' are omitted. Select figures by number, e.g. S12 S21 (default: all).
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "legend.fontsize": 15,
    "pdf.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "supplement" / "real_space_corr"
# key, scale, label, color, marker, size, zorder, first r (C_p pairs share sites below r = 2)
CURVES = [
    ("Cs", 1.0, r"$C_s(r)$", "#e34948", "o", 7, 4, 0),
    ("Cc", 50.0, r"$C_c(r) \times 50$", "#222222", "s", 7, 4, 0),
    ("Cp", 100.0, r"$C_p(r) \times 100$", "#6250d6", "D", 8, 3, 2),
]


def read_csv(path):
    """Columns of a CSV file with '#' comment lines, as float arrays."""
    with open(path) as f:
        names, *rows = csv.reader(line for line in f if not line.startswith("#"))
    return dict(zip(names, np.array(rows, dtype=float).T))


def draw(folder, outdir):
    cuts = read_csv(folder / "cuts.csv")
    R = cuts["r"].astype(int)
    half = R[-1]  # L / 2
    legend = folder.name.startswith("figS12")  # only the first figure carries the legend
    frac = 0.88 if legend else 1  # height fraction left for the panels
    ywin = 0.07 if "_tpm02_" in folder.name else 0.22
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8 * 0.88 / frac), sharey=True)
    titles = (r"(a) $\mathbf{r} = (r, 0)$", r"(b) $\mathbf{r} = (0, r)$")
    for ax, d, title in zip(axes, "xy", titles):
        ax.axhline(0, color="#b0b0b0", lw=0.8, zorder=0)
        for key, scale, label, color, marker, ms, z, r0 in CURVES:
            v = scale * cuts[f"{key}_{d}"][r0:]
            if np.isnan(v).all():  # no C_p for this state
                continue
            e = scale * cuts[f"{key}_{d}_err"][r0:]
            show = np.abs(v) <= 0.97 * ywin
            ax.plot(R[r0:], v, color=color, lw=1.2, alpha=0.5, zorder=1)
            ax.errorbar(R[r0:][show], v[show], e[show], ls="none", marker=marker, ms=ms, capsize=2,
                        color=color, mew=1.5, zorder=z, label=label)
        ax.set_ylim(-ywin, ywin)
        ax.set_xlabel(r"$r$")
        ax.set_xlim(-0.5, half + 0.5)
        ax.set_xticks(range(0, half + 1, 2 if half <= 8 else 4))
        ax.grid(axis="x", color="#e0e0e0", lw=0.6)
        ax.set_title(title, loc="left", fontsize=16)
    axes[0].set_ylabel("Scaled correlation")
    if legend:
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 1.0), columnspacing=2.5)
    fig.tight_layout(rect=(0, 0, 1, frac))
    out = outdir / f"{folder.name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("figures", nargs="*", help="figure numbers, e.g. S12 S21 (default: all)")
    parser.add_argument("--outdir", type=Path, default=HERE / "output")
    args = parser.parse_args()

    folders = sorted(DATA.glob("figS*_real_space_corr_*"))
    if args.figures:
        wanted = {s.upper().removeprefix("FIG").removeprefix("S") for s in args.figures}
        folders = [f for f in folders if f.name.split("_")[0][4:] in wanted]
    if not folders:
        raise SystemExit(f"no matching data folders under {DATA}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    for folder in folders:
        print("wrote", draw(folder, args.outdir))


if __name__ == "__main__":
    main()
