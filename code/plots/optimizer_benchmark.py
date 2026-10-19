"""Regenerate Fig. S4 (optimizer benchmark on 8x8) from data/.

Training energy of SR, SPRING, MARCH and AdamSR, averaged over non-overlapping 50-iteration bins.
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
    "legend.fontsize": 14,
    "pdf.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "supplement" / "figS04_optimizer_benchmark"
BIN = 50
# E/N is about -0.75; estimates below FLOOR are local-energy outliers (one SR point, E/N = -29.6)
# and are left out of their bin
FLOOR = -50.0 / 64
RUNS = [("SR", "C0"), ("SPRING", "C1"), ("MARCH", "C2"), ("AdamSR", "C3")]


def read_csv(path, dtype=float):
    """Columns of a CSV file with '#' comment lines, as arrays."""
    with open(path) as f:
        names, *rows = csv.reader(line for line in f if not line.startswith("#"))
    return dict(zip(names, np.array(rows, dtype=dtype).T))


def binned(y):
    """Means over non-overlapping BIN-iteration bins without outliers -> (bin centres, means)."""
    n = len(y) // BIN
    y = y[: n * BIN].reshape(n, BIN)
    ok = y > FLOOR
    return (np.arange(n) + 0.5) * BIN, np.where(ok, y, 0.0).sum(1) / ok.sum(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=HERE / "output")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # float32, the precision in which the energies were recorded
    data = read_csv(DATA / "training_curves.csv", dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for label, color in RUNS:
        ax.plot(*binned(data[f"{label}_E_per_site"]), color=color, lw=1.5, label=label)
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"$E/N$")
    ax.set_xlim(0, 5000)
    ax.set_ylim(-0.7475, -0.743)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = args.outdir / "figS04_optimizer_benchmark.pdf"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
