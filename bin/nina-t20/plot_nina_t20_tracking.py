"""Before/after phase space of the STN10 beam through the standard (Nina) TL20 + T20.

Reads the beams written by track_nina_t20.py and draws, one row each:

* the matched input beam at the start of TL20,
* the beam at the end of T20 tracked without CSR,
* the beam at the end of T20 tracked with CSR,

in three phase planes: x-x', y-y' and tau-delta.  Each panel is annotated with the
normalised projected emittance (transverse) or the RMS energy spread (longitudinal).  Rows
whose file is missing are skipped, so this also works while the CSR run is still going.

Usage:
    python plot_nina_t20_tracking.py --charge 250
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from ocelot.common.globals import m_e_GeV
from ocelot.cpbd.io import load_particle_array

HERE = Path(__file__).resolve().parent


def projected_emittance(u: np.ndarray, up: np.ndarray, gamma: float) -> float:
    """Normalised RMS projected emittance, in the units of u * up."""
    cov = np.cov(u, up)
    return gamma * np.sqrt(np.linalg.det(cov))


def load_rows(directory: Path, charge: int):
    candidates = [
        ("input (start of TL20)", f"nina-t20-nocsr-{charge}pC-parray0.npz"),
        ("end of T20, no CSR", f"nina-t20-nocsr-{charge}pC-parray1.npz"),
        ("end of T20, with CSR", f"nina-t20-csr-{charge}pC-parray1.npz"),
    ]
    rows = []
    for label, name in candidates:
        path = directory / name
        if path.exists():
            rows.append((label, load_particle_array(str(path))))
        else:
            print(f"Skipping {label}: {path} not found")
    return rows


def plot_phase_spaces(rows, charge: int, bins: int = 200):
    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=3,
        figsize=(13, 3.4 * len(rows)),
        sharex="col",
        sharey="col",
        layout="constrained",
        squeeze=False,
    )

    for (label, parray), row in zip(rows, axes):
        gamma = parray.E / m_e_GeV
        x, xp = parray.x(), parray.px()
        y, yp = parray.y(), parray.py()
        tau, delta = parray.tau(), parray.p()

        planes = [
            (x * 1e3, xp * 1e6, rf"$\varepsilon_{{n,x}}$ = {projected_emittance(x, xp, gamma) * 1e6:.3f} mm mrad"),
            (y * 1e3, yp * 1e6, rf"$\varepsilon_{{n,y}}$ = {projected_emittance(y, yp, gamma) * 1e6:.3f} mm mrad"),
            (tau * 1e6, delta * 1e3, rf"$\sigma_\delta$ = {delta.std() * 1e3:.4f} $\times 10^{{-3}}$"),
        ]
        for ax, (u, v, text) in zip(row, planes):
            ax.hist2d(u, v, bins=bins, norm=LogNorm(), cmap="viridis", rasterized=True)
            ax.text(
                0.03, 0.95, text, transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )
        row[0].annotate(
            label, xy=(-0.32, 0.5), xycoords="axes fraction", rotation=90,
            va="center", ha="center", fontsize=11,
        )

    bottom = axes[-1]
    bottom[0].set_xlabel(r"$x$ / mm")
    bottom[1].set_xlabel(r"$y$ / mm")
    bottom[2].set_xlabel(r"$\tau$ / µm")
    for row in axes:
        row[0].set_ylabel(r"$x'$ / µrad")
        row[1].set_ylabel(r"$y'$ / µrad")
        row[2].set_ylabel(r"$\delta$ / $10^{-3}$")

    fig.suptitle(f"STN10 {charge} pC through standard TL20 + T20")
    return fig, axes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--charge", type=int, choices=(100, 250), default=250)
    parser.add_argument("--dir", type=Path, default=HERE, help="where the .npz beams are")
    args = parser.parse_args()

    rows = load_rows(args.dir, args.charge)
    if not rows:
        raise SystemExit("No tracked beams found; run track_nina_t20.py first.")
    fig, _ = plot_phase_spaces(rows, args.charge)

    for suffix in ("png", "pdf"):
        fig.savefig(args.dir / f"nina-t20-tracking-{args.charge}pC.{suffix}", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
