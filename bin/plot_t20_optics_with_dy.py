"""Optics of the TL20 + T20 arc: beta functions, phase advances and both dispersions.

Same figure as the usual beta / mu / D_x plot, with D_y added to the bottom panel.
The vertical dispersion only exists in the full model (local_tl20 / local_t20), which
keeps the vertical bends and the dipole tilts; the flattened "x" models have D_y = 0
by construction.

s is absolute machine coordinate, measured from the cathode: it comes from local_tl20.tws0.s,
which ocelot carries through the twiss propagation and which also shifts the latdraw machine
panel.  The arc therefore runs 2034.59 m to 2123.33 m.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from latdraw.convert import from_ocelot
from latdraw.plot import subplots_with_lattices
from ocelot.cpbd.beam import Twiss
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.optics import twiss

sys.path.append(str(Path(__file__).resolve().parent))

import local_t20  # noqa: E402
import local_tl20  # noqa: E402

from felarc.lattice import Line  # noqa: E402


def make_lattice() -> tuple[MagneticLattice, Twiss]:
    """TL20 followed by T20, with the TL20 design Twiss at the start.

    local_tl20.tws0 also carries s = 2034.59 m, the position of STSUB.2034.TL from the
    cathode, which puts the whole figure on absolute machine coordinates.
    """
    line = Line(list(local_tl20.cell) + list(local_t20.cell))  # type: ignore[arg-type]
    twiss0 = local_tl20.tws0
    twiss0.Dy = local_t20.tws0.Dy  # non-zero vertical dispersion at injection
    return MagneticLattice(line), twiss0


def plot_optics_with_dy(
    mlat: MagneticLattice,
    twiss0: Twiss | None = None,
    npoints: int = 10000,
):
    llat = from_ocelot(mlat)
    df = twiss(mlat, tws0=twiss0, return_df=True, nPoints=npoints)

    s_offset = twiss0.s if twiss0 is not None else 0.0

    fig, ax = subplots_with_lattices([llat, None, None, None], s_offset=s_offset)
    (_, ax0, ax1, ax2) = ax

    ax0.plot(df.s, df.beta_x, label="$x$")  # type: ignore
    ax0.plot(df.s, df.beta_y, label="$y$")  # type: ignore

    ax1.plot(df.s, df.mux / np.pi, label="$x$")  # type: ignore
    ax1.plot(df.s, df.muy / np.pi, label="$y$")  # type: ignore

    ax2.plot(df.s, df.Dx)  # type: ignore
    ax2.plot(df.s, df.Dy)  # type: ignore
    ax2.axhline(0, alpha=0.2, linestyle="--", color="gray")

    ax0.legend()

    ax0.set_ylabel(r"$\beta$ / m")
    ax1.set_ylabel(r"$\mu\,/\,\pi$")
    ax2.set_ylabel(r"$D$ / m")
    ax2.set_xlabel(r"$s$ from cathode / m")

    return fig, ax


def main() -> None:
    mlat, twiss0 = make_lattice()
    fig, _ = plot_optics_with_dy(mlat, twiss0=twiss0)

    outdir = Path(__file__).resolve().parent
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"t20-optics-with-dy.{suffix}", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
