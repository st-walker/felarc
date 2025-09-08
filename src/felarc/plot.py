import latdraw
import matplotlib.pyplot as plt
import numpy as np
from latdraw.interfaces import lattice_from_ocelot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ocelot.cpbd.beam import Twiss
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.optics import twiss

from .analysis import get_curvature


def plot_optics(
        mlat: MagneticLattice,
        tws0: Twiss | None = None, 
) -> tuple[Figure, list[Axes]]:
    llat = lattice_from_ocelot(mlat)
    df = twiss(mlat, tws0, return_df=True, nPoints=10000)

    fig, ax = latdraw.subplots_with_lattices([llat, None, None, None])
    (_, ax0, ax1, ax2) = ax

    ax0.plot(df.s, df.beta_x, label="$x$") # type: ignore
    ax0.plot(df.s, df.beta_y, label="$y$") # type: ignore
    ax1.plot(df.s, df.mux / np.pi) # type: ignore
    ax1.plot(df.s, df.muy / np.pi) # type: ignore
    ax2.plot(df.s, df.Dx)  # type: ignore

    ax0.legend()

    ax0.set_ylabel(r"$\beta$ / m")
    ax1.set_ylabel(r"$\mu\,/\,\pi$")
    ax2.set_ylabel("$D_x$ / m")
    ax2.set_xlabel("$s$ / m")

    return fig, ax


def plot_r56_full(mlat: MagneticLattice, tws0: Twiss | None = None):
    df = twiss(mlat, tws0, return_df=True, nPoints=10000)
    s = df.s  # type: ignore
    dx = df.Dx  # type: ignore
    curvature = get_curvature(mlat.sequence, s)
    integrand = curvature * dx
    llat = lattice_from_ocelot(mlat)
    fig, ax = latdraw.subplots_with_lattices([llat, None, None, None, None])
    (_, ax1, ax2, ax3, ax4) = ax

    ax1.plot(s, dx)
    ax2.plot(s, curvature)
    ax3.plot(s, integrand)
    ax4.axhline(0, alpha=0.2, linestyle="--", color="gray")
    integral = np.cumsum(integrand * np.diff(s)[0])
    ax4.plot(s, integral*1e3, label=r"$\int_0^s \frac{D}{\rho}\mathrm{d}\sigma$")
    _, Rs, _, s = mlat.transfer_maps(energy=14, output_at_each_step=True)  # type: ignore
    ax4.plot(s, [R[4, 5]*1e3 for R in Rs], label="OCELOT")
    ax1.set_ylabel(r"$D_x$ / m")
    ax2.set_ylabel(r"$\rho^{-1}\,/\,\mathrm{m}^{-1}$")
    ax3.set_ylabel(r"$D\rho^{-1}$")
    ax4.set_ylabel(r"$R_{56}$ / mm")
    ax4.set_xlabel("$s$ / m")
    ax4.legend(ncol=2)

    return fig, ax

