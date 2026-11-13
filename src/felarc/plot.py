import latdraw
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from latdraw.convert import from_ocelot
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ocelot.cpbd.beam import Twiss
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.optics import twiss
from latdraw.plot import subplots_with_lattices

from felarc.analysis import (calculate_csr_dispersions,
                             get_principal_trajectories)
from felarc.csr import (LinearCSRTracker, gaussian_bunch_emittance_growth,
                        gaussian_bunch_espread_per_unit_length)
from felarc.optics import get_curvature, transfer_maps_z
from felarc.tba import (
    achromatic_m12_m22,
    achromatic_periodic_twiss,
    check_achromat_condition,
    isochronous_m21,
    csr_cancellation_m21s,
    csr_cancellation_m21s_old
)

from .constants import m_e_GeV


def plot_optics(
    mlat: MagneticLattice,
    twiss0: Twiss | None = None,
) -> tuple[Figure, list[Axes]]:
    llat = from_ocelot(mlat)
    df = twiss(mlat, tws0=twiss0, return_df=True, nPoints=10000)

    s_offset = 0
    if twiss0 is not None:
        s_offset = twiss0.s

    fig, ax = subplots_with_lattices(
        [llat, None, None, None], s_offset=s_offset
    )
    (_, ax0, ax1, ax2) = ax

    ax0.plot(df.s, df.beta_x, label="$x$")  # type: ignore
    ax0.plot(df.s, df.beta_y, label="$y$")  # type: ignore
    ax1.plot(df.s, df.mux / np.pi)  # type: ignore
    # ax1.plot(df.s, df.muy / np.pi) # type: ignore
    ax2.plot(df.s, df.Dx)  # type: ignore
    # ax2.plot(df.s, df.Dy) # type: ignore
    ax2.axhline(0, alpha=0.2, linestyle="--", color="gray")

    ax0.legend()

    ax0.set_ylabel(r"$\beta$ / m")
    ax1.set_ylabel(r"$\mu\,/\,\pi$")
    ax2.set_ylabel("$D$ / m")
    ax2.set_xlabel("$s$ / m")

    return fig, ax


def plot_r56_full(mlat: MagneticLattice, twiss0: Twiss | None = None):
    df = twiss(mlat, twiss0, return_df=True, nPoints=10000)

    s_offset = 0.0
    if twiss0 is not None:
        s_offset=  twiss0.s

    s = df.s  # type: ignore
    dx = df.Dx  # type: ignore
    curvature = get_curvature(mlat.sequence, s - s[0])
    integrand = curvature * dx
    llat = from_ocelot(mlat)
    fig, ax = subplots_with_lattices([llat, None, None, None, None], s_offset=s_offset)
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
    ax3.set_ylabel(r"$D_x\rho^{-1}$")
    ax4.set_ylabel(r"$R_{56}$ / mm")
    ax4.set_xlabel("$s$ / m")
    ax4.legend(ncol=2)

    return fig, ax


def plot_r51_full(mlat: MagneticLattice):
    principles = get_principal_trajectories(mlat, num=10000)

    s = principles.s
    cx = principles.cx
    curvature = get_curvature(mlat.sequence, principles.s)

    integrand = curvature * cx

    llat = from_ocelot(mlat)
    fig, ax = subplots_with_lattices([llat, None, None, None, None])
    (_, ax1, ax2, ax3, ax4) = ax

    ax1.plot(s, cx)
    ax2.plot(s, curvature)
    ax3.plot(s, integrand)
    ax4.axhline(0, alpha=0.2, linestyle="--", color="gray")
    integral = np.cumsum(integrand * np.diff(s)[0])
    ax4.plot(s, integral, label=r"$\int_0^s \frac{C}{\rho}\mathrm{d}\sigma$")
    _, Rs, _, s = mlat.transfer_maps(energy=14, output_at_each_step=True)  # type: ignore
    ax4.plot(s, [R[4, 0] for R in Rs], label="OCELOT")
    ax1.set_ylabel(r"$C_x$ / m")
    ax2.set_ylabel(r"$\rho^{-1}\,/\,\mathrm{m}^{-1}$")
    ax3.set_ylabel(r"$C_x\rho^{-1}$")
    ax4.set_ylabel(r"$R_{51}$")
    ax4.set_xlabel("$s$ / m")
    ax4.legend(ncol=2)

    return fig, ax

def plot_r52_full(mlat: MagneticLattice):
    principles = get_principal_trajectories(mlat, num=10000)

    s = principles.s
    sx = principles.sx
    curvature = get_curvature(mlat.sequence, principles.s)

    integrand = curvature * sx

    llat = from_ocelot(mlat)
    fig, ax = subplots_with_lattices([llat, None, None, None, None])
    (_, ax1, ax2, ax3, ax4) = ax

    ax1.plot(s, sx)
    ax2.plot(s, curvature)
    ax3.plot(s, integrand)
    ax4.axhline(0, alpha=0.2, linestyle="--", color="gray")
    integral = np.cumsum(integrand * np.diff(s)[0])
    ax4.plot(s, integral, label=r"$\int_0^s \frac{S}{\rho}\mathrm{d}\sigma$")
    _, Rs, _, s = mlat.transfer_maps(energy=14, output_at_each_step=True)  # type: ignore
    ax4.plot(s, [R[4, 1] for R in Rs], label="OCELOT")
    ax1.set_ylabel(r"$S_x$ / m")
    ax2.set_ylabel(r"$\rho^{-1}\,/\,\mathrm{m}^{-1}$")
    ax3.set_ylabel(r"$S_x\rho^{-1}$")
    ax4.set_ylabel(r"$R_{52}\,/\,\mathrm{m\,rad^{-1}}$")
    ax4.set_xlabel("$s$ / m")
    ax4.legend(ncol=2)

    return fig, ax


def plot_principal_trajectories(mlat: MagneticLattice, num: int):
    traj = get_principal_trajectories(mlat, num)

    llat = from_ocelot(mlat)
    fig, ax = subplots_with_lattices([llat, None, None, None])
    (_, ax1, ax2, ax3) = ax

    ax1.set_ylabel("C")
    ax2.set_ylabel("S")
    ax3.set_ylabel("D")
    ax3.set_xlabel("$s$ / m")

    ax1.plot(traj.s, traj.cx, label="$x$")
    ax2.plot(traj.s, traj.sx)
    ax3.plot(traj.s, traj.dx)
    ax1.plot(traj.s, traj.cy, label="$y$")
    ax2.plot(traj.s, traj.sy)
    ax3.plot(traj.s, traj.dy)

    ax1.set_ylabel("$C$")
    ax2.set_ylabel("$S$")
    ax3.set_ylabel("$D$")
    ax3.set_xlabel("$s$ / m")

    ax1.legend()

    return fig, ax


def plot_time_coupling_terms(mlat: MagneticLattice, num: int, s_offset=0.0):
    principles = get_principal_trajectories(mlat, num=100000)

    s = principles.s
    cx = principles.cx
    sx = principles.sx
    dx = principles.dx

    curvature = get_curvature(mlat.sequence, principles.s)

    r51_integrand = curvature * cx
    r52_integrand = curvature * sx
    r56_integrand = curvature * dx

    r51 = np.cumsum(r51_integrand)
    r52 = np.cumsum(r52_integrand)
    r56 = np.cumsum(r56_integrand)

    llat = from_ocelot(mlat)
    fig, ax = subplots_with_lattices([llat, None, None, None], s_offset=s_offset)
    (_, ax1, ax2, ax3) = ax

    ax1.plot(s+s_offset, r51_integrand)
    ax2.plot(s+s_offset, r52_integrand)
    ax3.plot(s+s_offset, r56_integrand)

    # ax4.axhline(0, alpha=0.2, linestyle="--", color="gray")
    # integral = np.cumsum(integrand * np.diff(s)[0])
    # ax4.plot(s, integral, label=r"$\int_0^s \frac{S}{\rho}\mathrm{d}\sigma$")
    # _, Rs, _, s = mlat.transfer_maps(energy=14, output_at_each_step=True)  # type: ignore
    # ax4.plot(s, [R[4, 1] for R in Rs], label="OCELOT")
    # ax1.set_ylabel(r"$S_x$ / m")
    # ax2.set_ylabel(r"$\rho^{-1}\,/\,\mathrm{m}^{-1}$")
    # ax3.set_ylabel(r"$S_x\rho^{-1}$")
    # ax4.set_ylabel(r"$R_{52}\,/\,\mathrm{m\,rad^{-1}}$")
    # ax4.set_xlabel("$s$ / m")
    # ax4.legend(ncol=2)

    return fig, ax

def plot_csr_dispersion(mlat: MagneticLattice, twiss0: Twiss | None = None):
    llat = from_ocelot(mlat)
    fig, ax = subplots_with_lattices([llat, None, None, None, None])
    (_, ax1, ax2, ax3, ax4) = ax

    # df = twiss(mlat, twiss0, return_df=True# , nPoints=10000
    #            )

    # curvature = get_curvature(mlat.sequence, df.s)

    # csr_disp = calculate_csr_dispersions(mlat)

    # We want to calculate the dispersion and the so-called "CSR dispersion".
    # So dispersion is where k=0 (i.e. no CSR) and dp0 = 1.
    # CSR Dsipersion is where k=1 (i.e. 1 unit of CSR) and dp0 = 0.

    csr_tracker = LinearCSRTracker(mlat, k=1)
    s, coords = csr_tracker.track(0, 0, 0, 0, ds=0.01)

    dx_tracker = LinearCSRTracker(mlat, k=0)
    dx_s, dx_coords = dx_tracker.track(0, 0, 1, 0, ds=0.01) #0.0001)

    # ax1.plot(df.s, df.Dx, marker=".", label="OCELOT")
    # ax2.plot(df.s, df.Dxp)

    # from IPython import embed; embed()

    # bunch_length = 7e6
    # bunch_charge = 500e-12


    ax1.plot(dx_s, dx_coords.x# , label="FELARC"# , marker="x"
             )
    ax2.plot(dx_s, dx_coords.px)
    ax3.plot(s, coords.x)
    ax4.plot(s, coords.px)
    # ax5.plot(s, coords.dp_csr)
    # ax[3].plot(df.s, df.Dx)
    # ax1.legend()

    # # csr_wake = calculate_linear_csr_wake(bunch_charge, bunch_length, curvature)

    # ax[-2].plot(df.s, csr_wake)
    ax1.set_ylabel(r"$D$ / m")
    ax2.set_ylabel(r"$D'$ / rad")
    ax3.set_ylabel(r"$\xi$ / m")
    ax4.set_ylabel(r"$\xi'$ / rad")

    # ax1.plot


    ax[-1].set_xlabel("$s$ / m")

    return fig, ax

def plot_m11_m21_tba_contours(
    tba,
    m11_range=(-3.0, 3.0),
    m21_range=(-15.0, 25.0),
    legend_loc="upper right",
    opt=None,
):
    m11 = np.linspace(*m11_range, num=1000)
    m21 = np.linspace(*m21_range, num=1000)

    m11_grid, m21_grid = np.meshgrid(m11, m21)

    achromatic_m12_grid, achromatic_m22_grid = achromatic_m12_m22(
        tba, m11=m11_grid, m21=m21_grid
    )

    Rd1, Rb1, Rm, Rb2, Rmflipped, Rb3, Rd2 = tba.component_matrices()

    # m11 and m21

    # We have 4 grids now, m11 and m21 are free variables.  The other 2 matrix entries,
    # m12 and m22 are constrained in order to ensure achromatism.
    # So for each pair of m11 and m21, we have also an m12 and m22.  We therefore need a matrix

    Rm_grid = np.empty((*m11_grid.shape, 6, 6))
    Rm2_grid = np.empty((*m11_grid.shape, 6, 6))

    for i in range(len(m11)):
        for j in range(len(m21)):
            M = Rm.copy()
            M[0, 0] = m11_grid[i, j]
            M[1, 0] = m21_grid[i, j]
            M[0, 1] = achromatic_m12_grid[i, j]  # type: ignore
            M[1, 1] = achromatic_m22_grid[i, j]  # type: ignore
            M_flipped = M.copy()
            M_flipped[0, 0], M_flipped[1, 1] = M[1, 1], M[0, 0]
            Rm_grid[i, j] = M
            Rm2_grid[i, j] = M_flipped

    tba_matrices_grid = Rd2 @ Rb3 @ Rm2_grid @ Rb2 @ Rm_grid @ Rb1 @ Rd1

    x2x2_grid = tba_matrices_grid[:, :, 0:2, 0:2]
    x_abs_trace_grid = np.abs(np.trace(x2x2_grid, axis1=2, axis2=3))
    x_abs_trace_grid_unmasked = x_abs_trace_grid

    x_abs_trace_grid_masked = np.ma.masked_where(
        np.abs(x_abs_trace_grid) > 2, x_abs_trace_grid
    )
    x_abs_trace_grid = x_abs_trace_grid_masked

    mask = np.ma.getmaskarray(x_abs_trace_grid_masked)

    # Custom coordinate formatter
    def format_coord(x_mouse, y_mouse):
        # Find nearest indices in the grid
        i = np.searchsorted(m11, x_mouse) - 1
        j = np.searchsorted(m21, y_mouse) - 1
        nrows, ncols = x_abs_trace_grid.shape
        if 0 <= i < ncols and 0 <= j < nrows:
            zval = x_abs_trace_grid[j, i]
            return f"x={x_mouse:1.3f}, y={y_mouse:1.3f}, z={zval:1.3f}"
        else:
            return f"x={x_mouse:1.3f}, y={y_mouse:1.3f}"
    fig, ax = plt.subplots()
    # ax.format_coord = format_coord

    x1, x2, y1, y2 = -0.9, -0.6, -0.2, 2.0  # subregion of the original image
    axins = ax.inset_axes(
        [0.50, 0.35, 0.45, 0.4],
        xlim=(x1, x2),
        ylim=(y1, y2),
        xticklabels=[],
        yticklabels=[]
    )

    cs = ax.contour(
        m11_grid,
        m21_grid,
        x_abs_trace_grid_unmasked,
        x_abs_trace_grid_masked,
        levels=[2.0],
        colors="black",
        linewidths=1.5,
        # label=r"$|\mathrm{Tr}(M_x)| = 2$",
    )

    cs = axins.contour(
        m11_grid,
        m21_grid,
        x_abs_trace_grid_unmasked,
        x_abs_trace_grid_masked,
        levels=[2.0],
        colors="black",
        linewidths=1.5,
        # label=r"$|\mathrm{Tr}(M_x)| = 2$",
    )
    
    # ax.contour(m11_grid, 21_grid

    if opt is None:
        pc = ax.pcolormesh(m11_grid, m21_grid, x_abs_trace_grid, shading="auto", cmap="cividis")
        cbar = fig.colorbar(pc, ax=ax)
        cbar.set_label(r"$|\mathrm{Tr}(M_x)|$")
    else:
        # mask = x_abs_trace_grid.mask
        mask = np.ma.getmaskarray(x_abs_trace_grid)
        # Shade only the unmasked region
        valid = ~mask
        overlay = np.ma.masked_where(~valid, valid.astype(float))
        # overlay = np.ma.masked_where(mask, np.ones_like(mask))
        from matplotlib.colors import ListedColormap
        grey_cmap = ListedColormap(["grey"])
        ax.pcolormesh(
            m11_grid,
            m21_grid,
            overlay,
            shading="auto",
            cmap=grey_cmap,
            alpha=0.2
        )
        axins.pcolormesh(
            m11_grid,
            m21_grid,
            overlay,
            shading="auto",
            cmap=grey_cmap,
            alpha=0.2
        )


        npoints = opt.iter
        mx2x2_intermediate = np.zeros((npoints, 2, 2))
        # my2x2_intermediate = np.zeros((npoints, 4, 4))

        for i in range(npoints):
            tba.update_quads(opt, index=i)
            mx2x2_intermediate[i] = tba.mx2x2_between_dipoles()
            # my2x2_intermediate[i] = tba.my2x2_between_dipoles()

        m11_intermediate = mx2x2_intermediate[:, 0, 0]
        m21_intermediate = mx2x2_intermediate[:, 1, 0]

        colors = np.linspace(0, 1, npoints)

        ax.scatter(m11_intermediate, m21_intermediate, c=colors, cmap="coolwarm", s=100)
        axins.scatter(m11_intermediate, m21_intermediate, c=colors, cmap="coolwarm", s=100)
        # Go back to optimal value.
        tba.update_quads(opt)
    # axins = ax.inset_axes(
    #     [-0.7, 2.0, 1.0, 5.0],
    #     xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    
    # axins.imshow(Z2, extent=extent, origin="lower")


    
    ax.plot(
        m11, isochronous_m21(tba, m11=m11), color="forestgreen", label=r"$R_{56} = 0$"
    )
    ax.plot(
        m11,
        csr_cancellation_m21s(tba, m11=m11)[0],
        color="crimson",
        label=r"$\Delta\,\varepsilon_n = 0$",
    )
    ax.plot(m11, csr_cancellation_m21s(tba, m11=m11)[1], color="crimson")

    axins.plot(
        m11, isochronous_m21(tba, m11=m11), color="forestgreen", label=r"$R_{56} = 0$"
    )
    axins.plot(
        m11,
        csr_cancellation_m21s(tba, m11=m11)[0],
        color="crimson",
        label=r"$\Delta\,\varepsilon_n = 0$",
    )
    axins.plot(m11, csr_cancellation_m21s(tba, m11=m11)[1], color="crimson")
    
    ax.indicate_inset_zoom(axins, edgecolor="black")

    # ax.plot(m11, csr_cancellation_m21s_old(m11, dipole_length=tba.dipole_length())[0], color="crimson", label=r"$\Delta\,\varepsilon_n = 0$")
    # ax.plot(m11, csr_cancellation_m21s_old(m11, dipole_length=tba.dipole_length())[1], color="crimson", label=r"$\Delta\,\varepsilon_n = 0$")

    # ax.plot(m11,

    # csr_line1, csr_line2 = csr_cancellation_m21s_old(m11, tba.dipole_length())
    # ax.plot(m11, csr_line1, color="red", label=r"$\Delta\,\varepsilon_n = 0$, old", marker="x", linestyle="")
    # ax.plot(m11, csr_line2, color="red", marker="x", linestyle="")

    ax.set_xlim(min(m11), max(m11))
    ax.set_ylim(min(m21), max(m21))

    ax.set_xlabel("$m_{11}$")
    ax.set_ylabel(r"$m_{21}\,/\,\mathrm{m^{-1}}$")
    ax.legend(loc=legend_loc)
    plt.show()
    return fig, ax


def plot_tba_periodic_beta0(tba, m11_range=(-3.0, 3.0), m21_range=(-15.0, 25.0),
                            opt=None,
                            legend_loc="upper right"):
    l_drift0 = 0.8
    m11 = np.linspace(*m11_range, num=1000)
    m21 = np.linspace(*m21_range, num=1000)

    dipole_length = tba.dipole_length()

    m11g, m21g = np.meshgrid(m11, m21)

    achromatic_beta0g, *_ = achromatic_periodic_twiss(l_drift0, m11g, m21g, dipole_length)

    achromatic_beta0g = np.ma.masked_invalid(achromatic_beta0g)

    # np.nan_to_num(achromatic_periodic_twiss


    fig, ax = plt.subplots()
    ax.plot(m11, isochronous_m21(m11, tba.dipole_length()), color="forestgreen", label=r"$R_{56} = 0$")
    csr_line1, csr_line2 = csr_cancellation_m21(m11, tba.dipole_length())

    if csr_line1.max() > csr_line2.max():
        csr_line = csr_line2
    else:
        csr_line = csr_line1

    ax.plot(m11, csr_line, color="crimson", label=r"$\Delta\,\varepsilon_n = 0$")

    # from IPython import embed; embed()

    # Custom coordinate formatter
    def format_coord(x_mouse, y_mouse):
        # Find nearest indices in the grid
        i = np.searchsorted(m11, x_mouse) - 1
        j = np.searchsorted(m21, y_mouse) - 1
        nrows, ncols = achromatic_beta0g.shape
        if 0 <= i < ncols and 0 <= j < nrows:
            zval = achromatic_beta0g[j, i]
            return f"x={x_mouse:1.3f}, y={y_mouse:1.3f}, z={zval:1.3f}"
        else:
            return f"x={x_mouse:1.3f}, y={y_mouse:1.3f}"

    ax.format_coord = format_coord

    if opt is not None:
        npoints = opt.iter
        mx2x2_intermediate = np.zeros((npoints, 2, 2))
        # my2x2_intermediate = np.zeros((npoints, 4, 4))

        for i in range(npoints):
            tba.update_quads(opt, index=i)
            mx2x2_intermediate[i] = tba.mx2x2_between_dipoles()
            # my2x2_intermediate[i] = tba.my2x2_between_dipoles()

        m11_intermediate = mx2x2_intermediate[:, 0, 0]
        m21_intermediate = mx2x2_intermediate[:, 1, 0]

        colors = np.linspace(0, 1, npoints)

        ax.scatter(m11_intermediate, m21_intermediate, c=colors, cmap="coolwarm", s=1)



    pc = ax.pcolormesh(m11g, m21g,
                       achromatic_beta0g,
                       shading="auto",
                       norm=colors.LogNorm(vmin=achromatic_beta0g.min(), vmax=achromatic_beta0g.max()))

    cbar = fig.colorbar(pc, ax=ax)
    cbar.set_label(r"$\beta_{x0}$")

    ax.set_xlim(min(m11), max(m11))
    ax.set_ylim(min(m21), max(m21))

    ax.set_xlabel("$m_{11}$")
    ax.set_ylabel(r"$m_{21}\,/\,\mathrm{m^{-1}}$")
    ax.legend(loc=legend_loc)


from felarc.csr import gaussian_bunch_emittance_growth


def plot_m11_m21_emittance_growth_contours(tba, emitn0, charge, bunch_length, energy,
                                           twiss0=None,
                                           m11_range=(-3.0, 3.0),
                                           m21_range=(-15.0, 25.0),
                                           legend_loc="upper right", opt=None):

    # Sample m11 and m21
    m11 = np.linspace(*m11_range, num=1000)
    m21 = np.linspace(*m21_range, num=1000)

    # Make grids for these sets of values
    m11_grid, m21_grid = np.meshgrid(m11, m21)

    # Set Corresopnding m12 and m22 grids for achromatic condition.
    achromatic_m12_grid, achromatic_m22_grid = achromatic_m12_m22(m11_grid, m21_grid, tba.dipole_length())

    # We get the matrices for the TBA.
    Rd1, Rb1, Rm, Rb2, Rmflipped, Rb3, Rd2 = tba.component_matrices()

    # We have 4 grids now, m11 and m21 are free variables.  The other 2 matrix entries,
    # m12 and m22 are constrained in order to ensure achromatism.
    # So for each pair of m11 and m21, we have also an m12 and m22.

    # The matrices for between dipoles we will manually edit to sample
    # the range of m11 and m21 we are interested in.
    Rm_grid = np.empty((*m11_grid.shape, 6, 6))
    Rm2_grid = np.empty((*m11_grid.shape, 6, 6))

    # For each dipole we must sample the R_16^s->s1, R_26^s->s2.  i.e.: the matrix from
    # a given point in the dipole to the end of the dipole.
    dipole_rmatrices_to_dipole_end = []
    espread_derivative_per_unit_length = []
    z_in_dipole = []
    for dipole in tba.dipoles:
        z, rmatrices_to_z = transfer_maps_z(MagneticLattice(dipole.elements), step_length=0.01) # type: ignore
        rmatrix_dipole_start_to_dipole_end = rmatrices_to_z[-1]
        inverses = np.linalg.inv(rmatrices_to_z)
        r_to_end = rmatrix_dipole_start_to_dipole_end @ inverses
        despread_ds = gaussian_bunch_espread_per_unit_length(dipole.curvature(), charge, bunch_length, energy)
        # Store the matrices, despread_ds, and z for this dipole.
        dipole_rmatrices_to_dipole_end.append((z, r_to_end))
        espread_derivative_per_unit_length.append(despread_ds)
        z_in_dipole.append(z)

    # Loop over the m11s and m21s to generate a 6x6 transfer matrix for each point on m11, m21 grid.
    for i in range(len(m11)):
        for j in range(len(m21)):
            # Update the matrices based on the m11 and m21 we see here.
            M = Rm.copy()
            M[0, 0] = m11_grid[i, j]
            M[1, 0] = m21_grid[i, j]
            M[0, 1] = achromatic_m12_grid[i, j] # type: ignore
            M[1, 1] = achromatic_m22_grid[i, j] # type: ignore
            M_flipped = M.copy()
            M_flipped[0, 0], M_flipped[1, 1] = M[1, 1], M[0, 0]
            Rm_grid[i, j] = M
            Rm2_grid[i, j] = M_flipped

    # Now, for each dipole, we need to calculate the integrals for
    # ⟨Δx²⟩, ⟨Δx Δx̂'⟩ and ⟨Δx'²⟩.  This involves calculating the
    # matrix elements from each point in the dipole to the end of the
    # lattice.
    # Get the matrices from each point in z along the dipole to the end of the dipole.
    dipole1_z, dipole1_rmatrices_to_dipole_end = dipole_rmatrices_to_dipole_end[0]
    dipole2_z, dipole2_rmatrices_to_dipole_end = dipole_rmatrices_to_dipole_end[1]
    dipole3_z, dipole3_rmatrices_to_dipole_end = dipole_rmatrices_to_dipole_end[2]
    # Get the matrices from the end of each dipole to the end of the lattice for each point in (m11, m21)-space.
    post_dipole1 = Rd2 @ Rb3 @ Rm2_grid @ Rb2 @ Rm_grid
    post_dipole2 = Rd2 @ Rb3 @ Rm2_grid
    post_dipole3 = Rd2
    # Now for each point in dipole0 we want to generate the transfer
    # matrix to the end of the lattice.  So in principle the output is
    # shape (nxgrid, nygrid, nzdipole, 6, 6).  So we need einsum to do
    # such a nested matrix multiplicatin fast.  Additionally we only
    # care about R16 and R26 so we drop additional matrix elements (we
    # keep only the first two rows of the matrices on the left, and
    # the 5th column of the matrices on the right in order to get R16
    # ([0, 5]) and R26 ([1, 5]).
    # i is the row of the grid (m11)
    # j is the column of the grid (m21)
    # a is the row of the 6x6 transport matrix
    # b is the columns of the 6x6 transport matrix.
    # k is the path length position along the dipole.
    dipole1_r12_r16_to_end = np.einsum('ijab,kb->ijka', post_dipole1[..., [0, 1], :], dipole1_rmatrices_to_dipole_end[..., :, 5])
    dipole2_r12_r16_to_end = np.einsum('ijab,kb->ijka', post_dipole2[..., [0, 1], :], dipole2_rmatrices_to_dipole_end[..., :, 5])
    dipole3_r12_r16_to_end = np.einsum("ab,kb->ka", post_dipole3[[0, 1]], dipole3_rmatrices_to_dipole_end[..., :, 5])

    # from IPython import embed; embed()
    #
    dipole1_r16_int = np.trapezoid(dipole1_r12_r16_to_end[..., 0] * espread_derivative_per_unit_length[0], x=dipole1_z)
    dipole1_r26_int = np.trapezoid(dipole1_r12_r16_to_end[..., 1] * espread_derivative_per_unit_length[0], x=dipole1_z)
    dipole2_r16_int = np.trapezoid(dipole2_r12_r16_to_end[..., 0] * espread_derivative_per_unit_length[1], x=dipole2_z)
    dipole2_r26_int = np.trapezoid(dipole2_r12_r16_to_end[..., 1] * espread_derivative_per_unit_length[1], x=dipole2_z)
    dipole3_r16_int = np.trapezoid(dipole3_r12_r16_to_end[..., 0] * espread_derivative_per_unit_length[2], x=dipole3_z)
    dipole3_r26_int = np.trapezoid(dipole3_r12_r16_to_end[..., 1] * espread_derivative_per_unit_length[2], x=dipole3_z)

    rms_dx2 = (dipole1_r16_int + dipole2_r16_int + dipole3_r16_int) ** 2
    rms_dxp2 = (dipole1_r26_int + dipole2_r26_int + dipole3_r26_int) ** 2
    rms_dxdx_hatp2 = (dipole1_r16_int + dipole2_r16_int + dipole3_r16_int) * (dipole1_r26_int + dipole2_r26_int + dipole3_r26_int)

    twiss1 = tba.twiss1()
    relgamma = energy / m_e_GeV
    emitg0 = emitn0 / relgamma
    emitg1 = np.sqrt(emitg0**2 + emitg0 * (twiss1.beta_x * rms_dxp2**2
                                           + 2 * twiss1.alpha_x * rms_dxdx_hatp2
                                           + twiss1.gamma_x * rms_dx2**2))
    emitn1 = emitg1 * relgamma


    # dipole1_r16 = np.trapezoid(dipole1_r12_r16_to_end[..., 0] * espread_derivative_per_unit_length[0], x=z_in_dipole[0])

    # from IPython import embed; embed()
    # # dipole1_r12_r16_to_end

    # # Now we calculate the integrals for each dipole, one by one.
    # for (r



    # for i, (dipole, (z, rmatrices_to_dipole_end)) in enumerate(zip(tba.dipoles, dipole_rmatrices_to_end)):
    #     # For each sequence of dipole matrices (from z -> z_dipole_end) we need the matrix
    #     # from the dipole end to the end of the lattice.

    #     from IPython import embed; embed()
    #     curvature = dipole.angle / dipole.l
    #     inverses = np.linalg.inv(rmatrices_to_dipole_end)
    #     r_end = rmatrices_to_dipole_end[-1]
    #     rmatrices_to_dipole_end = r_end @ inverses
    #     # rmatrices_to_lattice_end = rmatrices_to_dipole_end

    #     if i == 0:
    #         pass

        # despread_ds = gaussian_bunch_emittance_growth(
        #     dipole.mlat(),
        #     curvature, charge, bunch_length, energy
        # )
        # from IPython import embed; embed()


    # Now we recalculate the full matrices
    tba_matrices_grid = Rd2 @ Rb3 @ Rm2_grid @ Rb2 @ Rm_grid @ Rb1 @ Rd1

    x2x2_grid = tba_matrices_grid[:, :, 0:2, 0:2]
    x_abs_trace_grid = np.abs(np.trace(x2x2_grid, axis1=2, axis2=3))
    x_abs_trace_grid_unmasked = x_abs_trace_grid

    x_abs_trace_grid_masked = np.ma.masked_where(np.abs(x_abs_trace_grid) > 2, x_abs_trace_grid)
    x_abs_trace_grid = x_abs_trace_grid_masked

    # mask = np.ma.getmaskarray(x_abs_trace_grid_masked)

    fig, ax = plt.subplots()

    # Custom coordinate formatter
    def format_coord(x: float, y: float) -> str:
        # x is x_mouse, y is y_mouse coordinates.
        # Find nearest indices in the grid
        i = np.searchsorted(m11, x) - 1
        j = np.searchsorted(m21, y) - 1
        nrows, ncols = x_abs_trace_grid.shape
        if 0 <= i < ncols and 0 <= j < nrows:
            zval = x_abs_trace_grid[j, i]
            return f"x={x:1.3f}, y={y:1.3f}, z={zval:1.3f}"
        else:
            return f"x={x:1.3f}, y={y:1.3f}"

    ax.format_coord = format_coord

    ax.contour(m11_grid, m21_grid, x_abs_trace_grid_unmasked, x_abs_trace_grid_masked, levels=[2.0], colors="black", linewidths=1.5)

    if opt is not None:
        npoints = opt.iter
        mx2x2_intermediate = np.zeros((npoints, 2, 2))
        # my2x2_intermediate = np.zeros((npoints, 4, 4))

        for i in range(npoints):
            tba.update_quads(opt, index=i)
            mx2x2_intermediate[i] = tba.mx2x2_between_dipoles()
            # my2x2_intermediate[i] = tba.my2x2_between_dipoles()

        m11_intermediate = mx2x2_intermediate[:, 0, 0]
        m21_intermediate = mx2x2_intermediate[:, 1, 0]

        colors = np.linspace(0, 1, npoints)

        ax.scatter(m11_intermediate, m21_intermediate, c=colors, cmap="coolwarm", s=12)
        # Go back to optimal value.
        tba.update_quads(opt)

    # We need to express teh trace in terms of m12 and m11.

    # pc = plt.pcolormesh(xv, yv, z, cmap='viridis', shading='auto')

    # pc = ax.pcolormesh(m11_grid, m21_grid, x_abs_trace_grid, shading="auto")
    # cbar = fig.colorbar(pc, ax=ax)
    # cbar.set_label(r"$|\mathrm{Tr}(M_x)|$")

    ax.plot(m11, isochronous_m21(m11, tba.dipole_length()), color="forestgreen", label=r"$R_{56} = 0$")

    csr_line1, csr_line2 = csr_cancellation_m21(m11, tba.dipole_length())

    ax.plot(m11, csr_line1, color="crimson", label=r"$\Delta\,\varepsilon_n = 0$")
    ax.plot(m11, csr_line2, color="crimson")

    ax.set_xlim(min(m11), max(m11))
    ax.set_ylim(min(m21), max(m21))

    ax.set_xlabel("$m_{11}$")
    ax.set_ylabel(r"$m_{21}\,/\,\mathrm{m^{-1}}$")
    ax.legend(loc=legend_loc)

    return fig, ax

def plot_r16_r26_to_end(tba, ):
    pass


def plot_optimisation_history(opt, tba):
    fig, ax = plt.subplots(5, 1, sharex=True)

    m11_history = []
    m21_history = []
    xtrace_history = []
    ytrace_history = []
    r56_history = []
    for i in range(opt.iter):
        print(i)
        tba.update_quads(opt, index=i)
        mx2x2 = tba.mx2x2_between_dipoles()
        m11_history.append(mx2x2[0, 0])
        m21_history.append(mx2x2[1, 0])
        xtrace_history.append(np.trace(mx2x2))
        my2x2 = tba.my2x2_between_dipoles()
        ytrace_history.append(np.trace(my2x2))
        r56_history.append(tba.rtot()[4, 5])

    ax[0].plot(m11_history)
    ax[1].plot(m21_history)
    ax[2].plot(xtrace_history)
    ax[3].plot(ytrace_history)
    ax[4].plot(r56_history)

    ax[0].set_ylabel("$m_{11}$")
    ax[1].set_ylabel(r"$m_{21}\,/\,\mathrm{m^{-1}}$")
    ax[2].set_ylabel(r"$\mathrm{Tr}(M_x)$")
    ax[3].set_ylabel(r"$\mathrm{Tr}(M_y)$")
    ax[4].set_ylabel(r"$R_{56}$ / m")
    ax[4].set_xlabel("Iteration")

    tba.update_quads(opt)

    return fig, ax

def plot_optics_with_dispersions(mlat: MagneticLattice, twiss0: Twiss | None = None):
    llat = from_ocelot(mlat)
    fig, ax = subplots_with_lattices([llat, None, None, None, None])
    (_, ax1, ax2, ax3, ax4) = ax

    df = twiss(mlat, twiss0, return_df=True, nPoints=10000
               )

    ax1.plot(df.s, df.beta_x, label="$x$")  # type: ignore
    ax1.plot(df.s, df.beta_y, label="$y$")  # type: ignore

    csr_tracker = LinearCSRTracker(mlat, k=1)
    s, coords = csr_tracker.track(0, 0, 0, 0, ds=0.01)

    dx_tracker = LinearCSRTracker(mlat, k=0)
    dx_s, dx_coords = dx_tracker.track(0, 0, 1, 0, ds=0.01) #0.0001)

    # ax1.plot(df.s, df.Dx, marker=".", label="OCELOT")
    # ax2.plot(df.s, df.Dxp)

    # from IPython import embed; embed()

    # bunch_length = 7e6
    # bunch_charge = 500e-12


    ax2.plot(dx_s, dx_coords.x)
    # ax3.plot(dx_s, dx_coords.px)
    # ax2.plot(s, coords.x)
    ax3.plot(s, coords.x)
    # ax5.plot(s, coords.dp_csr)
    # ax[3].plot(df.s, df.Dx)
    # ax1.legend()
    ax4.plot(df.s, df.mux / np.pi)  # type: ignore

    ax1.legend()
    # # csr_wake = calculate_linear_csr_wake(bunch_charge, bunch_length, curvature)

    # ax[-2].plot(df.s, csr_wake)
    ax1.set_ylabel(r"$\beta$ / m")
    ax2.set_ylabel(r"$D$ / m")
    ax3.set_ylabel(r"$\xi\,/\,\mathrm{m}^2$")
    ax4.set_ylabel(r"$\mu_x\,/\,\pi$")
    # ax4.set_ylabel(r"$\xi'$ / rad")

    ax[-1].set_xlabel("$s$ / m")

    return fig, ax

