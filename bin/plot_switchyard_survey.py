"""Floor plan of the EuXFEL switchyard: the TLD, T1, T2 and T20 lines.

Topology, straight from the MAD-8 deck (XFEL_TL.txm)::

    CL -TL1-+- TL2TLD ------------------> TLD   (dump)
            +- TL2TL -+- TL3T1 -> T1 -> SA2     (south)
                      +- TL3TL -+- TL5T20 -> T20
                                +- TL5TL  -> T2 -> SA1   (north)

Three nested splits, all sharing TL1.  Each line is therefore surveyed from the
common start of TL1, which puts all four in one global frame without having to
place any of them by hand.

TLD, T1 and T2 come from the euxfel package (~/repos/EuXFEL-Lattice), generated
from the MAD-8 SURVEY tapes.  T20 is felarc's own lattice (bin/local_tl20.py +
bin/local_t20.py), spliced onto the north line at STSUB.2034.TL.

Two s conventions meet here and must not be confused:

* element names encode the tunnel Z coordinate (STSUB.2034.TL at Z = 2034.59 m),
  which is what felarc's lattice files use;
* the MAD-8 model measures s from the gun, which sits at Z = 23.2008 m
  (``set, z0_start, 23.2`` in Run_T20.txm), so euxfel s = tunnel Z - 23.2008.

The plot uses the tunnel Z, i.e. the convention the element names follow.

The survey is ocelot's own (``MagneticLattice.survey``), which honours each
dipole's tilt.  latdraw's survey does not -- it would draw TLD's 6.8 m dive to
the dump as a horizontal excursion.  One thing ocelot 25.6 does miss: euxfel's
SRot/YRot frame rotations on the T1 branch need ``get_transfer_geometry``, which
only the newer ocelot-collab has.  They are a ~10 urad roll patch, i.e. a few mm
at the end of T1, so nothing visible here.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from euxfel.subsequences import sase1, t1, tl2, tl2tld, tl34, tl34_sa2
from matplotlib.patches import Patch, Rectangle
from ocelot.cpbd.elements import (Drift, Quadrupole, RBend, SBend, Sextupole)
from ocelot.cpbd.magnetic_lattice import MagneticLattice, flatten

sys.path.append(str(Path(__file__).resolve().parent))

import local_t20  # noqa: E402
import local_tl20  # noqa: E402

#: Tunnel Z of the gun, GUN.23.I1 (``set, z0_start, 23.2``).  Element names are
#: in tunnel Z; the MAD-8 model's own s starts at the gun.
GUN_Z = 23.2008

#: Where TL20 leaves the north line: the marker STSUB.2034.TL.
T20_BRANCH_Z = local_tl20.tws0.s

#: Last marker of T2, before the SASE1 undulators.
T2_END = "ENSEC.2235.T2"


def cells(*modules):
    """Flatten the given subsequence modules into one element list."""
    out = []
    for module in modules:
        out.extend(flatten(module.cell))
    return out


def start_z(module) -> float:
    """Tunnel Z of the first element of a subsequence."""
    return module.twiss0.s + GUN_Z


def truncate_at_marker(elements, name):
    for i, element in enumerate(elements):
        if element.id == name:
            return elements[: i + 1]
    raise ValueError(f"No marker {name} in this path")


def truncate_at_z(elements, z0, z_cut):
    """Cut the line at tunnel Z = z_cut, splitting the drift that spans it."""
    out = []
    z = z0
    for element in elements:
        length = getattr(element, "l", 0.0) or 0.0
        if z + length <= z_cut:
            out.append(element)
            z += length
            continue
        if z < z_cut:
            if not isinstance(element, Drift):
                raise ValueError(
                    f"Branch point Z={z_cut} falls inside {element.id}, not a drift"
                )
            out.append(Drift(l=z_cut - z, eid=f"{element.id}_SPLIT"))
        return out
    raise ValueError(f"Line ends before Z={z_cut}")


def survey(elements, z0):
    """Survey an element list: tunnel Z, horizontal x and vertical y, in metres.

    The returned arrays have one point per element boundary, so element i of
    ``mlat.sequence`` spans points i and i + 1.
    """
    mlat = MagneticLattice(elements)
    x, y, z, _, _ = mlat.survey()
    return mlat, np.array(z) + z0, np.array(x), np.array(y)


#: Magnet types drawn on the floor plan: colour and box height in metres.
#: latdraw's colours, so the boxes read the same way as the machine panel of
#: the optics plots.  Everything else (drifts, markers, monitors, correctors)
#: is skipped: at 400 m across, a corrector is a pixel of clutter.  Boxes are
#: coloured by type rather than by line, because the four paths share a trunk
#: and would otherwise overdraw each other's magnets in the wrong colour.
MAGNET_STYLES = {
    SBend: ("blue", 0.5),
    RBend: ("blue", 0.5),
    Quadrupole: ("red", 0.35),
    Sextupole: ("green", 0.35),
}


def draw_magnets(axes, mlat, z, coord):
    """Draw each magnet as a box on the surveyed trajectory."""
    for i, element in enumerate(mlat.sequence):
        try:
            colour, height = MAGNET_STYLES[type(element)]
        except KeyError:
            continue
        axes.add_patch(
            Rectangle(
                (z[i], 0.5 * (coord[i] + coord[i + 1]) - 0.5 * height),
                z[i + 1] - z[i],
                height,
                facecolor=colour,
                edgecolor="none",
                zorder=3,
            )
        )


def build_paths():
    """The four branches, each as (elements, tunnel Z of its first element)."""
    north = cells(tl2, tl34, sase1)
    z0 = start_z(tl2)

    t20 = truncate_at_z(north, z0, T20_BRANCH_Z)
    t20 += cells(local_tl20, local_t20)

    return {
        "TLD": (cells(tl2tld), start_z(tl2tld)),
        "T1": (cells(tl2, tl34_sa2, t1), z0),
        "T2": (truncate_at_marker(north, T2_END), z0),
        "T20": (t20, z0),
    }


def plot_switchyard():
    """Floor plan (x) above elevation (y), sharing the tunnel Z axis.

    TLD separates almost entirely in the vertical, so the x panel alone would
    give a misleading picture of the fan-out.  Magnets are drawn on the x panel
    only; the y panel shows just the trajectories.
    """
    fig, (ax_x, ax_y) = plt.subplots(
        nrows=2, sharex=True, figsize=(11, 7), layout="constrained"
    )

    for label, (elements, z0) in build_paths().items():
        mlat, z, x, y = survey(elements, z0)
        (line,) = ax_x.plot(z, x, label=label, linewidth=0.8)
        ax_y.plot(z, y, color=line.get_color(), linewidth=0.8)
        # Magnets on the floor plan only; the elevation shows just the trajectories.
        draw_magnets(ax_x, mlat, z, x)
        for axes, coord in ((ax_x, x), (ax_y, y)):
            axes.annotate(
                label,
                xy=(z[-1], coord[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                color=line.get_color(),
            )

    for axes in (ax_x, ax_y):
        axes.axvline(T20_BRANCH_Z, color="gray", alpha=0.3, linestyle="--")
        axes.grid(alpha=0.2)

    ax_x.set_ylabel(r"$x$ / m")
    ax_y.set_ylabel(r"$y$ / m")
    ax_y.set_xlabel(r"$z$ / m")
    line_legend = ax_x.legend(title="line", loc="upper left")
    ax_x.add_artist(line_legend)
    ax_x.legend(
        handles=[
            Patch(facecolor="blue", label="dipole"),
            Patch(facecolor="red", label="quadrupole"),
            Patch(facecolor="green", label="sextupole"),
        ],
        title="magnet",
        loc="lower left",
    )

    return fig, (ax_x, ax_y)


def main() -> None:
    fig, _ = plot_switchyard()

    outdir = Path(__file__).resolve().parent
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"switchyard-survey.{suffix}", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
