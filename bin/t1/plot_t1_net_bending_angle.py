"""Net bending angle along the TL34/SA2 + T1 arc.

T1 counterpart of ../plot_t20_net_bending_angle.py, using the same plotting function: top
panel is the net horizontal and vertical angle, bottom panel the bend angle of each element,
both in degrees.

The line is TL34/SA2 (the vertical KL kickers and QF.2012) followed by T1, mirroring
TL20 + T20.  TL2 upstream has no bends, so it is left out.  s is absolute machine
coordinate from the cathode, taken from tl34_sa2.tws0.s = 1980.39 m.

As for T20, the QK quad in T1 is modelled as 200 thin slices, so its per-slice angle is
too small to see in the bottom panel, although it shows as a ramp in the top one.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from ocelot.cpbd.magnetic_lattice import MagneticLattice

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE.parent))

import t1  # noqa: E402
import tl34_sa2  # noqa: E402
from plot_t20_net_bending_angle import plot_net_bending_angle  # noqa: E402


def main() -> None:
    mlat = MagneticLattice(tl34_sa2.cell + t1.cell)
    fig, _ = plot_net_bending_angle(mlat, s_offset=tl34_sa2.tws0.s)

    for suffix in ("png", "pdf"):
        fig.savefig(HERE / f"t1-net-bending-angle.{suffix}", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
