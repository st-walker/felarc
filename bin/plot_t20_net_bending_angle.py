"""Net bending angle along the TL20 + T20 arc.

Companion to plot_t20_optics_with_dy.py: same lattice, same machine panel, same absolute s
(measured from the cathode, 2034.59 m to 2123.33 m), but instead of the optics it shows the
cumulative bending angle.

Each bend's angle is split into a horizontal part, angle * cos(tilt), and a vertical part,
angle * sin(tilt), and these are summed along the beamline.  This handles the vertical
bends (tilt = pi/2), the rolled BZ.2055.T20 and the sliced QK combined-function quad.  Within
a magnet the angle grows linearly with path length.  Summing the components linearly ignores
the non-commutativity of rotations, which is a second-order effect (~theta^2, i.e. ~1e-5 of
the angles here) and invisible at mrad scale.

The bottom panel is the total angle the beam has been bent through, the running sum of
|angle|, which is what matters for ISR and CSR, as opposed to the net direction change.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from latdraw.convert import from_ocelot
from latdraw.plot import subplots_with_lattices
from ocelot.cpbd.magnetic_lattice import MagneticLattice

sys.path.append(str(Path(__file__).resolve().parent))

from plot_t20_optics_with_dy import make_lattice  # noqa: E402


def net_bending_angles(
    mlat: MagneticLattice, s0: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative horizontal, vertical and total |angle| (rad) at element boundaries.

    Returns (s, theta_x, theta_y, theta_total), each sampled at the entrance and exit of
    every element, so a straight-line plot between points is exact.
    """
    s = [s0]
    theta_x = [0.0]
    theta_y = [0.0]
    theta_total = [0.0]

    for element in mlat.sequence:
        length = element.l
        angle = getattr(element, "angle", 0.0) or 0.0
        tilt = getattr(element, "tilt", 0.0) or 0.0

        s.append(s[-1] + length)
        theta_x.append(theta_x[-1] + angle * np.cos(tilt))
        theta_y.append(theta_y[-1] + angle * np.sin(tilt))
        theta_total.append(theta_total[-1] + abs(angle))

    return np.array(s), np.array(theta_x), np.array(theta_y), np.array(theta_total)


def plot_net_bending_angle(mlat: MagneticLattice, s_offset: float = 0.0):
    llat = from_ocelot(mlat)
    s, theta_x, theta_y, theta_total = net_bending_angles(mlat, s0=s_offset)

    fig, ax = subplots_with_lattices([llat, None, None], s_offset=s_offset)
    (_, ax0, ax1) = ax

    ax0.plot(s, theta_x * 1e3, label="$x$")
    ax0.plot(s, theta_y * 1e3, label="$y$")
    ax0.axhline(0, alpha=0.2, linestyle="--", color="gray")

    ax1.plot(s, theta_total * 1e3, color="black")

    ax0.legend()

    ax0.set_ylabel(r"$\theta_\mathrm{net}$ / mrad")
    ax1.set_ylabel(r"$\sum|\theta|$ / mrad")
    ax1.set_xlabel(r"$s$ from cathode / m")

    print(f"Net horizontal angle: {theta_x[-1] * 1e3: .4f} mrad")
    print(f"Net vertical angle:   {theta_y[-1] * 1e3: .4f} mrad")
    print(f"Total |angle|:        {theta_total[-1] * 1e3: .4f} mrad")

    return fig, ax


def main() -> None:
    mlat, twiss0 = make_lattice()
    fig, _ = plot_net_bending_angle(mlat, s_offset=twiss0.s)

    outdir = Path(__file__).resolve().parent
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"t20-net-bending-angle.{suffix}", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
