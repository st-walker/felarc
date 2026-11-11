"""Net bending angle along the TL20 + T20 arc.

Companion to plot_t20_optics_with_dy.py: same lattice, same machine panel, same absolute s
(measured from the cathode, 2034.59 m to 2123.33 m), but instead of the optics it shows
bending angles, in degrees.

Each bend's angle is split into a horizontal part, angle * cos(tilt), and a vertical part,
angle * sin(tilt).  This handles the vertical bends (tilt = pi/2), the rolled BZ.2055.T20
and the sliced QK combined-function quad.

Top panel: the net (cumulative) angle, summed along the beamline.  Within a magnet it grows
linearly with path length.  Summing the components linearly ignores the non-commutativity
of rotations, which is a second-order effect (~theta^2, i.e. ~1e-5 of the angles here).

Bottom panel: the bend angle of each individual element, drawn as a step across the
element's length; drifts, quads, markers etc. are 0.  Note the QK quad is modelled as 200
thin slices, so each slice's angle is tiny and it barely shows here even though it is
visible as a ramp in the top panel.
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


def element_bending_angles(
    mlat: MagneticLattice, s0: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-element horizontal and vertical bend angles (rad).

    Returns (s, angle_x, angle_y) where s has one more entry than the angle arrays: element
    i spans s[i] to s[i + 1] and bends by angle_x[i], angle_y[i].
    """
    s = [s0]
    angle_x = []
    angle_y = []

    for element in mlat.sequence:
        angle = getattr(element, "angle", 0.0) or 0.0
        tilt = getattr(element, "tilt", 0.0) or 0.0

        s.append(s[-1] + element.l)
        angle_x.append(angle * np.cos(tilt))
        angle_y.append(angle * np.sin(tilt))

    return np.array(s), np.array(angle_x), np.array(angle_y)


def plot_net_bending_angle(mlat: MagneticLattice, s_offset: float = 0.0):
    llat = from_ocelot(mlat)
    s, angle_x, angle_y = element_bending_angles(mlat, s0=s_offset)

    # Net angle at every element boundary; linear within an element, so exact.
    theta_x = np.concatenate([[0.0], np.cumsum(angle_x)])
    theta_y = np.concatenate([[0.0], np.cumsum(angle_y)])

    fig, ax = subplots_with_lattices([llat, None, None], s_offset=s_offset)
    (_, ax0, ax1) = ax

    ax0.plot(s, np.degrees(theta_x), label="$x$")
    ax0.plot(s, np.degrees(theta_y), label="$y$")
    ax0.axhline(0, alpha=0.2, linestyle="--", color="gray")

    # Constant value across each element: ax.stairs takes the n + 1 edges directly.
    ax1.stairs(np.degrees(angle_x), s, baseline=None)
    ax1.stairs(np.degrees(angle_y), s, baseline=None)
    ax1.axhline(0, alpha=0.2, linestyle="--", color="gray")

    ax0.legend()

    ax0.set_ylabel(r"$\theta_\mathrm{net}$ / °")
    ax1.set_ylabel(r"$\theta_\mathrm{element}$ / °")
    ax1.set_xlabel(r"$s$ from cathode / m")

    print(f"Net horizontal angle: {np.degrees(theta_x[-1]): .4f}°")
    print(f"Net vertical angle:   {np.degrees(theta_y[-1]): .4f}°")

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
