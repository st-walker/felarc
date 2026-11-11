"""Before/after comparison of the STN10 beam through the standard (Nina) TL20 + T20.

Same approach as ../plot_isochronous_t20.py, using ocelot's compare_beams on the beams
written by track_nina_t20.py:

1. start of TL20 vs end of T20, tracked without CSR;
2. end of T20 with CSR vs without CSR.

The second comparison is skipped if the CSR run has not been done yet.  Each figure is also
saved next to the beams.

Usage:
    python plot_nina_t20_tracking.py --charge 250
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from ocelot.cpbd.io import load_particle_array
from ocelot.gui.accelerator import compare_beams

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--charge", type=int, choices=(100, 250), default=250)
    parser.add_argument("--dir", type=Path, default=HERE, help="where the .npz beams are")
    args = parser.parse_args()

    charge = args.charge
    nocsr = args.dir / f"nina-t20-nocsr-{charge}pC"
    csr = args.dir / f"nina-t20-csr-{charge}pC"

    parray0 = load_particle_array(f"{nocsr}-parray0.npz")
    parray1_no_csr = load_particle_array(f"{nocsr}-parray1.npz")

    compare_beams(
        parray0,
        parray1_no_csr,
        legend_beam1="Start of TL20",
        legend_beam2="End of T20",
        title=f"Standard T20, {charge} pC, without CSR",
        nfig=1,  # compare_beams draws on figure 40 by default, so give each its own
    )
    plt.gcf().savefig(args.dir / f"nina-t20-{charge}pC-start-vs-end-nocsr.png", dpi=200)

    csr_file = Path(f"{csr}-parray1.npz")
    if csr_file.exists():
        parray1_csr = load_particle_array(str(csr_file))
        compare_beams(
            parray1_csr,
            parray1_no_csr,
            legend_beam1="With CSR",
            legend_beam2="Without CSR",
            title=f"Standard T20, {charge} pC, end of T20 with/without CSR",
            nfig=2,
        )
        plt.gcf().savefig(args.dir / f"nina-t20-{charge}pC-csr-vs-nocsr.png", dpi=200)
    else:
        print(f"No CSR run yet ({csr_file}); skipping the with/without CSR comparison")

    plt.show()


if __name__ == "__main__":
    main()
