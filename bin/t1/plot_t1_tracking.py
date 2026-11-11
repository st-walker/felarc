"""show_e_beam figures of the STN10 beam before and after TL34/SA2 + T1, for slides.

Reads the beams written by track_t1.py and makes one ocelot show_e_beam figure each for

* BEFORE T1: the matched input beam at the start of TL34/SA2,
* AFTER T1, no CSR: the beam at ENSEC.2197.T1 tracked without CSR,
* AFTER T1, with CSR: the same with CSR,

each saved as a PNG next to the beams.  A beam that does not exist yet (e.g. the CSR run
is still going) is skipped.

Usage:
    python plot_t1_tracking.py --charge 100
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from ocelot.cpbd.io import load_particle_array
from ocelot.gui.accelerator import show_e_beam

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--charge", type=int, choices=(100, 250), default=100)
    parser.add_argument("--dir", type=Path, default=HERE, help="where the .npz beams are")
    args = parser.parse_args()

    charge = args.charge
    figures = [
        ("BEFORE T1", f"t1-nocsr-{charge}pC-parray0.npz", f"t1-{charge}pC-before.png"),
        ("AFTER T1, no CSR", f"t1-nocsr-{charge}pC-parray1.npz", f"t1-{charge}pC-after-nocsr.png"),
        ("AFTER T1, with CSR", f"t1-csr-{charge}pC-parray1.npz", f"t1-{charge}pC-after-csr.png"),
    ]

    for nfig, (title, beam_name, png_name) in enumerate(figures, start=1):
        beam_path = args.dir / beam_name
        if not beam_path.exists():
            print(f"Skipping {title}: {beam_path} not found")
            continue
        parray = load_particle_array(str(beam_path))
        # figname: show_e_beam draws on figure 40 by default, so give each its own.
        show_e_beam(parray, title=title, figname=nfig)
        plt.gcf().savefig(args.dir / png_name, dpi=200)
        print(f"Saved {args.dir / png_name}")

    plt.show()


if __name__ == "__main__":
    main()
