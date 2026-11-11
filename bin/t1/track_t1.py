"""Track the S2E STN10 beam through TL34/SA2 + T1.

T1 counterpart of ../nina-t20/track_nina_t20.py.  The line is tl34_sa2 + t1 (the vertical
KL kickers and QF.2012, then the T1 arc and its FODO to ENSEC.2197.T1), built with
second-order maps, starting at s = 1980.39 m with tl34_sa2.tws0.

Tracking is felarc's standard track_with_test_beam: the DESY start-to-end beam at STN10
(100 or 250 pC), Twiss-matched to tl34_sa2.tws0 with offsets removed, tracked in 0.1 m
steps with optional ocelot 1D CSR over the whole line.  The input beam is regenerated with
load_test_parray, which is exactly what track_with_test_beam starts from.

Both beams are saved next to this script for plot_t1_tracking.py.

Usage:
    python track_t1.py --charge 100 --no-csr
    python track_t1.py --charge 100 --csr
"""

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

from ocelot.cpbd.io import save_particle_array
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.transformations import SecondTM

import t1
import tl34_sa2
from felarc.tracking import load_test_parray, track_with_test_beam

HERE = Path(__file__).resolve().parent


def output_stem(outdir: Path, charge: int, csr: bool) -> Path:
    return outdir / f"t1-{'csr' if csr else 'nocsr'}-{charge}pC"


def track_t1(charge: int, csr: bool):
    mlat = MagneticLattice(tl34_sa2.cell + t1.cell, method={"global": SecondTM})
    twiss0 = tl34_sa2.tws0

    parray0 = load_test_parray(twiss0, charge=charge)  # type: ignore[arg-type]
    # track_with_test_beam reads the lattice from a TBA's .mlat; give it just that.
    parray1 = track_with_test_beam(
        SimpleNamespace(mlat=mlat), twiss0, csr=csr, charge=charge  # type: ignore[arg-type]
    )
    return parray0, parray1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--charge", type=int, choices=(100, 250), default=100)
    parser.add_argument("--csr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outdir", type=Path, default=HERE)
    args = parser.parse_args()

    start = time.time()
    parray0, parray1 = track_t1(args.charge, args.csr)
    print(f"Tracked {parray0.n} particles in {time.time() - start:.0f} s")

    stem = output_stem(args.outdir, args.charge, args.csr)
    save_particle_array(f"{stem}-parray0.npz", parray0)
    save_particle_array(f"{stem}-parray1.npz", parray1)
    print(f"Saved {stem}-parray{{0,1}}.npz")


if __name__ == "__main__":
    main()
