"""Track the S2E STN10 beam through the standard (Nina) TL20 + T20 model.

The lattice is the full model, local_tl20 + local_t20: vertical kickers, dipole tilts and
the sliced QK quad all included (not the flattened "x" variant used in
isochronous-t20/adjust-t20.py).  It is built with second-order maps.

Tracking is felarc's standard track_with_test_beam: the DESY start-to-end beam at STN10
(100 or 250 pC, downloaded and cached), Twiss-matched to local_tl20.tws0 with offsets
removed, tracked in 0.1 m steps with optional ocelot 1D CSR over the whole line.  The input
beam is regenerated with load_test_parray, which is exactly what track_with_test_beam starts
from.  The small injected D_y (local_t20.tws0.Dy = -0.2 mm) is not imposed on the beam.

Both beams are saved next to this script so that plot_nina_t20_tracking.py can show
before/after.

Usage:
    python track_nina_t20.py --charge 250 --csr
    python track_nina_t20.py --charge 250 --no-csr
"""

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

from ocelot.cpbd.io import save_particle_array
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.transformations import SecondTM

import local_t20
import local_tl20
from felarc.tracking import load_test_parray, track_with_test_beam

HERE = Path(__file__).resolve().parent


def output_stem(outdir: Path, charge: int, csr: bool) -> Path:
    return outdir / f"nina-t20-{'csr' if csr else 'nocsr'}-{charge}pC"


def track_nina_t20(charge: int, csr: bool):
    mlat = MagneticLattice(
        local_tl20.cell + local_t20.cell, method={"global": SecondTM}
    )
    twiss0 = local_tl20.tws0

    parray0 = load_test_parray(twiss0, charge=charge)  # type: ignore[arg-type]
    # track_with_test_beam reads the lattice from a TBA's .mlat; give it just that.
    parray1 = track_with_test_beam(
        SimpleNamespace(mlat=mlat), twiss0, csr=csr, charge=charge  # type: ignore[arg-type]
    )
    return parray0, parray1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--charge", type=int, choices=(100, 250), default=250)
    parser.add_argument("--csr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outdir", type=Path, default=HERE)
    args = parser.parse_args()

    start = time.time()
    parray0, parray1 = track_nina_t20(args.charge, args.csr)
    print(f"Tracked {parray0.n} particles in {time.time() - start:.0f} s")

    stem = output_stem(args.outdir, args.charge, args.csr)
    save_particle_array(f"{stem}-parray0.npz", parray0)
    save_particle_array(f"{stem}-parray1.npz", parray1)
    print(f"Saved {stem}-parray{{0,1}}.npz")


if __name__ == "__main__":
    main()
