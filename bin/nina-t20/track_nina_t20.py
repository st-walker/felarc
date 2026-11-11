"""Track the S2E STN10 beam through the standard (Nina) TL20 + T20 model.

The lattice is the full model, local_tl20 + local_t20: vertical kickers, dipole tilts and
the sliced QK quad all included (not the flattened "x" variant used in
isochronous-t20/adjust-t20.py).

The input beam is the DESY start-to-end beam at STN10 (100 or 250 pC, downloaded and cached
by felarc.tracking.load_test_parray) with its Twiss matched to local_tl20.tws0 and its
offsets removed.  The small injected D_y (local_t20.tws0.Dy = -0.2 mm) is not imposed on
the beam.

Tracking uses second-order maps throughout, and optionally ocelot's 1D CSR from the start
to the end of the line.  Both the matched input and the output beams are saved next to this
script so that plot_nina_t20_tracking.py can show before/after.

Usage:
    python track_nina_t20.py --charge 250 --csr
    python track_nina_t20.py --charge 250 --no-csr
    python track_nina_t20.py --charge 250 --csr --nth 100   # quick run, 1 % of particles
"""

import argparse
import time
from pathlib import Path

from ocelot.cpbd.csr import CSR
from ocelot.cpbd.io import save_particle_array
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.navi import Navigator
from ocelot.cpbd.track import track
from ocelot.cpbd.transformations import SecondTM

import local_t20
import local_tl20
from felarc.tracking import load_test_parray

HERE = Path(__file__).resolve().parent


def output_stem(outdir: Path, charge: int, csr: bool, nth: int) -> Path:
    thinned = f"-nth{nth}" if nth > 1 else ""
    return outdir / f"nina-t20-{'csr' if csr else 'nocsr'}-{charge}pC{thinned}"


def track_nina_t20(charge: int, csr: bool, nth: int = 1, unit_step: float = 0.1):
    mlat = MagneticLattice(
        local_tl20.cell + local_t20.cell, method={"global": SecondTM}
    )

    parray0 = load_test_parray(local_tl20.tws0, charge=charge)  # type: ignore[arg-type]
    if nth > 1:
        parray0 = parray0.thin_out(nth=nth)

    navi = Navigator(mlat, unit_step=unit_step)
    if csr:
        navi.add_physics_proc(
            CSR(sigma_min=parray0.tau().std() * 0.1, rk_traj=True, energy=parray0.E),
            mlat.sequence[0],
            mlat.sequence[-1],
        )

    _, parray1 = track(mlat, parray0.copy(), navi, overwrite_progress=False)
    return parray0, parray1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--charge", type=int, choices=(100, 250), default=250)
    parser.add_argument("--csr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--nth", type=int, default=1, help="keep every nth particle (1 = full beam)"
    )
    parser.add_argument("--outdir", type=Path, default=HERE)
    args = parser.parse_args()

    start = time.time()
    parray0, parray1 = track_nina_t20(args.charge, args.csr, nth=args.nth)
    print(f"Tracked {parray0.n} particles in {time.time() - start:.0f} s")

    stem = output_stem(args.outdir, args.charge, args.csr, args.nth)
    save_particle_array(f"{stem}-parray0.npz", parray0)
    save_particle_array(f"{stem}-parray1.npz", parray1)
    print(f"Saved {stem}-parray{{0,1}}.npz")


if __name__ == "__main__":
    main()
