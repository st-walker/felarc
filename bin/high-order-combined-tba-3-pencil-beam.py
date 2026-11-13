from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from felarc.tba import TBADefinition, make_tba
from felarc.tracking import load_test_parray, make_pencil_beam
from ocelot.cpbd.csr import CSR
from ocelot.cpbd.io import save_particle_array
from ocelot.cpbd.navi import Navigator
from ocelot.cpbd.optics import twiss
from ocelot.cpbd.physics_proc import SaveBeam
from ocelot.cpbd.track import track
from ocelot.gui.accelerator import show_e_beam

NFIG = 0

def main():
    tbad = TBADefinition(
        l_bend=0.4,
        angle_bend=tuple(np.deg2rad([2, 2, 2])),
        l_quad=0.2,
        k1_quad1=-8.889,
        k1_quad2=9.293,
        k1_quad3=7.480,
        l_sext=0.3,
        k2_sextf=321.474,
        k2_sextd=-551.934,
        l_drift1=0.817,
        l_drift2=0.331,
        l_drift3=0.102,
        l_drift4=1.315,
        l_drift5=0.125,
        l_drift6=0.139,
        l_drift7=0.370,
    )

    tba = make_tba(tbad)
    mlat = tba.mlat
    twiss0 = tba.twiss0()

    # show_e_beam(parray0)
    plt.show()

    track_pencil(tba, twiss0, csr=True, charge=250, out="tba-tracking/tba-tracking-rectangle-with-csr-250pC") # type: ignore
    track_pencil(tba, twiss0, csr=False, charge=250, out="tba-tracking/tba-tracking-rectangle-without-csr-250pC") # type: ignore

    track_pencil(tba, twiss0, csr=True, charge=100, out="tba-tracking/tba-tracking-rectangle-with-csr-100pC") # type: ignore
    track_pencil(tba, twiss0, csr=False, charge=100, out="tba-tracking/tba-tracking-rectangle-without-csr-100pC") # type: ignore

#     plt.show()

def track_pencil(tba, twiss0, *, csr: bool, charge: Literal[100, 250] = 250, out: str = ""):
    global NFIG
    mlat = tba.mlat

    # parray0 = load_test_parray(twiss0, charge)
    parray0 = make_pencil_beam(twiss0=twiss0)

    # from IPython import embed; embed()

    outpath = Path(out)
    dump = tba.markers.mid_bend2
    # dump = next(ele for ele in mlat.sequence if ele.id == "start_bend2")
    navi = Navigator(mlat, unit_step=0.1) # type: ignore

    parray0_path = outpath.with_stem(f"{outpath.stem}-parray0.npz")
    parray1_path = outpath.with_stem(f"{outpath.stem}-parray1.npz")
    save_particle_array(parray0_path, parray0)
    # middle_dipole_path = f"{}"
    middle_dipole_path = outpath.with_stem(f"{outpath.stem}-middle-of-middle-dipole.npz")
    navi.add_physics_proc(SaveBeam(middle_dipole_path), dump, dump)
    print("Wrote", middle_dipole_path)

    if csr:
        navi.add_physics_proc(CSR(sigma_min=parray0.tau().std() * 0.1,
                                  rk_traj=True,
                                  energy=parray0.E),
                              mlat.sequence[0],
                              mlat.sequence[-1])

    # title = "After Tracking WITH csr" if csr else "After tracking NO CSR"
    _, parray1 = track(mlat, parray0.copy(), navi, overwrite_progress=False)
    # show_e_beam(parray1, title=title, nfig=NFIG)
    if out:
        save_particle_array(parray1_path, parray1)
        print("Wrote", out)

    NFIG += 1


if __name__ == "__main__":
    main()
