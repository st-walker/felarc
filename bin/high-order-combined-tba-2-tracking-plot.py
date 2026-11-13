from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from felarc.tba import TBADefinition, make_tba
from felarc.tracking import load_test_parray
from ocelot.cpbd.csr import CSR
from ocelot.cpbd.io import load_particle_array, save_particle_array
from ocelot.cpbd.navi import Navigator
from ocelot.cpbd.optics import twiss
from ocelot.cpbd.physics_proc import SaveBeam
from ocelot.cpbd.track import track
from ocelot.gui.accelerator import compare_beams, show_e_beam

DIR = Path("tba-tracking")


def main():

    # parray0 = load_particle_array(DIR / "tba-tracking-rectangle-with-csr-100pC-parray0.npz")
    # parray1 = load_particle_array(DIR / "tba-tracking-rectangle-with-csr-100pC-parray1.npz")
    # parraym = load_particle_array(DIR / "tba-tracking-rectangle-with-csr-100pC-middle-of-middle-dipole.npz")

    parray0 = load_particle_array(DIR / "tba-tracking-rectangle-without-csr-100pC-parray0.npz")
    parray1 = load_particle_array(DIR / "tba-tracking-rectangle-without-csr-100pC-parray1.npz")
    parraym = load_particle_array(DIR / "tba-tracking-rectangle-without-csr-100pC-middle-of-middle-dipole.npz")

    

    # compare_beams(parraym, parray1, legend_beam1="Middle", legend_beam2="End")
    # compare_beams(parray0, parray1, legend_beam1="Start", legend_beam2="End")
    compare_beams(parray0, parraym, legend_beam1="Start", legend_beam2="Middle")

    # show_e_beam(parray0)


    # parray0_100pC = load_test_parray(twiss0=tba.twiss0(), charge=100)
    # parray0_250pC = load_test_parray(twiss0=tba.twiss0(), charge=250)

    # # parray0_100pC = load_particle_array(DIR / "tba-tracking-with-csr-100pC.npz")
    # parray1_100pC = load_particle_array(DIR / "tba-tracking-with-csr-100pC.npz")
    # parray1_250pC = load_particle_array(DIR / "tba-tracking-with-csr-250pC.npz")

    # # sho
    # # compare_beams(parray0_100pC, parray1_100pC, legend_beam1="Before", legend_beam2="After")
    # compare_beams(parray0_250pC, parray1_250pC, legend_beam1="Before", legend_beam2="After")

    # show_e_beam(parray1_100pC)

    # compare_beams()
    # track_stn10(mlat, twisses[0], csr=True, charge=250, out="TBA-tracking-with-csr-250pC.npz") # type: ignore
    # track_stn10(mlat, twisses[0], csr=False, charge=250, out="TBA-tracking-without-csr-250pC.npz") # type: ignore
    
    # track_stn10(mlat, twisses[0], csr=True, charge=100, out="TBA-tracking-with-csr-100pC.npz") # type: ignore
    # track_stn10(mlat, twisses[0], csr=False, charge=100, out="TBA-tracking-without-csr-100pC.npz") # type: ignore

    plt.show()


if __name__ == "__main__":
    main()
