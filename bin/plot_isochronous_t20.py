


import matplotlib.pyplot as plt
from felarc.tracking import load_test_parray
from ocelot.cpbd.io import load_particle_array
from ocelot.gui.accelerator import compare_beams, show_e_beam

import local_tl20

TWISS0 = local_tl20.tws0


def compare_without_csr():
    parray0_250pC = load_test_parray(charge=250, twiss0=TWISS0)
    parray0_100pC = load_test_parray(charge=100, twiss0=TWISS0)


    parray1_no_csr_100pC = load_particle_array('isochronous-t20/isochronous-t20-csr=False_charge=100pC.npz')
    parray1_no_csr_250pC = load_particle_array('isochronous-t20/isochronous-t20-csr=False_charge=250pC.npz')
    parray1_csr_100pC = load_particle_array('isochronous-t20/isochronous-t20-csr=True_charge=100pC.npz')
    parray1_csr_250pC = load_particle_array('isochronous-t20/isochronous-t20-csr=True_charge=250pC.npz')
    

    compare_beams(parray0_250pC, parray1_no_csr_250pC,
                  legend_beam1="Start", legend_beam2="Stop",
                  title="250pC without CSR")

    plt.show()

    compare_beams(parray1_csr_250pC, parray1_no_csro_250pC,
                  legend_beam1="with CSR ", legend_beam2="Without",
                  title="250pC With/without CSR")

    plt.show()


    compare_beams(parray0_100pC, parray1_no_csr_100pC,
                  legend_beam1="Start", legend_beam2="Stop",
                  title="100pC without CSR")

    plt.show()

    compare_beams(parray1_csr_100pC, parray1_no_csr_100pC,
                  legend_beam1="with CSR ", legend_beam2="Without",
                  title="100pC With/without CSR")

    plt.show()
    

def main():
    compare_without_csr()


if __name__ == "__main__":
    main()
