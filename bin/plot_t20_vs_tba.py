
import matplotlib.pyplot as plt
from ocelot.cpbd.io import load_particle_array
from ocelot.gui.accelerator import compare_beams


def main():
    parray_t20 = load_particle_array("/Users/stuartwalker/repos/t20/bin/section_STN10_250pC-without-csr-end-of-t20.npz")
    parray_tba = load_particle_array("/Users/stuartwalker/repos/felarc/notebooks/section_STN10_250pC-without-csr-end-of-tba.npz")

    compare_beams(parray_t20, parray_tba, legend_beam1="T20", legend_beam2="TBA")

    plt.show()


if __name__ == "__main__":
    main()
