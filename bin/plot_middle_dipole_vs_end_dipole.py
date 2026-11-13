
import matplotlib.pyplot as plt
from ocelot.cpbd.io import load_particle_array
from ocelot.gui.accelerator import compare_beams


def main():
    parray_end = load_particle_array("tba-tracking/tba-tracking-with-csr-250pC.npz")
    parray_middle = load_particle_array("tba-tracking/tba-tracking-without-csr-250pC-entrance-to-middle-dipole.npz")

    compare_beams(parray_middle, parray_end, legend_beam1="End of TBA", legend_beam2="Start Dipole 2")

    plt.show()


if __name__ == "__main__":
    main()

