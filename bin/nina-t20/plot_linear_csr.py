from felarc.plot import plot_csr_dispersion
from ocelot.cpbd.magnetic_lattice import MagneticLattice

import local_t20
import local_tl20

TWISS0 = local_tl20.tws0
CELL = local_tl20.cell + local_t20.cell


def main():
    plot_csr_dispersion(MagneticLattice(CELL))
    plt.show()


if __name__ == "__main__":
    main()
