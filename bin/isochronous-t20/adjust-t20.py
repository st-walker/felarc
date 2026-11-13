from typing import Literal

import matplotlib.pyplot as plt
from felarc.analysis import describe_arc_geometry, describe_arc_optics
from felarc.lattice import Line
from felarc.matching import close_dispersions, match_matrix_element
from felarc.plot import plot_optics, plot_r56_full, plot_principal_trajectories, plot_r51_full, plot_r52_full, plot_time_coupling_terms
from felarc.tracking import track_with_test_beam
from ocelot.cpbd.io import save_particle_array
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.transformations import SecondTM

# import local_t20
# import local_tl20
import local_t20x
import local_tl20x

# TWISS0 = local_tl20.tws0
TWISS0 = local_tl20x.tws0
# TLT20 = Line(local_tl20.cell + local_t20.cell) # type: ignore
TLT20 = Line(local_tl20x.cell + local_t20x.cell) # type: ignore


def close_r56(mlat):
    # intitial_guesses:
    TLT20['QL.2070.T20'].k1 *= 1.1
    TLT20['QL.2078.T20'].k1 *= 0.9

    # print(describe_arc_optics(mlat, twiss0=TWISS0))

    res = match_matrix_element(
        mlat, [TLT20["QL.2070.T20"], TLT20["QL.2078.T20"]], [[[4, 5], 0]]
    )

    plot_optics(mlat, twiss0=TWISS0)
    # plot_r56_full(mlat, twiss0=TWISS0)
    print(res)
    # plt.show()


def close_isochrony(mlat):
    # TLT20['QL.2070.T20'].k1 *= 1.1
    # TLT20['QL.2078.T20'].k1 *= 0.9

    # print(describe_arc_optics(mlat, twiss0=TWISS0))

    res = match_matrix_element(
        mlat, [TLT20["QL.2070.T20"],
               TLT20["QL.2074.T20"],
               TLT20["QL.2078.T20"]],
        [[[4, 5], 0],
         [[0, 5], 0],
         [[1, 5], 0]
         ]
    )

    return res
    

# def close_dispersion(mlat):
#     res = close_dispersions(mlat,
#                             [
#                                 TLT20["QL.2092.T20"],
#                                 TLT20["QL.2096.T20"],
#                                 TLT20["QL.2102.T20"],
#                                 TLT20["QL.2109.T20"],
#                              # TLT20["QL.2114.T20"],
#                              # TLT20["QL.2116.T20"]
#                              ],
#                             twiss0=TWISS0)


def track_stn10(mlat, *, csr: bool, charge: Literal[100, 250] = 250):
    parray1 = track_with_test_beam(mlat,
                                   TWISS0,
                                   csr=csr,
                                   charge=charge)
    save_particle_array(f"isochronous-t20/isochronous-t20-{csr=}_{charge=}pC.npz", parray1)

def main():
    mlat = MagneticLattice(TLT20# , method={"global": SecondTM}
                           ) # type: ignore
    # close_r56(mlat)
    # close_isochrony(mlat)
    # Vertical:
    # TLT20['QL.2070.T20'].k1 = 0.3354496
    # TLT20['QL.2078.T20'].k1 = 0.3366288

    # Horizontal only
    # TLT20['QL.2070.T20'].k1 = 0.33477881
    # TLT20['QL.2078.T20'].k1 = 0.34606583

    # plot_time_coupling_terms(mlat, 1000)
    # plot_principal_trajectories(mlat, 1000)
    # plot_r51_full(mlat)
    # plot_r52_full(mlat)
    # plt.show()
    

    # # close_dispersion(mlat)
    # print(describe_arc_optics(mlat, twiss0=TWISS0))

    # plot_optics(mlat, twiss0=TWISS0)
    plot_r56_full(mlat, twiss0=TWISS0)
    # # print(res)
    plt.show()
    # # track_stn10(mlat, charge=100, csr=True)
    # # track_stn10(mlat, charge=100, csr=False)
    # # track_stn10(mlat, charge=250, csr=True)    
    # # track_stn10(mlat, charge=250, csr=False)


if __name__ == "__main__":
    main()

