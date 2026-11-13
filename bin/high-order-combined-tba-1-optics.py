import latdraw
import matplotlib.pyplot as plt
import numpy as np
from felarc.analysis import describe_arc_geometry, describe_arc_optics
from felarc.csr import gaussian_bunch_emittance_growth
from felarc.optics import transfer_maps_z
from felarc.plot import (plot_csr_dispersion,
                         plot_m11_m21_emittance_growth_contours, plot_optics,
                         plot_r56_full)
from felarc.tba import SecondOrderTBAParameters
from latdraw.interfaces import lattice_from_ocelot
from latdraw.plot import s_label
from ocelot.cpbd.optics import twiss


def main():
    tbad = SecondOrderTBAParameters(
        l_bend=0.4,
        angle_bend=tuple(np.deg2rad([2.2, 2.2, 2.2])),
        l_quad=0.2,
        k1_quad1=7.480,
        k1_quad2=-8.889,
        k1_quad3=9.293,
        l_sext=0.3,
        # k2_sextf=0,
        # k2_sextd=0,
        k2_sextf=321.474,
        k2_sextd=-551.934,
        l_drift0=0.817,
        l_drift1=0.331,
        l_drift2=0.102,
        l_drift3=1.315,
        l_drift4=0.125,
        l_drift5=0.139,
        l_drift6=0.370,
    )

    tba = tbad.make_tba()
    # tba= make_tba(tbad)
    mlat = tba.mlat
    print(tba.mx2x2_between_dipoles())
    print(tba.my2x2_between_dipoles())    

    llat = lattice_from_ocelot(mlat)

    # fig, ax = latdraw.subplots_with_lattices([llat, None, None, None, None])
    # (_, ax0, ax1, ax2, ax3) = ax
    # z, rmatrices = transfer_maps_z(mlat, 0.01)
    # rend = rmatrices[-1]

    # from IPython import embed; embed()
    # inverses = np.linalg.inv(rmatrices)
    # r_to_end = rend @ inverses
    # # r_to_end2 = rend @ inverses @ np.linalg.inv(rend)

    # # gaussian_bunch_emittance_growth(tba.mlat, 1e-6, 4e-6, 1)
    # gaussian_bunch_emittance_growth(tba.mlat, 1e-6, 250e-12, 4e-6, 1)

    # from IPython import embed; embed()

    # for rmatrix in rmatrices:
    #     rend @ np.lrmatrix.inv()

    # r_16_s_to_end = rend[0]
    
    # from IPython import embed; embed()
    # ax0.plot(z, rmatrices[:, 0, 5])
    # ax1.plot(z, r_to_end[:, 0, 5])
    # ax2.plot(z, rmatrices[:, 1, 5])
    # ax3.plot(z, r_to_end[:, 1, 5])
    # print(np.trapezoid(r_to_end[:, 0, 5], x=z))
    # print(np.trapezoid(r_to_end[:, 1, 5], x=z))

    # ax0.set_ylabel(r"$R_{16}$")
    # ax1.set_ylabel(r"$R_{16}^{s\rightarrow s_f}$")
    # ax2.set_ylabel(r"$R_{26}$")
    # ax3.set_ylabel(r"$R_{26}^{s\rightarrow s_f}$")

    # # from IPython import embed; embed()

    # # plot_m11_m21_emittance_growth_contours(tba, 1e-6, 100e-12, 5e-6, 16)

    # # s_label(ax3)

    # # plt.plot(z, r_to_end2[:, 0, 5])
    # plt.show()
    # # import sys
    # sys.exit()

    # plot_csr_dispersion(mlat)
    # _, R, _, s =  mlat.transfer_maps(energy=16, # type: ignore
    #                                  output_at_each_step=True)

    # plt.show()

    print(describe_arc_geometry(mlat))
    print(describe_arc_optics(mlat))
    plot_r56_full(mlat)
    plot_optics(mlat)
    plt.show()

if __name__ == "__main__":
    main()
