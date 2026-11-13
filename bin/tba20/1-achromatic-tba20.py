from sys import exit

import matplotlib.pyplot as plt
import numpy as np
from felarc.analysis import describe_arc_geometry, describe_arc_optics
from felarc.matching import match_matrix_element
from felarc.plot import (plot_csr_dispersion, plot_optics, plot_r56_full,
                         plot_tba_periodic_beta0, plot_m11_m21_tba_contours)
from felarc.tba import MagneticSequence  # , FirstOrderTBAParameters
from felarc.tba import (FirstOrderTBAParameters, SecondOrderTBAParameters,
                        csr_isochronous_crossings_from_tba,
                        check_achromat_condition, match_achromatic_condition)
from felarc.tracking import (load_test_parray, make_pencil_beam,
                             track_with_test_beam)
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.optics import twiss
from ocelot.gui.accelerator import compare_beams, show_e_beam


def main():
    # scale_factor = 50/10
    scale_factor = 40 / (10 - 2*0.817)  # External drifts.
    scale = 1
    # tbad = TBADefinition(
    #     l_bend=0.4*scale_factor,
    #     angle_bend=tuple(np.deg2rad([2.3, 2.3, 2.3])),
    #     l_quad=scale_factor*0.2,
    #     k1_quad1=-0.1, # middle quad.
    #     k1_quad2=0.19,  # quad closest to centre.
    #     k1_quad3=0.1, # Quad furthest out.
    #     l_sext=scale_factor*0.3,
    #     k2_sextf=0,
    #     k2_sextd=0,
    #     l_drift1=0.817,
    #     l_drift2=scale_factor*0.331,
    #     l_drift3=scale_factor*0.102,
    #     l_drift4=scale_factor*1.315,
    #     l_drift5=scale_factor*0.125,
    #     l_drift6=scale_factor*0.139,
    #     l_drift7=scale_factor*0.370,
    # )

    # scale_factor = 2.0

    # tbad = TBADefinition(
    #     l_bend=scale_factor*0.4,
    #     angle_bend=tuple(np.deg2rad([2, 2, 2])),
    #     l_quad=scale_factor*0.2,
    #     k1_quad1=scale_factor*-8.889,
    #     k1_quad2=scale_factor*9.293,
    #     k1_quad3=scale_factor*7.480,
    #     l_sext=scale_factor*0.3,
    #     # k2_sextf=0,
    #     # k2_sextd=0,
    #     k2_sextf=scale_factor*321.474,
    #     k2_sextd=scale_factor*-551.934,
    #     l_drift1=scale_factor*0.817/scale_factor,
    #     l_drift2=scale_factor*0.331,
    #     l_drift3=scale_factor*0.102,
    #     l_drift4=scale_factor*1.315,
    #     l_drift5=scale_factor*0.125,
    #     l_drift6=scale_factor*0.139,
    #     l_drift7=scale_factor*0.370,
    # )

    # scale_factor = 4.0
    scale_factor = 40 / (10 - 2*0.817)  # External drifts.
    # scale_factor = 1.05
    tbad = SecondOrderTBAParameters(
        l_bend=scale_factor*0.4,
        # angle_bend=tuple(np.deg2rad([2.3, 2.3, 2.3])),
        angle_bend=tuple(np.deg2rad([2.3, 2.3, 2.3])),
        l_quad=scale_factor*0.2,
        k1_quad1=7.480/scale_factor,
        k1_quad2=-8.889/scale_factor,
        k1_quad3=9.293/scale_factor,
        l_sext=scale_factor*0.3,
        # k2_sextf=0,
        # k2_sextd=0,
        k2_sextf=321.474,
        k2_sextd=-551.934,
        l_drift0=scale_factor*0.817/scale_factor,
        l_drift1=scale_factor*0.331,
        l_drift2=scale_factor*0.102,
        l_drift3=scale_factor*1.315,
        l_drift4=scale_factor*0.125,
        l_drift5=scale_factor*0.139,
        l_drift6=scale_factor*0.370,
    )

    tbad = tbad.to_first_order()

    tba = tbad.make_tba()
    # tba = make_tba(tbad)
    # plot_tba_periodic_beta0(tba)
    # plot_m11_m21_tba_contours(tba)


    plt.show()
    mlat = tba.lat

    opt = match_achromatic_condition(tba, k1_quads=tba.quads, position_quads=[tba.quads[0]])

    # from IPython import embed; embed()

    tba.update_quads(opt)

    opt = match_achromatic_condition(tba,
                                     k1_quads=tba.quads,
                                     position_quads=tba.quads[:1],
                                     # match_isochronous=True,
                                     # force_match_m11_m21=(-0.85, 1.2) # XXX !!!!
                                     )

    tba.update_quads(opt)

    # # from IPython import embed; embed()
    # opt = match_achromatic_condition(tba, k1_quads=tba.quads, position_quads=tba.quads[:2], cancel_csr=True, match_isochronous=True)

    # tba.update_quads(opt)
    
    # print("CANCELLING CSR!!!")
    # opt = match_achromatic_condition(tba, k1_quads=tba.quads, position_quads=tba.quads, cancel_csr=True)
    # tba.update_quads(opt)

    
    # parray0 = make_pencil_beam(n=500_000, twiss0=tba.twiss0())
    # parray0 = load_test_parray(tba.twiss0(), charge=100)
    # # show_e_beam(parray0)
    # parray1 = track_with_test_beam(tba.mlat, tba.twiss0(), csr=True, charge=100, rectangle=False)
    # show_e_beam(parray1)
    # plt.show()
    # compare_beams(parray0, parray1, legend_beam1="Start", legend_beam2="End")

    # # plt.show()
    plot_optics(mlat)

    plt.show()

    # from IPython import embed; embed()
    # print(tba.mx2x2_between_dipoles())
    # print(tba.traces())
    # k1lim = 0.4
    # k1_quad1 = np.linspace(-k1lim, k1lim, num=30)
    # k1_quad3 = np.linspace(-k1lim, k1lim, num=30)
    # k1_quad1_grid, k1_quad3_grid = np.meshgrid(k1_quad1, k1_quad3)
    # twisses = twiss(mlat)
    # res = match_achromatic_condition(tba, tba.quads, [tba.quads[1]])
    plot_m11_m21_tba_contours(tba, m11_range=(-4, 2), m21_range=(-2, 1.5), opt=opt)
    # plot_tba_periodic_beta0(tba, m11_range=(-2.050, -1.925), m21_range=(-0.025, 0.1), opt=opt)
    plt.show()
    # exit()
    # from IPython import embed; embed()
    # match_stable(tba, [tba.quads[0], tba.quads[2]])
    # for row in k1_quad1:
    #     for
    # tba.quads[0].k1 = -0.2
    # tba.quads[2].k1 = 0.1
    # for k1 in np.linspace(-0.2, 0.2):
    #     tba.quads[1].k1 = k1
    #     print(tba.traces())
    # import sys
    # sys.exit()
    # def f(k1_quad1, k1_quad3):
    #     tba.quads[0].k1 = k1_quad1
    #     tba.quads[2].k1 = k1_quad3
    #     # x, y = tba.traces()
    #     return tba.traces()
    # f_vec = np.frompyfunc(f, 2, 2)
    # xtraces, ytraces = f_vec(k1_quad1_grid, k1_quad3_grid)
    # X = k1_quad1_grid
    # Y = k1_quad3_grid
    # Zxtrace = xtraces.astype(np.float64)
    # Zytrace = ytraces.astype(np.float64)
    # trace = Zytrace
    # # from IPython import embed; embed()
    # fig, ax = plt.subplots()
    # cp = ax.contourf(X, Y, trace, levels=50, cmap='viridis')
    # ax.contour(X, Y, np.abs(trace), levels=[2], colors='red', linewidths=2)
    # fig.colorbar(cp, label='Z')
    # ax.set_xlabel('$K_1^{(1)}$')
    # ax.set_ylabel('$K_1^{(3)}$')
    # # ax.set_title('Hover to see (x, y, Z)')
    # # Interactive hover: updates title with Z at nearest point
    # def on_move(event):
    #     if not event.inaxes:
    #         return
    #     xm, ym = event.xdata, event.ydata
    #     if xm is None or ym is None:
    #         return
    #     # find nearest grid point
    #     i = np.abs(k1_quad1 - xm).argmin()
    #     j = np.abs(k1_quad3 - ym).argmin()
    #     zval = Zytrace[j, i]
    #     ax.set_title(f"x={xm:.2f}, y={ym:.2f}, Z={zval:.3f}")
    #     fig.canvas.draw_idle()
    # fig.canvas.mpl_connect('motion_notify_event', on_move)
    # plt.show()
    # # from IPython import embed; embed()
    # fig, ax = plt.subplots()
    # im = ax.pcolormesh(k1_quad1_grid, k1_quad3_grid, xtrace)
    # fig.colorbar(im=im, ax=ax)
    # from IPython import embed; embed()
    # check_achromat_condition(tba)
    # achromatic_m12_m22
    # result = match_twiss(
    #      dba.mlat,
    #      verbose=True,
    #      max_iter=1_000_000,
    #      constr={
    #          # tba.markers.middle_dipole_start: {"Dx": dx, "Dxp": dxp},
    #          dba.markers.mid_bend2: {"alpha_x": -alpha0, "beta_x": beta01}
    #                },
    #         vars=dba.quadrupoles,
    #         twiss0=twiss0,
    #         method="simplex"
    #     )
    # result = match_matrix_element(
    #      mlatc2c,
    #      dba.quadrupoles,
    #      [((1, 0), 12/dipole_length), ((0, 0), -1),  ((1, 1), -1)]
    #     )
    

    
    plot_csr_dispersion(mlat)

    # plt.show()

    # print(twisses[0]) # type: ignore

    print(describe_arc_geometry(mlat))
    print(describe_arc_optics(mlat))
    # plot_r56_full(mlat)
    # plot_optics(mlat)
    plt.show()

if __name__ == "__main__":
    main()
