from sympy.solvers import nsolve
from sympy import Symbol

from ocelot import *
from ocelot.gui import *
import matplotlib.pyplot as plt
import numpy as np
from felarc.simple_tba import get_tba_dispersion_condition, get_quad_strengths, TBADefinition, make_tba, TripleBendAchromat
from felarc.plot import plot_optics, plot_r56_full
from felarc.t20 import ARC_ANGLE_RAD
from felarc.maps import sequence_to_map, OpticalParameters, thin_quadrupole, drift, sbend


DIPOLE_ANGLE = ARC_ANGLE_RAD / 3
DIPOLE_LENGTH = 3


def main():
    dx, dxp = get_tba_dispersion_condition(dipole_length=DIPOLE_LENGTH, dipole_angle=DIPOLE_ANGLE)


    print(dx, dxp)

    ad = TBADefinition(dipole_length=DIPOLE_LENGTH, dipole_angle=DIPOLE_ANGLE,
                       l1=1., l2=1, l3=2.,
                       qlength=1e-6,
                       k11=1, k12=-1)
    tba = make_tba(ad)
    k11, k12 = get_quad_strengths(ad)

    start_index = 0
    # stop_index = start_index+2
    stop_index = tba.mlat.sequence.index(tba.markers.middle) + 1
    tbamap = sequence_to_map(tba.mlat.sequence[start_index:stop_index],
                             14,
                             variables=[tba.components.q1, tba.components.q2])

    # print(tbamap.rmat)

    # target_optics = OpticalParameters(beta_x=10, alpha_x=0, beta_y=10, alpha_y=0, dx=0, dy=0)
    input_optics = OpticalParameters(beta_x=10, alpha_x=1, beta_y=20, alpha_y=-1, dx=0, dy=0)

    output_twiss = tbamap * input_optics

    # repr(output_twiss.alpha_x.expand())


    # Solve system of equations for alpha_x and alpha_y
    # solution = solve([output_twiss.alpha_x, output_twiss.alpha_y],
    #                   output_twiss.alpha_x.free_symbols,
    #                   check=True,
    #                   warn=True)
    # print(solutions)
    # from IPython import embed; embed()
    solutions = nsolve(
        (output_twiss.alpha_x, output_twiss.alpha_y),
        list(output_twiss.alpha_x.free_symbols),
        (0.0, 0.0),
        dict=True
    )

    # (symbol.name, value) for symbol, value in zip(list(output_twiss.alpha_x.free_symbols), 

    for solution in solutions:
        print(f"Checking Solution: {solution}")
        print("Alpha_x = ",output_twiss.alpha_x.subs(solution))
        print("Alpha_y = ",output_twiss.alpha_y.subs(solution))
        print("Beta_x = ",output_twiss.beta_x.subs(solution))
        print("Beta_y = ",output_twiss.beta_y.subs(solution))
        print("Dx", output_twiss.dx.subs(solution))
        print("Dx'", output_twiss.dxp.subs(solution))        

    kl_q1 = float(solutions[0][Symbol("K_q1", real=True)])
    kl_q2 = float(solutions[0][Symbol("K_q2", real=True)])

    # from IP
    check_twiss_maps(tba, input_optics)
    check_with_ocelot(tba, input_optics.to_twiss(), kl_q1, kl_q2)

    # from IPython import embed; embed()
    # print(alpha_x

    

def check_with_ocelot(tba, twiss0, kl_q1, kl_q2):
    tba.components.quads[0].k1l = kl_q1
    tba.components.quads[1].k1l = kl_q2

    stop = tba.mlat.sequence.index(tba.markers.middle) + 1

    tba.mlat.sequence = tba.mlat.sequence[:stop]

    twisses = twiss(tba.mlat, tws0=twiss0, nPoints=1000)

    plot_opt_func(tba.mlat, twisses, top_plot=["Dx", "Dxp"])
    plt.show()


def check_twiss_maps(tba, input_optics):
    twiss0 = input_optics.to_twiss()
    tba.mlat.sequence = tba.mlat.sequence
    # from IPython import embed; embed()
    full_sequence = tba.mlat.sequence
    print(input_optics)
    for i, _ in enumerate(tba.mlat.sequence, start=1):
        print("\n\n\n")
        print("--------------")
        tba.mlat.sequence = tba.mlat.sequence[:i]
        print(tba.mlat.sequence[-1])
        twisses = twiss(tba.mlat, twiss0)

        # from IPython import embed; embed()
        # print("Ocelot")
        end_twiss = twisses[-1]
        print(f"OCELOT, beta_x={end_twiss.beta_x}, alpha_x={end_twiss.alpha_x}, beta_y={end_twiss.beta_y}, alpha_y={end_twiss.alpha_y}")

        # print("MY Matrices:")
        smap = sequence_to_map(tba.mlat.sequence, 14)
        print("MINE:", smap * input_optics)

        tba.mlat.sequence = full_sequence

        # twissmat = get_cs_transport_matrix(smap.rmat)



        
    # from IPython import embed; embed()
    


if __name__ == "__main__":
    main()

# MagneticLattice([SBend(l=3, angle=DIPOLE_ANGLE),
#                  Drift(l=1),
#                  Quadrupole(l=1e-6, k1=k1*1e6),
#                  # Drift(l=1),
#                  # Quadrupole(l=1e-6, k1=k2*1e6)
#                 ]).transfer_maps(energy=14)[1]
# # MagneticLattice([SBend(l=3, angle=DIPOLE_ANGLE), Drift(l=1)]).transfer_maps(energy=14)[1]
