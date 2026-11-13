


import matplotlib.pyplot as plt
from felarc.analysis import describe_arc_optics
from felarc.lattice import Line
from felarc.matching import close_dispersions, match_matrix_element
from felarc.plot import plot_optics, plot_r56_full
from ocelot.cpbd.elements import Drift, Quadrupole, SBend
from ocelot.cpbd.magnetic_lattice import MagneticLattice

from local_tl20 import tws0

# l = rho * theta
# 1/rho = theta / l
# theta = (1/rho) * l

def main():
    # d1 = Drift(l=1)
    # d2 = Drift(l=0.6)
    # d3 = Drift(l=0.4)
    # m0 = SBend(l=1, angle=+1.030)
    # m1 = 
    # m2 = SBend(l=1, angle=-0.293)
    # q1 = Quadrupole(l=1, k1=+1.409)
    # q2 = Quadrupole(l=1, k1=-0.617)

    sign = -1

    sequence = Line([Drift(l=1),
                SBend(l=1, angle=sign*-1.030),
                Drift(l=0.6),
                Quadrupole(l=1, k1=sign*-1.409, eid="quad1"),
                Drift(l=0.4),
                Quadrupole(l=1, k1=sign*+0.617, eid="quad2"),
                Drift(l=0.6),
                SBend(l=1, angle=sign*-0.739),
                Drift(l=0.4),
                SBend(l=1, angle=sign*+0.293)])
    


    mlat = MagneticLattice(sequence)


    # res = match_matrix_element(
    #     mlat, [sequence["quad1"], sequence["quad2"]], [[[4, 5], 0]]
    # )

    # res = close_dispersions(
    #     mlat, [sequence["quad1"], sequence["quad2"]], twiss0=tws0
    # )
    
    
    
    print(describe_arc_optics(mlat, twiss0=tws0))

    plot_r56_full(mlat, twiss0=tws0)
    plt.show()


if __name__ == "__main__":
    main()
