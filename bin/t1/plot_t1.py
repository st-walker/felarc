


import matplotlib.pyplot as plt
from felarc.analysis import describe_arc_geometry, describe_arc_optics
from felarc.plot import plot_optics, plot_r56_full, plot_time_coupling_terms
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.transformations import SecondTM

from t1 import cell as t1cell
from tl2 import cell as tl2cell
from tl2 import tws0 as TWISS0
from tl34_sa2 import cell as tl34_sa2cell

CELL = tl2cell + tl34_sa2cell + t1cell
CELL = tl2cell + tl34_sa2cell + t1cell

def main():
    # TWISS0.s = 0.0
    mlat = MagneticLattice(CELL)
    # mlat = MagneticLattice(CELL, method={"global": SecondTM})    
    # fig, axes = plot_optics(mlat, TWISS0)
    # axes[-1].set_xlim(2095, 2210)
    # fix, axes = plot_r56_full(mlat, TWISS0)
    fig, axes = plot_time_coupling_terms(mlat, 1000, s_offset=TWISS0.s)
    axes[-1].set_xlim(2095, 2210)


    # print(describe_arc_optics(mlat, twiss0=TWISS0))
    # print(describe_arc_geometry(mlat, twiss0=TWISS0))    
    plt.show()


if __name__ == "__main__":
    main()



# end of first big 3 set of dipoles:  3pi
# start of first negative dipole: 3.3pi
# end of first neg dip: 3.35
# start of second neg: 3.6
# enf of second neg: 3.65
# start of second big 3 set of dipoles: 3.86 pi

# gap between vertical dipoles: 0.2pi
