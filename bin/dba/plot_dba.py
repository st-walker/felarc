


import matplotlib.pyplot as plt
import numpy as np
from felarc.dba import (DBADefinition, csr_achromat_c2c_matrix,
                        csr_dipole_radius_ratio, csr_optical_ratio, make_dba,
                        symmetric_dba_beta0)
from felarc.matching import match_matrix_element, match_twiss
from felarc.plot import plot_csr_dispersion, plot_optics
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.optics import Twiss, twiss

ENERGY = 14

def main():
    dipole_angle = np.deg2rad(3)
    dipole_length = 1
    dipole_bending_radius = dipole_length / dipole_angle
    dbad = DBADefinition(l_bend1=dipole_length,
                         angle1=dipole_angle,
                         k1_quad1=2,
                         k1_quad2=-1,
                         l_quad=0.5,
                         l_drift1=1.0+2.5,
                         l_drift2=1.0+2.5,
                         l_drift0=0.0)

    dba = make_dba(dbad)

    dba.mlat

    desired_2x2 = csr_achromat_c2c_matrix(dipole_angle, dipole_angle, dipole_bending_radius, 0, 0, 10)
    dipole_ratio = csr_dipole_radius_ratio(dipole_angle, dipole_angle)
    optical_ratio = csr_optical_ratio(dipole_angle, dipole_angle, dipole_bending_radius)

    # Optics:
    # We know we must have equal beta1 and beta2 (in centres), because we have equal angles.
    # Let's pick beta1=beta2=10m.

    alpha0 = np.sqrt(24.001) * 2
    beta01 = symmetric_dba_beta0(dipole_length, alpha0)[0]
    # beta02 = symmetric_dba_beta0(dipole_length, alpha0)[1]

    twiss0 = Twiss(beta_x=beta01, alpha_x=alpha0, alpha_y=-alpha0, beta_y=beta01, E=ENERGY)

    twisses = twiss(dba.mlat, twiss0, return_df=True)
    twiss1 = twisses[twisses.id == "mid_bend1"]
    beta1 = twiss1.beta_x
    alpha1 = twiss1.alpha_x

    c2c = dba.mlat.get_sequence_part(dba.markers.mid_bend1, dba.markers.mid_bend2)

    mlatc2c = MagneticLattice(c2c)

    result = match_twiss(
         dba.mlat,
         verbose=True,
         max_iter=1_000_000,
         constr={
             # tba.markers.middle_dipole_start: {"Dx": dx, "Dxp": dxp},
             dba.markers.mid_bend2: {"alpha_x": -alpha0, "beta_x": beta01}
                   },
            vars=dba.quadrupoles,
            twiss0=twiss0,
            method="simplex"
        )


    result = match_matrix_element(
         mlatc2c,
         dba.quadrupoles,
         [((1, 0), 12/dipole_length), ((0, 0), -1),  ((1, 1), -1)]
        )
    
    plot_optics(dba.mlat, twiss0=twiss0)
    plot_csr_dispersion(dba.mlat, twiss0)

    from IPython import embed; embed()

    plt.show()

    pass

if __name__ == "__main__":
    main()
