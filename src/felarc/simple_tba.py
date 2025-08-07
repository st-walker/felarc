from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from ocelot.cpbd.elements import Drift, Quadrupole, SBend
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.match import match
from ocelot.cpbd.optics import Twiss

from felarc.matching import match_twiss


@dataclass
class ArcDefinition:
    dipole_length: float
    dipole_angle: float
    l1: float
    l2: float
    l3: float
    qlength: float
    k11: float
    k12: float


def make_arc_to_inner_dipole_start(ad: ArcDefinition) -> MagneticLattice:
    sbend = SBend(l=ad.dipole_length, angle=ad.dipole_angle)
    d1 = Drift(l=ad.l1)
    d2 = Drift(l=ad.l2)
    d3 = Drift(l=ad.l3)
    qlength = ad.qlength
    qf1 = Quadrupole(l=qlength, k1=ad.k11) # type: ignore (must be fixed on ocelot-side...)
    qd1 = Quadrupole(l=qlength, k1=ad.k12) # type: ignore 

    return MagneticLattice(  # type: ignore (must be fixed on ocelot side..)
        [
            sbend,
            d1,
            qf1,
            d2,
            qd1,
            d3
        ]
    )

def get_quad_strengths(ad: ArcDefinition) -> ArcDefinition:
    # The book suggests varying a drift and a quad strength to match the two conditions
    # But here instead I'm going to use two quads to reach the condition.
    mlat = make_arc_to_inner_dipole_start(ad)
    dx, dxp = get_isochronous_arc_dispersion_condition(ad.dipole_length, ad.dipole_angle)

    twiss0 = Twiss(beta_x=20, beta_y=10)
    result = deepcopy(ad)
    # Get the quads from the lattice.
    quads = [ele for ele in mlat.sequence if isinstance(ele, Quadrupole)]
    quads[0].k1 = -1
    quads[1].k1 = 1
    result = match_twiss(
        mlat,
        verbose=False,
        max_iter=1_000_000,
        constr={mlat.sequence[-1]: {"Dx": dx, "Dxp": dxp}},
        vars=quads,
        tw=twiss0,
    )
    return result # type: ignore


def make_arc(ad: ArcDefinition) -> MagneticLattice:
    sbend = SBend(l=ad.dipole_length, angle=ad.dipole_angle)
    d1 = Drift(l=ad.l1)
    d2 = Drift(l=ad.l2)
    d3 = Drift(l=ad.l3)
    qlength = ad.qlength
    qf = Quadrupole(l=qlength, k1=ad.k11)  # type: ignore
    qd = Quadrupole(l=qlength, k1=ad.k12)  # type: ignore

    return MagneticLattice( # type: ignore
        [
            sbend,
            d1,
            qf,
            d2,
            qd,
            d3,
            sbend,
            d3,
            qf,
            d2,
            qd,
            d1,
            sbend
        ]
    )

def get_isochronous_arc_dispersion_condition(dipole_length: float, dipole_angle: float) -> tuple[float, float]:
    bending_radius = dipole_length / dipole_angle
    # Note: the book does not have this factor of -1 (i.e. -1.5 * ...).
    # In the book a positive bend seems to give a negative dispersion.  So
    # This is jsut a matter of convention.  I introduce the negative sign here
    # because with a positive dispersion in the first dipole, we of course need
    # a negative one in the middle dipole to get the R56 integral to sum to 0.
    dxp = - ((3/2) * dipole_angle - np.sin(dipole_angle))
    # Is this also correct?
    dx = bending_radius * ((np.tan(dipole_angle / 2.0)**-1) * dxp + 1)
    return dx, dxp
