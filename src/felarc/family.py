from copy import deepcopy
from dataclasses import dataclass
from copy import deepcopy


import numpy as np
from ocelot.cpbd.beam import Twiss
from ocelot.cpbd.elements import Drift, Marker, Quadrupole, SBend
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.match import match


@dataclass
class ArcDefinition:
    outer_angle: float
    outer_length: float
    inner_angle: float
    inner_length: float
    l1: float
    l2: float
    l3: float
    quad_length: float
    k1_outer: float
    k1_inner: float


def _cot(x):
    return 1 / np.tan(x)


def get_isochronous_dispersions_at_middle_dipole_entrance(
    outer_angle: float, outer_length: float, inner_angle: float, inner_length: float
) -> tuple[float, float]:
    r"""
    $D_j = \rho_2\left [ D_j' \cot{(\phi_2) + 1} \right ]$

    $D'_j = -\frac{\rho_1}{\rho_2} \left ( \frac{3}{2} \phi_1 - \sin\phi_1 \right )$
    """
    rho_outer = outer_length / outer_angle
    rho_inner = inner_length / inner_angle
    dxp = -(rho_outer / rho_inner) * (1.5 * outer_angle - np.sin(outer_angle))
    dx = rho_inner * (dxp * _cot(inner_angle / 2.0) + 1)
    return dx, dxp


def get_quad_strengths(arc_definition: ArcDefinition) -> ArcDefinition:
    ad = deepcopy(arc_definition)
    ad.k1_inner = 55
    ad.k1_outer = -55

    mlat = make_arc_to_inner_dipole_start(arc_definition)
    dx, dxp = get_isochronous_dispersions_at_middle_dipole_entrance(
        arc_definition.outer_angle,
        arc_definition.outer_length,
        arc_definition.inner_angle,
        arc_definition.inner_length,
    )
    twiss0 = Twiss(beta_x=20, beta_y=10)
    result = deepcopy(arc_definition)
    quads = []
    for ele in mlat.sequence:
        if isinstance(ele, Quadrupole):
            quads.append(ele)
    result.k1_outer, result.k1_inner = match(
        mlat,
        verbose=False,
        max_iter=1_000_000,
        constr={mlat.sequence[-1]: {"Dx": dx, "Dxp": dxp}},
        vars=quads,
        tw=twiss0,
    )
    return result

def make_arc_to_inner_dipole_start(ad: ArcDefinition) -> MagneticLattice:
    sbend_outer = SBend(l=ad.outer_length, angle=ad.outer_angle)
    d1 = Drift(l=ad.l1)
    d2 = Drift(l=ad.l2)
    d3 = Drift(l=ad.l3)
    qlength = ad.quad_length
    qf1 = Quadrupole(l=qlength, k1=ad.k1_outer)
    qd1 = Quadrupole(l=qlength, k1=ad.k1_inner)

    middle_dipole_start = Marker()
    
    return MagneticLattice(
        [
            sbend_outer,
            d1,
            qf1,
            d2,
            qd1,
            d3,
            middle_dipole_start,
        ]
    )



def make_arc(ad: ArcDefinition) -> MagneticLattice:
    d1 = Drift(l=ad.l1)
    d2 = Drift(l=ad.l2)
    d3 = Drift(l=ad.l3)
    qlength = ad.quad_length
    qf1 = Quadrupole(l=qlength, k1=ad.k1_outer)
    qd1 = Quadrupole(l=qlength, k1=ad.k1_inner)

    sbend_outer = SBend(l=ad.outer_length, angle=ad.outer_angle)
    sbend_inner = SBend(l=ad.inner_length, angle=ad.inner_angle)

    return MagneticLattice(
        [
            sbend_outer,
            d1,
            qf1,
            d2,
            qd1,
            d3,
            sbend_inner,
            d3,
            qd1,
            d2,
            qf1,
            d1,
            sbend_outer,
        ]
    )

    
