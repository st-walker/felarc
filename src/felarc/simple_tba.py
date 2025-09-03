from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from ocelot.cpbd.elements import Drift, Quadrupole, SBend, Marker
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.optics import Twiss

from felarc.matching import match_twiss


@dataclass
class TBADefinition:
    dipole_length: float
    dipole_angle: float
    l1: float
    l2: float
    l3: float
    qlength: float
    k11: float
    k12: float

    # def to_mlat(self) -> MagneticLattice

@dataclass
class TBAComponents:
    d1: Drift
    d2: Drift
    d3: Drift
    q1: Quadrupole
    q2: Quadrupole

    @property
    def quads(self) -> list[Quadrupole]:
        return [self.q1, self.q2]

    @property
    def drifts(self) -> list[Drift]:
        return [self.d1, self.d2, self.d3]

    @property
    def all_elements(self) -> list[Quadrupole | Drift]:
        return self.quads + self.drifts

@dataclass
class TBAMarkers:
    start: Marker = field(default_factory=lambda: Marker("Start"))
    middle: Marker = field(default_factory=lambda: Marker("Middle"))
    end: Marker = field(default_factory=lambda: Marker("End"))
    middle_dipole_start: Marker = field(default_factory=lambda: Marker("Middle Dipole Start"))


@dataclass
class TripleBendAchromat:
    mlat: MagneticLattice
    components: TBAComponents
    markers: TBAMarkers


def get_quad_strengths(ad: TBADefinition) -> TBADefinition:
    # The book suggests varying a drift and a quad strength to match the two conditions
    # But here instead I'm going to use two quads to reach the condition.
    # mlat = make_arc_to_inner_dipole_start(ad)
    tba = make_tba(ad)
    dx, dxp = get_tba_dispersion_condition(ad.dipole_length, ad.dipole_angle)

    twiss0 = Twiss(beta_x=20, beta_y=10)
    result = deepcopy(ad)
    # Get the quads from the lattice.
    result = match_twiss(
        tba.mlat,
        verbose=False,
        max_iter=1_000_000,
        constr={tba.markers.middle_dipole_start: {"Dx": dx, "Dxp": dxp}},
        vars=tba.components.quads,
        twiss0=twiss0,
    )
    return result # type: ignore

def make_tba(ad: TBADefinition) -> TripleBendAchromat:
    sbend = SBend(l=ad.dipole_length, angle=ad.dipole_angle)
    sbend_half = SBend(l=sbend.l/2, angle=sbend.angle/2)
    d1 = Drift(l=ad.l1, eid="d1")
    d2 = Drift(l=ad.l2, eid="d2")
    d3 = Drift(l=ad.l3, eid="d3")
    qlength = ad.qlength
    q1 = Quadrupole(l=qlength, k1=ad.k11, eid="q1")  # type: ignore
    q2 = Quadrupole(l=qlength, k1=ad.k12, eid="q2")  # type: ignore

    markers = TBAMarkers()

    components = TBAComponents(d1, d2, d3, q1, q2)

    mlat = MagneticLattice( # type: ignore
        [
            markers.start,
            sbend,
            d1,
            q1,
            d2,
            q2,
            d3,
            markers.middle_dipole_start,
            sbend_half,
            markers.middle,
            sbend_half,
            d3,
            q2,
            d2,
            q1,
            d1,
            sbend,
            markers.end
        ]
    )

    return TripleBendAchromat(mlat, components, markers)

def get_tba_dispersion_condition(dipole_length: float, dipole_angle: float) -> tuple[float, float]:
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


