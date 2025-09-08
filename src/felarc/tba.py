

from dataclasses import dataclass

from ocelot.cpbd.elements import Drift, Marker, Quadrupole, SBend, Sextupole
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.transformations import SecondTM


@dataclass
class TBADefinition:
    l_bend: float
    angle_bend: float
    l_quad: float
    k1_quad1: float
    k1_quad2: float
    k1_quad3: float
    l_sext: float
    k2_sextf: float
    k2_sextd: float
    l_drift1: float
    l_drift2: float
    l_drift3: float
    l_drift4: float
    l_drift5: float
    l_drift6: float
    l_drift7: float    


def make_tba(ad: TBADefinition) -> MagneticLattice:
    bend = SBend(l=ad.l_bend, angle=ad.angle_bend, eid="bend")
    half_bend = SBend(l=ad.l_bend/2, angle=ad.angle_bend/2, eid="half-bend")

    start_bend1 = Marker("start_bend1")
    stop_bend1 = Marker("stop_bend1")
    start_bend2 = Marker("start_bend2")
    stop_bend2 = Marker("stop_bend2")
    start_bend3 = Marker("start_bend3")
    stop_bend3 = Marker("stop_bend3")

    bend1 = [start_bend1, bend, stop_bend1]
    bend2 = [start_bend2, half_bend, half_bend, stop_bend2]
    bend3 = [start_bend3, bend, stop_bend3]
    
    d1 = Drift(l=ad.l_drift1, eid="d1")
    d2 = Drift(l=ad.l_drift2, eid="d2")
    d3 = Drift(l=ad.l_drift3, eid="d3")
    d4 = Drift(l=ad.l_drift4, eid="d4")
    d5 = Drift(l=ad.l_drift5, eid="d5")
    d6 = Drift(l=ad.l_drift6, eid="d6")
    d7 = Drift(l=ad.l_drift7, eid="d7")

    sf = Sextupole(l=ad.l_sext, eid="sf", k2=ad.k2_sextf)
    sd = Sextupole(l=ad.l_sext, eid="sd", k2=ad.k2_sextd)
    
    q1 = Quadrupole(l=ad.l_quad, k1=ad.k1_quad1, eid="q1")  # type: ignore
    q2 = Quadrupole(l=ad.l_quad, k1=ad.k1_quad2, eid="q2")  # type: ignore
    q3 = Quadrupole(l=ad.l_quad, k1=ad.k1_quad3, eid="q3")  # type: ignore

    cell = [d1, bend1,
            d2, sd, d3, q3, d4, q1, d5, q2, d6, sf, d7,
            bend2,
            d7, sf, d6, q2, d5, q1, d4, q3, d3, sd, d2,
            bend3, d1]
    return MagneticLattice(cell, method={"global": SecondTM}) # type: ignore

@dataclass
class TBAMarkers:
    start_bend1: Marker
    stop_bend1: Marker
    start_bend2: Marker
    stop_bend2: Marker
    start_bend3: Marker
    stop_bend3: Marker
    

@dataclass
class TBA:
    mlat: MagneticLattice
    markers: TBAMarkers
