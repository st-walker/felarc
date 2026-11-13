

from dataclasses import dataclass

import numpy as np
from ocelot.cpbd.elements import Drift, Marker, Quadrupole, SBend
from ocelot.cpbd.magnetic_lattice import MagneticLattice


@dataclass(init=False)
class DBADefinition:
    l_bend1: float
    angle1: float
    k1_quad1: float
    k1_quad2: float
    l_quad: float
    l_drift1: float
    l_drift2: float
    l_bend2: float
    angle2: float
    l_drift0: float = 1.0

    def __init__(
        self,
        l_bend1: float,
        angle1: float,
        k1_quad1: float,
        k1_quad2: float,
        l_quad: float,
        l_drift1: float,
        l_drift2: float,
        l_bend2: float | None = None,
        angle2: float | None = None,
        l_drift0: float = 1.0,
    ):
        self.l_bend1 = l_bend1
        self.angle1 = angle1
        self.k1_quad1 = k1_quad1
        self.k1_quad2 = k1_quad2
        self.l_quad = l_quad
        self.l_drift1 = l_drift1
        self.l_drift2 = l_drift2
        self.l_bend2: float = self.l_bend1 if l_bend2 is None else l_bend2
        self.angle2: float = self.angle1 if angle2 is None else angle2
        self.l_drift0 = l_drift0

    def __post_init__(self):
        self.l_bend2 = self.l_bend1 if self.l_bend1 is None else self.l_bend1
        self.angle2 = self.angle1 if self.angle1 is None else self.angle1

# change angles
# track beam in ocelot, show lps at different points.
# take this tba

# just do isochronous arc without caring about CSR.

# just check in middle.  with/without csr.
# you might expect to see it bad in middle, worse at end.
# or ideally bad in middle, better at end.

# what if we introduce quadrupole errors?

# change angle.  changed legnth.  got new quads.
# change length of this arc.  to approxiamtely what we have.  then check if we still have.
# i need to set up tracking for ocelot!


@dataclass
class DBAMarkers:
    start: Marker
    stop: Marker
    start_bend1: Marker
    mid_bend1: Marker
    stop_bend1: Marker
    start_bend2: Marker
    mid_bend2: Marker
    stop_bend2: Marker


@dataclass
class DBA:
    mlat: MagneticLattice
    markers: DBAMarkers

    def c2c(self, energy: float = 16.0):
        """get centre to centre matrix"""
        _, R, _ = self.mlat.transfer_maps(energy=energy, # type: ignore
                                          start=self.markers.mid_bend1,
                                          stop=self.markers.mid_bend2)
        return R

    @property
    def quadrupoles(self):
        return list(set([x for x in self.mlat.sequence if isinstance(x, Quadrupole)]))

    # def bending_radius_ratio(self):


# @dataclass
# class DBAElements:
#     quad1: Quadrupole
#     quad2: Quadrupole


def make_dba(ad: DBADefinition) -> DBA:
    bend1_half = SBend(l=ad.l_bend1/2, angle=ad.angle1/2, eid="bend1")
    bend2_half = SBend(l=ad.l_bend2/2, angle=ad.angle2/2, eid="bend2")

    start = Marker("start")
    stop = Marker("stop")

    start_bend1 = Marker("start_bend1")
    mid_bend1 = Marker("mid_bend1")
    stop_bend1 = Marker("stop_bend1")
    start_bend2 = Marker("start_bend2")
    mid_bend2 = Marker("mid_bend2")
    stop_bend2 = Marker("stop_bend2")

    # COMPARE T20 VS TBA BEAMS!!

    
    bend1 = [start_bend1, bend1_half, mid_bend1, bend1_half, stop_bend1]
    bend2 = [start_bend2, bend2_half, mid_bend2, bend2_half, stop_bend2]

    if ad.l_drift0 != 0.0:
        d0 = Drift(l=ad.l_drift0, eid="d0")
        start = [start, d0]
        stop = [d0, stop]

    d1 = Drift(l=ad.l_drift1, eid="d1")
    d2 = Drift(l=ad.l_drift2, eid="d2")

    q1 = Quadrupole(l=ad.l_quad, k1=ad.k1_quad1, eid="q1")  # type: ignore
    q2 = Quadrupole(l=ad.l_quad, k1=ad.k1_quad2, eid="q2")  # type: ignore
    q3 = Quadrupole(l=ad.l_quad, k1=ad.k1_quad2, eid="q3")  # type: ignore

    # q2 = Drift(l=ad.l_quad)

    # q3 = q1
    

    cell = [start, bend1,
            d1, q1, d2, q2, d2, q3, d1,
            bend2,  stop]
    mlat =  MagneticLattice(cell# , method={"global": SecondTM}
                            )  # type: ignore

    return DBA(mlat, DBAMarkers(start=start,
                                stop=stop,
                                start_bend1=start_bend1,
                                mid_bend1=mid_bend1,
                                stop_bend1=stop_bend1,  
                                start_bend2=start_bend2,
                                mid_bend2=mid_bend2,
                                stop_bend2=stop_bend2)# , DBAElements(q1, q2)
               )

def csr_dipole_radius_ratio(angle1, angle2) -> float:
    """calculate the required rigidity of the second dipole w.r.t the first one that gives
    complete CSR cancellation in the linear regime"""
    s1 = np.sin(angle1/2)
    c1 = np.cos(angle1/2)
    s2 = np.sin(angle2/2)
    c2 = np.cos(angle2/2)
    numerator = (s1*(2*s1 - c1*angle1)) ** (3/4)
    denominator = (s2*(2*s2 - c2*angle2)) ** (3/4)
    return np.sign(angle1/ angle2) * numerator / denominator

def csr_optical_ratio(angle1, angle2, radius1) -> float:
    """calculate the required ra of the second dipole w.r.t the first one that gives
    complete CSR cancellation in the linear regime

    """
    s1 = np.sin(angle1/2)
    c1 = np.cos(angle1/2)
    rstar = csr_dipole_radius_ratio(angle1, angle2)
    numerator = angle1 + rstar**(1/3) * angle2
    denominator = radius1 * (-2 + angle1*c1/s1)

    return numerator / denominator

def achromatic_c2c_betatron_matrix(angle1, angle2, alpha1, alpha2, beta1):
    c = np.sin(angle2/2) / np.sin(angle1/2)
    return np.array([[-1/c, 0.0],
                     [-(alpha1 - alpha2)*c/beta1, -c]])
    

def csr_achromat_c2c_matrix(angle1, angle2, radius1, alpha1, alpha2, beta1):
    achromat_c2c_m = achromatic_c2c_betatron_matrix(angle1, angle2, alpha1, alpha2, beta1)
    csr_entry = csr_optical_ratio(angle1, angle2, radius1)
    c = np.sin(angle2/2) / np.sin(angle1/2)
    achromat_c2c_m[1, 0] = - csr_entry * c
    return achromat_c2c_m

def symmetric_dba_beta0(magnet_length, alpha0) -> tuple[float, float]:
    """Employs small angle approximation for bending angles"""
    lb = magnet_length
    return ((lb / 12) * (5 * alpha0 - np.sqrt(alpha0**2 - 24)),
            (lb / 12) * (5 * alpha0 + np.sqrt(alpha0**2 - 24)))
