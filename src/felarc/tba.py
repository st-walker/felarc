from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache

import numpy as np
import numpy.typing as npt
from ocelot.cpbd.elements import Drift, Marker, Quadrupole, SBend, Sextupole
from ocelot.cpbd.elements.optic_element import OpticElement
from ocelot.cpbd.magnetic_lattice import MagneticLattice, flatten
from ocelot.cpbd.optics import Twiss, twiss
from ocelot.cpbd.transformations import SecondTM
from scipy.optimize import fmin, differential_evolution

from felarc.matching import match_matrix_element

# with lattice_builder(s) as y:
#     # y.add_at(


@dataclass
class SecondOrderTBAParameters:
    l_bend: float
    angle_bend: float | tuple[float, float, float]
    l_quad: float
    k1_quad1: float
    k1_quad2: float
    k1_quad3: float
    l_sext: float
    k2_sextf: float
    k2_sextd: float
    l_drift0: float
    l_drift1: float
    l_drift2: float
    l_drift3: float
    l_drift4: float
    l_drift5: float
    l_drift6: float

    def make_tba(self) -> TBA:
        ad = self
        try:
            angle_bend1, angle_bend2, angle_bend3 = ad.angle_bend
        except TypeError:
            angle_bend1 = angle_bend2 = angle_bend3 = ad.angle_bend

        bend1 = SBend(l=ad.l_bend, angle=angle_bend1, eid="bend1")
        half_bend2 = SBend(l=ad.l_bend/2, angle=angle_bend2/2, eid="half-bend")
        half_bend2 = SBend(l=ad.l_bend/2, angle=angle_bend2/2, eid="half-bend")
        bend3 = SBend(l=ad.l_bend, angle=angle_bend3, eid="bend3")

        start = Marker("start")
        stop = Marker("stop")

        start_bend1 = Marker("start_bend1")
        # mid_bend1 = Marker("mid_bend1")
        stop_bend1 = Marker("stop_bend1")
        start_bend2 = Marker("start_bend2")
        mid_bend2 = Marker("mid_bend2")
        stop_bend2 = Marker("stop_bend2")
        start_bend3 = Marker("start_bend3")
        # mid_bend3 = Marker("mid_bend3")
        stop_bend3 = Marker("stop_bend3")

        # COMPARE T20 VS TBA BEAMS!!

        bend1 = [start_bend1, bend1, stop_bend1]
        bend2 = [start_bend2, half_bend2, mid_bend2, half_bend2, stop_bend2]
        bend3 = [start_bend3, bend3, stop_bend3]

        d0 = Drift(l=ad.l_drift0, eid="d0")
        d1 = Drift(l=ad.l_drift1, eid="d1")
        d2 = Drift(l=ad.l_drift2, eid="d2")
        d3 = Drift(l=ad.l_drift3, eid="d3")
        d4 = Drift(l=ad.l_drift4, eid="d4")
        d5 = Drift(l=ad.l_drift5, eid="d5")
        d6 = Drift(l=ad.l_drift6, eid="d6")

        sd = Sextupole(l=ad.l_sext, eid="sd", k2=ad.k2_sextd)
        sf = Sextupole(l=ad.l_sext, eid="sf", k2=ad.k2_sextf)

        q1 = Quadrupole(l=ad.l_quad, k1=ad.k1_quad1, eid="q1")  # type: ignore
        q2 = Quadrupole(l=ad.l_quad, k1=ad.k1_quad2, eid="q2")  # type: ignore
        q3 = Quadrupole(l=ad.l_quad, k1=ad.k1_quad3, eid="q3")  # type: ignore
        between_dipoles = [d1, sd, d2, q1, d3, q2, d4, q3, d5, sf, d6]

        cell = [start, d0, bend1,
                between_dipoles,
                bend2,
                between_dipoles[::-1],
                bend3, d0, stop]
        mlat =  MagneticLattice(cell# , method={"global": SecondTM}
                                )  # type: ignore

        # mseq = MagneticSequence(flatten(cell)) # type: ignore

        return TBA(
            ad,
            mlat,
            TBAMarkers(
                start=start,
                stop=stop,
                start_bend1=start_bend1,
                stop_bend1=stop_bend1,
                start_bend2=start_bend2,
                mid_bend2=mid_bend2,
                stop_bend2=stop_bend2,
                start_bend3=start_bend3,
                stop_bend3=stop_bend3,
            ),
            [sd, sf],
            [q1, q2, q3],
            [d0, d1, d2, d3, d4, d5, d6],
            [TBADipole(bend1), TBADipole(bend2), TBADipole(bend3)]
        )

    def to_first_order(self):
        return FirstOrderTBAParameters(l_bend=self.l_bend,
                                angle_bend=self.angle_bend,
                                l_quad=self.l_quad,
                                k1_quad1=self.k1_quad1,
                                k1_quad2=self.k1_quad2,
                                k1_quad3=self.k1_quad3,
                                l_drift0=self.l_drift0,
                                l_drift1=self.l_drift1+self.l_sext+self.l_drift2,
                                l_drift2=self.l_drift3,
                                l_drift3=self.l_drift4,
                                l_drift4=self.l_sext+self.l_drift5+self.l_drift6)


    def reset_quadrupole_positions(self, drifts) -> None:
        drifts[0].l = self.l_drift0
        drifts[1].l = self.l_drift1
        drifts[2].l = self.l_drift2
        drifts[3].l = self.l_drift3
        drifts[4].l = self.l_drift4
        drifts[5].l = self.l_drift5
        drifts[6].l = self.l_drift6

@dataclass
class FirstOrderTBAParameters:
    l_bend: float
    angle_bend: float
    l_quad: float
    k1_quad1: float
    k1_quad2: float
    k1_quad3: float
    l_drift0: float
    l_drift1: float
    l_drift2: float
    l_drift3: float
    l_drift4: float

    def make_tba(self) -> TBA:
        ad = self
        try:
            angle_bend1, angle_bend2, angle_bend3 = ad.angle_bend
        except TypeError:
            angle_bend1 = angle_bend2 = angle_bend3 = ad.angle_bend

        bend1 = SBend(l=ad.l_bend, angle=angle_bend1, eid="bend1")
        half_bend2 = SBend(l=ad.l_bend/2, angle=angle_bend2/2, eid="half-bend")
        half_bend2 = SBend(l=ad.l_bend/2, angle=angle_bend2/2, eid="half-bend")
        bend3 = SBend(l=ad.l_bend, angle=angle_bend3, eid="bend3")

        start = Marker("start")
        stop = Marker("stop")

        start_bend1 = Marker("start_bend1")
        # mid_bend1 = Marker("mid_bend1")
        stop_bend1 = Marker("stop_bend1")
        start_bend2 = Marker("start_bend2")
        mid_bend2 = Marker("mid_bend2")
        stop_bend2 = Marker("stop_bend2")
        start_bend3 = Marker("start_bend3")
        # mid_bend3 = Marker("mid_bend3")
        stop_bend3 = Marker("stop_bend3")

        # COMPARE T20 VS TBA BEAMS!!

        bend1 = [start_bend1, bend1, stop_bend1]
        bend2 = [start_bend2, half_bend2, mid_bend2, half_bend2, stop_bend2]
        bend3 = [start_bend3, bend3, stop_bend3]

        d0 = Drift(l=ad.l_drift0, eid="d0")
        d1 = Drift(l=ad.l_drift1, eid="d1")
        d2 = Drift(l=ad.l_drift2, eid="d2")
        d3 = Drift(l=ad.l_drift3, eid="d3")
        d4 = Drift(l=ad.l_drift4, eid="d4")

        q1 = Quadrupole(l=ad.l_quad, k1=ad.k1_quad1, eid="q1")  # type: ignore
        q2 = Quadrupole(l=ad.l_quad, k1=ad.k1_quad2, eid="q2")  # type: ignore
        q3 = Quadrupole(l=ad.l_quad, k1=ad.k1_quad3, eid="q3")  # type: ignore
        between_dipoles = [d1, q1, d2, q2, d3, q3, d4]

        cell = [start, d0, bend1,
                between_dipoles,
                bend2,
                between_dipoles[::-1],
                bend3, d0, stop]

        mlat = MagneticLattice(cell) # type: ignore

        # mlat = MagneticSequence(flatten(cell))

        return TBA(
            self,
            mlat,
            TBAMarkers(
                start=start,
                stop=stop,
                start_bend1=start_bend1,
                stop_bend1=stop_bend1,
                start_bend2=start_bend2,
                mid_bend2=mid_bend2,
                stop_bend2=stop_bend2,
                start_bend3=start_bend3,
                stop_bend3=stop_bend3,
            ),
            [],
            [q1, q2, q3],
            [d0, d1, d2, d3, d4],
            [TBADipole(bend1), TBADipole(bend2), TBADipole(bend3)]
        )

    def reset_quadrupole_positions(self, drifts) -> None:
        # for drift_length, drift in zip(self.parameters.drift
        drifts[0].l = self.l_drift0
        drifts[1].l = self.l_drift1
        drifts[2].l = self.l_drift2
        drifts[3].l = self.l_drift3
        drifts[4].l = self.l_drift4


class TBADipole:
    def __init__(self, elements):
        self.elements = elements

    @property
    def radius(self) -> float:
        return self.l / self.angle

    @property
    def angle(self) -> float:
        result = 0.0
        for e in self.elements:
            try:
                result += e.angle
            except AttributeError:
                pass
        return result

    @property
    def l(self) -> float:
        return sum((e.l for e in self.elements))

    def curvature(self) -> float:
        return self.angle / self.l

    def mlat(self):
        return MagneticLattice(self.elements) # type: ignore


@dataclass(frozen=True)
class TBAMarkers:
    start: Marker
    stop: Marker
    start_bend1: Marker
    stop_bend1: Marker
    start_bend2: Marker
    mid_bend2: Marker
    stop_bend2: Marker
    start_bend3: Marker
    stop_bend3: Marker


class MagneticSequence:
    MINIMUM_DRIFT_SIZE = 1e-6
    def __init__(self, iterable = None# , length=None
                 ):
        self._sequence: list[tuple[float, OpticElement]] = []

        element_start_s = 0.0
        for element in iterable or []:
            if not isinstance(element, Drift):
                self._sequence.append((element_start_s, element))
            element_start_s += element.l

    def __repr__(self) -> str:
        return f"<MagneticSequence: {repr(self._sequence)}>"

    def to_mlat(self) -> MagneticLattice:
        sequence_with_drifts = []
        s_before = 0.0
        for s, element in self._sequence:
            delta_s = s - s_before
            if delta_s > self.MINIMUM_DRIFT_SIZE:
                sequence_with_drifts.append(Drift(l=delta_s))
            sequence_with_drifts.append(element)
            s_before = s + element.l

        return MagneticLattice(sequence_with_drifts) # type: ignore

    def remove(self, value: OpticElement) -> None:
        index = next(i for (i, (_, element)) in enumerate(self._sequence) if element is value)
        del self._sequence[index]

    def remove_all(self, value: OpticElement) -> None:
        indices = [i for (i, (_, element)) in enumerate(self._sequence) if element is value]
        for index in reversed(indices):
            del self._sequence[index]

    def __delitem__(self, index: int) -> None:
        del self._sequence[index]

    # def stretch(self, stretch_factor: float) -> None:
    #     pass

    def __getitem__(self, index: int) -> tuple[float, OpticElement]:
        return self._sequence[index]

    def point_is_element(self, s: float) -> bool:
        for s_start, element in self._sequence:
            if s_start <= s < (s_start + element.l):
                return True
        return False

    def point_is_drift(self, s: float) -> bool:
        return not self.point_is_element(s)

    def shift(self, index: int, delta_z: float) -> None:
        s_start, element = self[index]
        new_s = s_start + delta_z
        previous_element = self[index-1]
        next_element = self[index+1]

        # if self.point_is_element(
        pass

    def scale(self, factor: float) -> None:
        self._sequence = [(pos*factor, element) for (pos, element) in self._sequence]
        # for _, element in self._sequence:
        #     element.l *= factor


@dataclass
class TBA:
    parameters: FirstOrderTBAParameters | SecondOrderTBAParameters
    lat: MagneticSequence | MagneticLattice
    markers: TBAMarkers
    sexts: list[Sextupole]
    quads: list[Quadrupole]
    drifts: list[Drift]
    dipoles: list[TBADipole]

    @property
    def mlat(self) -> MagneticLattice:
        try:
            return self.lat.to_mlat()
        except AttributeError:
            return self.lat

    def twiss0(self) -> Twiss:
        twisses = twiss(self.mlat)
        return twisses[0] # type: ignore

    def twiss1(self, twiss0=None) -> Twiss:
        twisses = twiss(self.mlat, tws0=twiss0)
        return twisses[1] # type: ignore

    def mx2x2_between_dipoles(self, energy: float = 16.0):
        _, R, *_ = self.mlat.transfer_maps(energy=energy,
                                           start=self.markers.stop_bend1,
                                           stop=self.markers.start_bend2)
        return R[:2, :2] # type: ignore

    def my2x2_between_dipoles(self, energy: float = 16.0):
        _, R, *_ = self.mlat.transfer_maps(energy=energy,
                                           start=self.markers.stop_bend1,
                                           stop=self.markers.start_bend2)
        return R[2:4, 2:4] # type: ignore

    def traces(self, energy: float = 16.0):
        _, R, *_ = self.mlat.transfer_maps(energy=energy) # type: ignore
        return np.trace(R[0:2, 0:2]).item(), np.trace(R[2:4, 2:4]).item() # type: ignore

    def dipole_length(self) -> float:
        dipole = self.mlat.get_sequence_part(start=self.markers.start_bend1,
                                             stop=self.markers.stop_bend1)
        return sum((x.l for x in dipole))

    def my2x2(self, energy: float = 16.0):
        _, R, *_ = self.mlat.transfer_maps(energy=energy,
                                           start=self.markers.stop_bend1,
                                           stop=self.markers.start_bend2)
        return R[2:4, 2:4]

    def reset_mseq(self):
        tba = make_tba(self.parameters)
        self.mseq = tba.mseq

    def shift_quadrupole(self, quadrupole: Quadrupole, delta_s: float, minimum_drift_length=0.1):
        mlat_index = self.mlat.sequence.index(quadrupole)
        drift_before = self.mlat.sequence[mlat_index-1]
        drift_after = self.mlat.sequence[mlat_index+1]


        # If drift before goes under minimum_drift_length, then we have to increase delta_s back so
        # That it keeps the drift at at least 0.1.
        if (drift_before.l - delta_s) < minimum_drift_length:
            delta_s = drift_before.l - minimum_drift_length
        if (drift_after.l + delta_s) < minimum_drift_length:
            delta_s = minimum_drift_length - drift_after.l


        drift_before.l -= delta_s
        drift_after.l += delta_s

        if drift_before.l < 0:
            raise ValueError("Negative drift length")
        if drift_after.l < 0:
            raise ValueError("Negative drift length")

    def component_matrices(self, energy = 16.0):
        mlat = self.mlat
        _, Rd1, *_ = mlat.transfer_maps(energy=energy, # type: ignore
                                            stop=self.markers.start_bend1)
        _, Rb1, *_ = mlat.transfer_maps(energy=energy, # type: ignore
                                            start=self.markers.start_bend1,
                                            stop=self.markers.stop_bend1)
        _, Rm, *_ = mlat.transfer_maps(energy=energy, # type: ignore
                                            start=self.markers.stop_bend1,
                                            stop=self.markers.start_bend2)
        _, Rb2, *_ = mlat.transfer_maps(energy=energy, # type: ignore
                                            start=self.markers.start_bend2,
                                            stop=self.markers.stop_bend2)
        _, Rm2, *_ = mlat.transfer_maps(energy=energy, # type: ignore
                                                   start=self.markers.stop_bend2,
                                                   stop=self.markers.start_bend3)
        _, Rb3, *_ = mlat.transfer_maps(energy=energy, # type: ignore
                                            start=self.markers.start_bend3,
                                            stop=self.markers.stop_bend3)
        _, Rd2, *_ = mlat.transfer_maps(energy=energy, # type: ignore
                                            start=self.markers.stop_bend3)

        return Rd1, Rb1, Rm, Rb2, Rm2, Rb3, Rd2

    def rtot(self, energy: float = 16.0) -> np.ndarray:
        _, R, *_ = self.mlat.transfer_maps(energy=energy)
        return R
        # Rd1, Rb1, Rm, Rb2, Rm2, Rb3, Rd2 = self.component_matrices(energy=energy)
        # return Rd2 @ Rb3 @ Rm2 @ Rb2 @ Rm @ Rb1 @ Rd1

    # def full_ma(self, m_between_dipoles):
    #     # We track to end of first dipole,
    #     # Then we do m
    #     pass

    # def dipole_matrix(self):
    #     from IPython import embed; embed()
    #     pass

    # def trace_contour(self):
    #     from IPython import embed; embed()
    #     pass

    def update_quadrupole_strengths(self) -> None:
        self.parameters.l_drift1 = self.drifts[0].l
        self.parameters.l_drift2 = self.drifts[1].l
        self.parameters.l_drift3 = self.drifts[2].l

    def update_quadrupole_positions(self) -> None:
        self.parameters.l_drift1 = self.drifts[0].l
        self.parameters.l_drift2 = self.drifts[1].l
        self.parameters.l_drift3 = self.drifts[2].l
        self.parameters.l_drift4 = self.drifts[3].l
        self.parameters.l_drift5 = self.drifts[4].l
        self.parameters.l_drift6 = self.drifts[5].l
        self.parameters.l_drift7 = self.drifts[6].l


    def reset_quadrupole_positions(self) -> None:
        self.parameters.reset_quadrupole_positions(self.drifts)
        # # for drift_length, drift in zip(self.parameters.drift
        # self.drifts[0].l = self.parameters.l_drift1
        # self.drifts[1].l = self.parameters.l_drift2
        # self.drifts[2].l = self.parameters.l_drift3
        # self.drifts[3].l = self.parameters.l_drift4
        # self.drifts[4].l = self.parameters.l_drift5
        # self.drifts[5].l = self.parameters.l_drift6
        # self.drifts[6].l = self.parameters.l_drift7

    # def named_element(self, name):
    #     for ele in self.mlat.sequence:
    #         if ele.id == name:
    #             return ele

    def update_quads(self, opt: OptimisationOutput, index: int = -1) -> None:
        for quad, k1 in opt.new_k1s(index=index):
            quad.k1 = k1

        self.reset_quadrupole_positions()
        for quad, delta_z in opt.new_quad_shifts(index=index):
            self.shift_quadrupole(quad, delta_z)

    def angles(self) -> list[float]:
        return [d.angle for d in self.dipoles]

    def is_symmetric(self) -> bool:
        bends = self.dipoles
        return all([bends[0].l == bends[2].l,
                    bends[0].angle == bends[2].angle])


def achromatic_m12_m22(tba, m11=None, m21=None) -> tuple[float, float]:
    if not tba.is_symmetric():
        raise RuntimeError()

    # breakpoint()

    if m11 is None or m21 is None:
        m = tba.mx2x2_between_dipoles()
        if m11 is None:
            m11 = m[0, 0]
        if m21 is None:
            m21 = m[1, 0]

    dipoles = tba.dipoles

    angle_inner = dipoles[1].angle
    angle_outer = dipoles[0].angle
    lb_outer = dipoles[0].l

    m12 = (-m11 * angle_inner - 2 * angle_outer - lb_outer * m11 * m21 * angle_outer) / (2 * m21 * angle_outer)
    m22 = (-angle_inner - lb_outer * m21 * angle_outer) / (2 * angle_outer)

    return m12, m22

#     pass

# def achromatic_m12_m22(m11, m21, dipole_length) -> tuple[npt.ArrayLike, npt.ArrayLike]:
#     lb = dipole_length
#     m12 = (-2 - m11 - lb * m11 * m21) / (2*m21)
#     m22 = 0.5 * (-1 - lb * m21)
#     return m12, m22


def isochronous_m21(tba, m11=None) -> float:
    if not tba.is_symmetric():
        raise RuntimeError()

    if m11 is None:
        m = tba.mx2x2_between_dipoles()
        m11 = m[0, 0]

    dipoles = tba.dipoles

    angle_inner = dipoles[1].angle
    angle_outer = dipoles[0].angle
    lb_outer = dipoles[0].l
    lb_inner = dipoles[1].l

    return -6 * ((m11 * angle_inner**2 + 2 * angle_inner * angle_outer)
                 / (lb_inner * angle_inner**2 - 4 * lb_outer * angle_outer**2))


# def csr_cancellation_m21(

# def isochronous_m21(m11, dipole_length):
#     return 2 * (2 + m11) / dipole_length


def csr_cancellation_m21s_old(m11, dipole_length):
    def f(sign):
        return (5 - 2 * m11 + sign * np.sqrt(121 + 28 * m11 + 4*m11**2)) / (2 * dipole_length)
    return f(+1), f(-1)


from typing import Tuple


def r_small(Li: float, Lo: float, theta_i: float, theta_o: float) -> float:
    return (Li**(4/3) * theta_i**(5/3)) / (Lo**(4/3) * theta_o**(5/3))


def s1_small(Lo: float) -> float:
    return -6.0 / Lo


def s2_small(Li: float, Lo: float, theta_i: float, theta_o: float) -> float:
    return (
        -6.0 * theta_i
        * (Li**(1/3) * theta_i**(2/3) + 2.0 * Lo**(1/3) * theta_o**(2/3))
        / (Lo**(4/3) * theta_o**(5/3))
    )


def s3_small(Li: float, Lo: float, theta_i: float, theta_o: float) -> float:
    return (
        -6.0 * theta_o
        * (3.0 * Lo**(1/3) * theta_o**(2/3) + 2.0 * Li**(1/3) * theta_i**(2/3))
        / (Lo**(4/3) * theta_o**(5/3))
    )


def csr_compatibility_residual(
    s1: float, s2: float, s3: float, theta_i: float, theta_o: float
) -> float:
    return (s1 + s3) * theta_i - 2.0 * s2 * theta_o


def discriminant(
    m11: float,
    Li: float,
    r: float,
    s1: float,
    s2: float,
    s3: float,
) -> float:
    denom = s1 + s3
    return (
        16.0 * Li * (s1 - s3) * (s1 + m11 * s2 + s3) / denom
        + (2.0 * r + 4.0 * m11 + Li * s2 * (s3 - s1) / denom) ** 2
    )


def csr_cancellation_m21s(tba, m11: float | None = None) -> tuple[float, float]:
    if not tba.is_symmetric():
        raise RuntimeError()

    if m11 is None:
        m = tba.mx2x2_between_dipoles()
        m11 = m[0, 0]

    dipoles = tba.dipoles

    angle_inner = dipoles[1].angle
    angle_outer = dipoles[0].angle
    lb_outer = dipoles[0].l
    lb_inner = dipoles[1].l

    r = r_small(lb_inner, lb_outer, angle_inner, angle_outer)
    s1 = s1_small(lb_outer)
    s2 = s2_small(lb_inner, lb_outer, angle_inner, angle_outer)
    s3 = s3_small(lb_inner, lb_outer, angle_inner, angle_outer)

    residual = csr_compatibility_residual(s1, s2, s3, angle_inner, angle_outer)
    if abs(residual) > 1e-6:
        raise ValueError(f"CSR compatibility residual is too large: {residual}")

    return _csr_cancellation_m21s(m11, lb_inner, r, s1, s2, s3)

def _csr_cancellation_m21s(
    m11: float,
    Li: float,
    r: float,
    s1: float,
    s2: float,
    s3: float,
) -> Tuple[float, float]:
    denom = s1 + s3
    D = discriminant(m11, Li, r, s1, s2, s3)

    root = np.sqrt(D)

    base = -2.0 * r - 4.0 * m11 + Li * s2 - 2.0 * Li * s2 * s3 / denom

    return (
        (base + root) / (4.0 * Li),
        (base - root) / (4.0 * Li),
    )


def match_achromatic_condition(
    tba: TBA,
    k1_quads: list[Quadrupole] | None = None,
    position_quads: list[Quadrupole] | None = None,
    match_isochronous: bool = False,
    cancel_csr: bool = False,
    stable_solution: bool = True,
    pathological_solution: bool = False,
    weights: dict[str, float] | None = None,
    match_m11_m21: tuple[float, float] | None = None,
    beta_max: float | None = None,
) -> OptimisationOutput:
    # Order or arguments for the cost function
    # Quadrupole k1s then quadrupole positions.

    # if path

    if not k1_quads and not position_quads:
        raise ValueError()

    k1_quads = k1_quads or []
    position_quads = position_quads or []

    initial_guess = [q.k1 for q in k1_quads] + [0.0] * len(position_quads)

    mweights = np.array([[1.0, 1.0], [1.0, 1.0]])

    wxtrace = wytrace = 10.0
    if weights is not None:
        mweights[0, 0] = weights.get("m11", mweights[0, 0])
        mweights[0, 1] = weights.get("m12", mweights[0, 1])
        mweights[1, 0] = weights.get("m21", mweights[1, 0])
        mweights[1, 1] = weights.get("m22", mweights[1, 1])

        wxtrace = weights.get("xtrace", wxtrace)
        wytrace = weights.get("ytrace", wytrace)

    def error_function(quad_k1s_and_delta_zs):
        tba.reset_quadrupole_positions()
        # Update the lattice from the arguments
        # first extract k1s from the 1d list of inputs
        for quadrupole, k1 in zip(k1_quads, quad_k1s_and_delta_zs):
            quadrupole.k1 = k1
        # Now get second half of the list (the delta_s):
        for quadrupole, delta_z in zip(
            position_quads, quad_k1s_and_delta_zs[len(k1_quads) :]
        ):
            tba.shift_quadrupole(quadrupole, delta_z)

        # Get 2x2 between the dipoles having updated the quadrupole strengths.
        actual_m = tba.mx2x2_between_dipoles()
        # ((m11, m12), (m21, m22)) = m
        target_m = actual_m.copy()
        # np.

        # from IPython import embed; embed()
        # First we get any constraints on m11 and m21.  If we match
        # for only isochronicity or only for csr-cancellation, we
        # don't care about m11.  We only get combined m11 and m12
        # constraints if we want to match both isochronicity and
        # csr-cancellation.  We always match for m12 and m22 in order
        # to be achromatic.

        # We always want to match the achromatic conditions m22 and
        # m12.

        # However, we may or may not have to match also for m21 and
        # m11.  If only matching for isochronicity OR CSR-mitigation,
        # we only match for m21 and m11 is a free parameter.  If in
        # addition matching for achromatism, we also match for both
        # isochronicity AND CSR-mitigation, then we have to match for
        # ALL FOUR matrix elements m11, m12, m21, m22 and we have no free parmaters

        if match_m11_m21 is not None:
            target_m[0, 0] = match_m11_m21[0]
            target_m[1, 0] = match_m11_m21[1]
        elif match_isochronous and cancel_csr:
            crossing1, crossing2 = csr_isochronous_crossings_from_tba(tba)
            # Pick the one with m21 = 0 because it appears to be less
            # pathological than the other one.
            target_m21_zero_epsilon = 0.05

            if crossing1.m21 == 0:
                pathological_crossing = crossing1
                well_behaved_crossing = crossing2
            elif crossing2.m21 == 0:
                pathological_crossing = crossing2
                well_behaved_crossing = crossing1
            else:
                raise ValueError("Neither crossing has m21 = 0, something is wrong.")

            if pathological_solution:
                target_m11, target_m21 = (
                    pathological_crossing.m11,
                    target_m21_zero_epsilon,
                )
            else:
                target_m11, target_m21 = (
                    well_behaved_crossing.m11,
                    well_behaved_crossing.m21,
                )

            target_m[0, 0] = target_m11
            target_m[1, 0] = target_m21
        elif match_isochronous:
            target_m21 = isochronous_m21(tba)
            mweights[0, 0] = 0.0
            target_m[1, 0] = target_m21
        elif cancel_csr:
            target_m21, _ = csr_cancellation_m21s(tba, m11=actual_m[0, 0])
            mweights[0, 0] = 0.0
            target_m[1, 0] = target_m21            
        else:
            # Then they are free parameters and we add no cost for them.
            mweights[0, 0] = mweights[1, 0] = 0.0


        # Given our choices for m11 and m21, we now calculate the
        # corresponding achromatic m11 and m12 for that configuration.
        target_m12, target_m22 = achromatic_m12_m22(tba, m11=target_m[0, 0], m21=target_m[1, 0])
        target_m[0, 1] = target_m12
        # Don't match target_m22, we just assume it takes care of itself...
        target_m[1, 1] = target_m22

        # from IPython import embed; embed()
        if np.isnan(target_m).any():
            raise ValueError(f"NaN in target_m: {target_m}")

        cost = 0

        m_cost = ((actual_m - target_m)) ** 2 * mweights

        # breakpoint()
        print("---")
        for i in range(2):
            for j in range(2):
                if mweights[i, j] > 0:
                    print(
                        f"m{i + 1}{j + 1} = {actual_m[i, j]:.3g}, target_m{i + 1}{j + 1} = {target_m[i, j]:.3g}, cost = {m_cost[i, j]}"
                    )
        cost += m_cost.sum()

        xtrace, ytrace = tba.traces()
        if stable_solution or pathological_solution:
            xabstrace = abs(xtrace)
            yabstrace = abs(ytrace)
            xtrace_cost = max(0, xabstrace - 1.99) ** 2 * wxtrace
            ytrace_cost = max(0, yabstrace - 1.99) ** 2 * wytrace
            print(f"|Tr(x)| = {xabstrace:.3g}, xtrace_cost = {xtrace_cost}")
            print(f"|Tr(y)| = {yabstrace:.3g}, ytrace_cost = {ytrace_cost}")
            cost += xtrace_cost + ytrace_cost
            # cost += max(0, abs(xtrace) - 1.999) ** 2 * wxtrace
            # cost += max(0, abs(ytrace) - 1.999) ** 2 * wytrace

            max_beta = 200
            if xabstrace < 2 and yabstrace < 2:
                twisses = twiss(tba.mlat, return_df=True)
                max_beta_x = twisses["beta_x"].max()
                max_beta_y = twisses["beta_y"].max()

                cost_beta_x = max(0, max_beta_x - max_beta) ** 2
                cost_beta_y = max(0, max_beta_y - max_beta) ** 2

                cost+= cost_beta_x + cost_beta_y

                print(f"Max beta_x = {max_beta_x:.3g}, {cost_beta_x}")
                print(f"Max beta_y = {max_beta_y:.3g}, {cost_beta_y}")

                print("Stable solution found within tolerance.")

        # twiss(
            
        # for quadrupole, k1 in zip(k1_quads, quad_k1s_and_delta_zs):
        #     print(k1)
        #     # quadrupole.k1 = k1
            


        # print(
        #     quad_k1s_and_delta_zs,
        #     f"XTRACE = {xtrace}",
        #     f"YTRACE = {ytrace}",
        #     f"M12 COST = {(target_m12 - actual_m[0, 1]) ** 2}",
        #     f"M22 COST = {(target_m22 - actual_m[1, 1]) ** 2}",
        #     f"COST = {cost}",
        # )
        # print(f"m11={actual_m[0, 0]}, m21={actual_m[1, 0]}")

        # print("R56 = ", tba.rtot()[4, 5])

        print(quad_k1s_and_delta_zs)

        # print(quad_k1s_and_delta_zs, xtrace, ytrace, (target_m12 - m12)**2,  (target_m22 - m22)**2, f"COST = {cost}")

        return cost

    res = fmin(
        error_function,
        initial_guess,
        xtol=1e-6,
        ftol=1e-6,
        maxiter=10000,
        maxfun=10000,
        full_output=True,
        retall=True,
    )

    # differential_evolution(
    #     error_function,
    #     bounds=[(-10, 10)] * len(initial_guess),
    #     tol=1e-6,
    #     maxiter=10000,
    #     # maxfun=10000,
    # )

    return OptimisationOutput(res[0], res[1], res[2], res[5], k1_quads, position_quads)


@dataclass(frozen=True)
class CSRIsochronousCrossing:
    """
    Intersection point between an isochronous solution
    and a CSR-cancellation solution.
    """

    m11: float
    m21: float


def _csr_isochronous_crossings_from_ratios(
    outer_length: float,
    angle_ratio: float,
    length_ratio: float,
) -> tuple[CSRIsochronousCrossing, CSRIsochronousCrossing]:
    """
    Internal implementation using dimensionless ratios.

    Parameters
    ----------
    outer_length:
        Length of the outer dipoles.

    angle_ratio:
        inner_angle / outer_angle

    length_ratio:
        inner_length / outer_length
    """

    if outer_length == 0:
        raise ValueError("outer_length must be nonzero.")

    if angle_ratio == 0:
        raise ValueError("angle_ratio must be nonzero.")

    if length_ratio <= 0:
        raise ValueError("length_ratio must be positive.")

    # ------------------------------------------------------------------
    # Branch 1:
    #
    # Singular branch.
    # Corresponds to the generalized version of (-2, 0).
    # ------------------------------------------------------------------

    m11_branch_1 = -2.0 / angle_ratio
    m21_branch_1 = 0.0

    # ------------------------------------------------------------------
    # Branch 2:
    #
    # Regular branch.
    # Reduces to:
    #
    #   m11 = 7/4
    #   m21 = 15 / (2 L)
    #
    # for identical dipoles.
    # ------------------------------------------------------------------

    u = length_ratio * angle_ratio**2

    m11_branch_2 = (
        16.0 * (u**(1.0 / 3.0) + 1.0)
        - u * (2.0 * u + u**(4.0 / 3.0) + 8.0)
    ) / (4.0 * angle_ratio * (u + 2.0))

    m21_branch_2 = (
        (3.0 * angle_ratio) / (2.0 * outer_length)
        * (
            length_ratio**(4.0 / 3.0)
            * angle_ratio**(8.0 / 3.0)
            + 2.0 * length_ratio * angle_ratio**2
            + 4.0
            * length_ratio**(1.0 / 3.0)
            * angle_ratio**(2.0 / 3.0)
            + 8.0
        )
        / (length_ratio * angle_ratio**2 + 2.0)
    )

    return (
        CSRIsochronousCrossing(
            m11=m11_branch_1,
            m21=m21_branch_1,
        ),
        CSRIsochronousCrossing(
            m11=m11_branch_2,
            m21=m21_branch_2,
        ),
    )

def csr_isochronous_crossings_from_tba(tba: TBA) -> tuple[CSRIsochronousCrossing, CSRIsochronousCrossing]:
    if not tba.is_symmetric():
        raise RuntimeError("TBA must be symmetric.")

    dipoles = tba.dipoles

    outer_length = dipoles[0].l
    inner_length = dipoles[1].l
    outer_angle = dipoles[0].angle
    inner_angle = dipoles[1].angle

    return _csr_isochronous_crossings_from_ratios(
        outer_length=outer_length,
        angle_ratio=inner_angle / outer_angle,
        length_ratio=inner_length / outer_length,
    )

def csr_isochronous_crossings(
    outer_length: float,
    inner_length: float,
    outer_angle: float,
    inner_angle: float,
) -> tuple[CSRIsochronousCrossing, CSRIsochronousCrossing]:
    """
    Compute the intersections between the isochronous line
    and the CSR-cancellation branches for a symmetric TBA.

    Parameters
    ----------
    outer_length:
        Length of the outer dipoles.

    inner_length:
        Length of the central dipole.

    outer_angle:
        Bend angle of the outer dipoles.

    inner_angle:
        Bend angle of the central dipole.
    """

    if outer_length == 0:
        raise ValueError("outer_length must be nonzero.")

    if outer_angle == 0:
        raise ValueError("outer_angle must be nonzero.")

    length_ratio = inner_length / outer_length
    angle_ratio = inner_angle / outer_angle

    return _csr_isochronous_crossings_from_ratios(
        outer_length=outer_length,
        angle_ratio=angle_ratio,
        length_ratio=length_ratio,
    )

@dataclass
class OptimisationOutput:
    xopt: np.ndarray
    fopt: float
    iter: int
    all_x: np.ndarray
    k1_quads: list[Quadrupole]
    movable_quads: list[Quadrupole]

    # def new_k1s(self):
    #     return {quad.id: k1 for (quad, k1) in zip(self.k1_quads, self.xopt)}

    # def new_quad_shifts(self):
    #     return {quad.id: delta_z for (quad, delta_z) in zip(self.k1_quads, self.xopt[len(self.k1_quads):])}

    def new_k1s(self, index:int = -1):
        return list(zip(self.k1_quads, self.all_x[index]))

    def new_quad_shifts(self, index = -1):
        return list(zip(self.k1_quads, self.all_x[index][len(self.k1_quads):]))

    # def intermediate_k1s(self, index):
    #     return list(zip(self.k1_quads, self.all_x[index]))

    # def intermediate_movable_quads(self, index):
    #     return list(zip(self.movable_quads, self.all_x[index][len(self.k1_quads):]))



# def match_stable(tba, quadrupoles):
#     def error_function(k1s):
#         cost = 0
#         for quadrupole, k1 in zip(quadrupoles, k1s):
#             quadrupole.k1 = k1

#         xtrace, ytrace = tba.traces()

#         cost += xtrace**2
#         cost += ytrace**2

#         print(xtrace, ytrace)

#         return cost

#     initial_guess = [q.k1 for q in quadrupoles]

#     res = fmin(error_function,
#                initial_guess,
#                xtol=1e-4,
#                maxiter=1000,
#                maxfun=1000)

#     return res


def check_achromat_condition(tba):
    m = tba.mx2x2_between_dipoles()
    ((m11, m12), (m21, m22)) = m
    lb = tba.dipole_length()
    achromatic_m12, achromatic_m22 = achromatic_m12_m22(m11, m21, lb)
    print(m12, achromatic_m12)
    print(m22, achromatic_m22)

def achromatic_periodic_twiss(l_drift0: float, m11: np.ndarray, m21: np.ndarray, dipole_length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = l_drift0
    lb = dipole_length
    numerator = np.sqrt(
        (-1 + 2*d*m21 + lb*m21) *
        (
            -4
            + (lb**2) * (m21**2)
            + lb * m21 * (-1 + 2*d*m21)
            + 2*m11 * (-1 + 2*d*m21 + lb*m21)
        )
    )
    denominator = 2 * np.sqrt(- (m21**2) * (2*m11 + lb*m21))
    beta0 =  numerator / denominator
    alpha0 = np.zeros_like(beta0)
    gamma0 = 1/beta0

    return beta0, alpha0, gamma0
