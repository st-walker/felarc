import numpy as np
import pytest
from felarc.maps import drift, sbend, thin_quadrupole
from felarc.t20 import ARC_ANGLE_RAD
from ocelot.cpbd.elements import Drift, Quadrupole, SBend
from ocelot.cpbd.magnetic_lattice import MagneticLattice

ENERGY = 14
THINNING_FACTOR = 1e-6

# @pytest.fixture

@pytest.fixture
def odrift():
    return Drift(l=1)

@pytest.fixture
def othin_quad():
    return Quadrupole(l=1*THINNING_FACTOR, k1=5/THINNING_FACTOR) # type: ignore

@pytest.fixture
def drift_map(odrift):
    return drift(length=odrift.l, energy=ENERGY)

@pytest.fixture
def thin_quad_map(othin_quad):
    return thin_quadrupole(k1l=othin_quad.k1l)

@pytest.fixture
def osbend():
    return SBend(l=3, angle=ARC_ANGLE_RAD/3)

@pytest.fixture
def sbend_map(osbend):
    return sbend(length=osbend.l, angle=osbend.angle, energy=ENERGY)


def test_sbend_map(osbend):
    ocelot_map = osbend.R(energy=ENERGY)[1]

    # my map
    my_map = sbend(length=osbend.l, angle=osbend.angle, energy=ENERGY)
    my_map = np.array(my_map.rmat).astype(np.float64)

    np.testing.assert_equal(my_map, ocelot_map)

def test_thin_quadrupole_map():
    quad_length = 1
    k1l = 5
    thinning_factor = 1e-6
    oq = Quadrupole(l=quad_length*thinning_factor, k1=k1l / thinning_factor) # type: ignore
    ocelot_map = oq.R(energy=ENERGY)[0]

    my_map = np.array(thin_quadrupole(k1l=k1l).rmat).astype(np.float64)

    np.testing.assert_allclose(my_map, ocelot_map, rtol=2e-6, atol=1e-6)
    
def test_drift_map():
    length = 4
    od = Drift(l=length)
    ocelot_map = od.R(energy=ENERGY)[0]

    my_map = np.array(drift(length=length, energy=ENERGY).rmat).astype(np.float64)

    np.testing.assert_allclose(my_map, ocelot_map, rtol=1e-6)


def test_tba_map(osbend, odrift, othin_quad, sbend_map, drift_map, thin_quad_map):
    mlat = MagneticLattice([osbend,
                            odrift,
                            othin_quad,
                            odrift,
                            othin_quad,
                            odrift,
                            osbend,
                            odrift,
                            othin_quad,
                            odrift,
                            othin_quad,
                            osbend
                            ]) # type: ignore
    ocelot_map = mlat.transfer_maps(energy=ENERGY)[1] # type: ignore

    rmat = (sbend_map
            * drift_map
            * thin_quad_map
            * drift_map
            * thin_quad_map
            * drift_map
            * sbend_map
            * drift_map
            * thin_quad_map
            * drift_map
            * thin_quad_map
            * sbend_map
            ).rmatnp

    np.testing.assert_allclose(rmat, ocelot_map, rtol=1e6)

