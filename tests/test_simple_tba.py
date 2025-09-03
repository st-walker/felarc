import felarc.simple_tba as simple

import pytest

from felarc.t20 import

from felarc.t20 import ARC_ANGLE_RAD

@pytest.fixture
def arc():
    dipole_angle = ARC_ANGLE_RAD / 3
    dipole_length = 3

    simple.ArcDefinition
    ad = ArcDefinition(dipole_length=dipole_length,
                       dipole_angle=dipole_angle,
                       l1=1., l2=1, l3=2.,
                       qlength=1.,
                       k11=1,
                       k12=-1)
    return ad

