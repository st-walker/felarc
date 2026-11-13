from dataclasses import dataclass
from math import pi

from .constants import m_e_GeV

ARC_ANGLE_DEG = 6.0
ARC_ANGLE_RAD = ARC_ANGLE_DEG * pi / 180

EMITN = 1e-6
ENERGY = 16
EMITG = ENERGY / (ENERGY / m_e_GeV)
CHARGE_100 = 100e-12
CHARGE_250 = 250e-12
GAUSSIAN_BUNCH_LENGTH_6KA = 5e-6

@dataclass
class BeamParameters:
    emitn: float
    energy: float
    charge: float


# GAUSSIAN_BEAM_6KA_100 = BeamParameters(1e-6, 
