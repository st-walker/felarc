import numpy as np
from ocelot.cpbd.magnetic_lattice import MagneticLattice

def _sequence_length(sequence) -> float:
    l = 0.
    for element in sequence:
        try:
            l += element.l
        except AttributeError:
            pass
    return l


# def make_dipole_mask(sequence, ssampl:)

def make_curvature_mask(sequence, ssample) -> np.array():
    s = 0.0
    total_length = _sequence_length(sequence)


    for element in sequence:
        try:
            curvature = element.angle / element.l
        except AttributeError:
            

    for s in ssample:
        if s < 

    for element in sequence:
        try:
            angle = element.angle
        except AttributeError:
            
