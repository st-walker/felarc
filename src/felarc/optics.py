

import numpy as np
from ocelot.cpbd.elements import Marker
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.navi import Navigator
from ocelot.cpbd.physics_proc import EmptyProc


def transfer_maps_z(lat, step_length, energy=16) -> tuple[np.ndarray, np.ndarray]:
    m1 = Marker()
    m2 = Marker()
    lat_new = MagneticLattice([m1, lat.sequence, m2])

    navi = Navigator(lat_new, unit_step=step_length)
    navi.add_physics_proc(EmptyProc(), m1, m2) # to force the navigator to take steps.

    # Total number of steps 
    total_steps = int(np.ceil(lat.totalLen / step_length))
    # Total number of rmatrices is total_steps+1 because we start at z=0 with eye(6)
    total_r_matrices = total_steps + 1
    # Initialise the arrays we will return for zs and Rs
    r_matrices_at_each_step = np.zeros((total_r_matrices, 6, 6))
    r_matrices_at_each_step[0] = np.eye(6)
    z_at_each_step = np.zeros(total_r_matrices)

    for i, (t_maps, dz, _, _) in enumerate(navi.get_next_step(), start=1):
        # Accumulate the Rmatrices for the >=1 number of TransferMap instances for this step into 1 rmatrix
        this_step_r = np.eye(6)
        for tm in t_maps:
            R = tm.get_params(energy).get_rotated_R()
            this_step_r = R @ this_step_r

        # We want the matrices that go from z=0 to z=s.
        r_matrices_at_each_step[i] = this_step_r @ r_matrices_at_each_step[i - 1]
        z_at_each_step[i] = z_at_each_step[i - 1] + dz

    return z_at_each_step, r_matrices_at_each_step

# def maps_


def get_curvature(sequence, s_sample):
    curvature = np.zeros_like(s_sample)

    s_accumulated = 0
    for element in sequence:
        try:
            length = element.l
        except AttributeError:
            continue

        s_min = s_accumulated
        s_max = s_min + length

        try:
            angle = element.angle
        except AttributeError:
            continue

        mask = (s_sample >= s_min) & (s_sample < s_max)
        curvature[mask] = angle / length
        s_accumulated = s_max

    return curvature
