from dataclasses import dataclass
from functools import reduce
from operator import matmul

import numpy as np
from ocelot.cpbd.elements import (Drift, Hcor, Quadrupole, RBend, SBend,
                                  Sextupole, Vcor)
from ocelot.cpbd.navi import Navigator
from ocelot.cpbd.optics import twiss
from scipy.constants import c, e, mu_0, physical_constants, pi

from felarc.optics import get_curvature, transfer_maps_z

from .constants import m_e_GeV

ELECTRON_RADIUS, *_ = physical_constants["classical electron radius"]

def linear_csr_k(charge: float, bunch_length):
    # erms per rho^(2/3) per disatnce
    rho = bending_radius
    sigma_z = bunch_length
    q = charge
    return 0.2459 * e * q * mu_0 * c**2 / ((4 * pi * sigma_z **(4/3)))


@dataclass
class CSRVector:
    x: float
    px: float
    dp: float
    dp_csr: float
    k: float


# def dipole_csr_matrix(angle, bending_radius):
#     theta = angle
#     rho = bending_radius
#     from IPython import embed; embed()
#     return np.array([
#         [np.cos(theta), rho * np.sin(theta), rho * (1 - np.cos(theta)), rho * (1 - np.cos(theta)), rho**(4/3) * (theta - np.sin(theta))],
#         [-np.sin(theta) / rho, np.cos(theta), np.sin(theta), np.sin(theta), rho**(1/3) * (1 - np.cos(theta))],
#         [0, 0, 1, 0, 0],
#         [0, 0, 0, 1, rho**(1/3) * theta],
#         [0, 0, 0, 0, 1]
#     ])

def _reduce_rmats(rmats):
    return reduce(matmul, rmats, np.eye(6))

def drift_csr_matrix(energy, length):
    r6d = _reduce_rmats(Drift(length).R(energy=energy))
    horizontal = r6d[:2, :2]
    dx = r6d[0, 4]
    dxp = r6d[0, 5]
    result = np.eye(5)
    result[:2, :2] = horizontal
    result[0, 3] = result[0, 4] = dx
    result[1, 3] = result[1, 4] = dxp
    return result

def quadrupole_csr_matrix(energy, length, k1):
    r6d = _reduce_rmats(Quadrupole(l=length, k1=k1).R(energy=energy))
    horizontal = r6d[:2, :2]
    dx = r6d[0, 5]
    dxp = r6d[1, 5]
    result = np.eye(5)
    result[:2, :2] = horizontal
    result[0, 3] = result[0, 4] = dx
    result[1, 3] = result[1, 4] = dxp
    return result

def sbend_csr_matrix(energy, length, angle):
    r6d = _reduce_rmats(SBend(l=length, angle=angle).R(energy=energy))
    horizontal = r6d[:2, :2]
    dx = r6d[0, 5]
    dxp = r6d[1, 5]
    result = np.eye(5)
    result[:2, :2] = horizontal
    result[0, 2] = result[0, 3] = dx
    result[1, 2] = result[1, 3] = dxp
    rho = length / angle
    result[0, 4] = rho**(4/3) * (angle - np.sin(angle))
    result[1, 4] = rho**(1/3) * (1 - np.cos(angle))
    result[3, 4] = rho**(1/3) * angle
    return result

def _linear_csr_matrices(mlat, energy, ds = None):

    if ds is None: # Step just to the ends of each element in other words...
        ds = np.inf

    s0 = 0.0
    s1 = 0.0
    element_start = s0
    all_s = [s0]
    matrices = [np.eye(5)]
    for element in mlat.sequence:
        # Skip elements like Markers:
        try:
            length = element.l
        except AttributeError:
            continue

        if not element.l:
            continue

        # Get end of the element we are currently in in S coordinate.
        element_end = element_start + length
        while s0 < element_end and not np.isclose(s0, element_end):
            step_size = ds
            s1 = s0 + step_size
            # The end of the element in s coordinates
            # Does the step go beyond the end of the element? If so, we only track
            # to the end of the element.
            if s1 > element_end:
                step_size = element_end - s0
                s1 = element_end

            if isinstance(element, (SBend, RBend)):
                csrmat = sbend_csr_matrix(energy, step_size, element.angle * (step_size/length))
            elif isinstance(element, (Drift, Sextupole, (Vcor, Hcor))):
                csrmat = drift_csr_matrix(energy, step_size)
            elif isinstance(element, Quadrupole):
                csrmat = quadrupole_csr_matrix(energy, step_size, element.k1)
            else:
                from IPython import embed; embed()

            s0 += step_size
            all_s.append(s1)
            matrices.append(csrmat)

        element_start = element_end

    return all_s, matrices


class LinearCSRTracker:
    def __init__(self, mlat, k: float = 1):
        self._mlat = mlat
        self.k = k

    def set_kfactor(self, charge, bunch_length):
        self.k = linear_csr_k(charge, bunch_length)

    def track(self, x0, px0, dp0, dp_csr0, energy: float = 16, ds=None):
        x0 = np.array([x0, px0, dp0, dp_csr0, self.k]).T
        tracked_particle = [x0]
        s, matrices = _linear_csr_matrices(self._mlat, energy=energy, ds=ds)

        for matrix in matrices[1:]:
            tracked_particle.append(matrix @ tracked_particle[-1])

        tracked_particle = np.array(tracked_particle)
        result = CSRVector(x=tracked_particle[..., 0],
                  px=tracked_particle[..., 1],
                  dp=tracked_particle[..., 2],
                  dp_csr=tracked_particle[..., 3],
                  k=tracked_particle[..., 4])

        return s, result

# class EmittanceTracker:
#     def __init__(self, mlat, k: float = 1)


# def mean_delta_x_squared(mlat, k: float = 1):
#     for mlat
#     # Integral of R_16


def gaussian_bunch_espread_per_unit_length(curvature, charge, bunch_length, energy):
    """
    Computes dσ_δ/ds ≈ 0.246 * Nb * (rc * θ^(2/3)) / (γ * σ_z^(4/3) * L_B^(2/3))
    All args can be floats or NumPy arrays.
    """
    nb = charge / e
    rc = ELECTRON_RADIUS
    curvature23 = curvature**2/3
    gamma = energy / m_e_GeV
    sigma_z = bunch_length

    return 0.246 * nb * (rc * curvature23) / (gamma * sigma_z**(4/3))


def gaussian_bunch_emittance_growth(mlat, emitn0, charge, bunch_length, energy, twiss0=None):
    relgamma = energy / m_e_GeV
    emitg0 = emitn0 / relgamma

    z, rmatrices = transfer_maps_z(mlat, step_length=0.1)
    rend = rmatrices[-1]
    inverses = np.array([np.linalg.inv(r) for r in rmatrices])
    r_to_end = rend @ inverses

    curvature = get_curvature(mlat.sequence, s_sample=z)
    dipole_mask = curvature.copy()
    dipole_mask[dipole_mask > 0] = 1

    # This allows for varying dipole strengths.
    espread_per_unit_length = gaussian_bunch_espread_per_unit_length(curvature, charge, bunch_length, energy)
    despread_ds = espread_per_unit_length

    rms_delta_x2 = np.trapz(dipole_mask * r_to_end[:, 0, 5] * despread_ds, x=z) ** 2
    rms_delta_xp2 = np.trapz(dipole_mask * r_to_end[:, 1, 5] * despread_ds, x=z) ** 2
    rms_delta_xxp = (np.trapz(dipole_mask * r_to_end[:, 0, 5] * despread_ds, x=z)
                     * np.trapz(dipole_mask * r_to_end[:, 1, 5] * despread_ds, x=z))

    twisses = twiss(mlat, twiss0) # type: ignore

    twiss1 = twisses[-1]

    emitg1 = np.sqrt(emitg0**2 + emitg0 * (twiss1.beta_x * rms_delta_xp2**2
                                           + 2 * twiss1.alpha_x * rms_delta_xxp
                                           + twiss1.gamma_x * rms_delta_x2**2))
    emitn1 = emitg1 * relgamma

    return emitn1
