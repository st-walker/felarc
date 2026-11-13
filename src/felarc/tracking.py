from __future__ import annotations
from ocelot import *

from typing import Literal

from astropy.utils.data import download_file
from ocelot.cpbd.beam import ParticleArray, beam_matching, generate_parray
from ocelot.cpbd.csr import CSR
from ocelot.cpbd.io import load_particle_array_from_npz
from ocelot.cpbd.navi import Navigator
from ocelot.cpbd.optics import Twiss
from ocelot.cpbd.track import track
from ocelot.cpbd.physics_proc import SpontanRadEffects
from ocelot.cpbd.beam import cov_matrix_from_twiss, cov_matrix_to_parray
from ocelot.common.globals import m_e_GeV

_SECTION_STN10_100PC_URL = "https://www.desy.de/fel-beam/s2e/xfel/Nominal/100pC/section_STN10.npz"
_SECTION_STN10_250PC_URL = "https://www.desy.de/fel-beam/s2e/xfel/Nominal/250pC/section_STN10.npz"

def load_test_parray(twiss0, charge: Literal[100, 250] = 250) -> ParticleArray:
    url = _SECTION_STN10_250PC_URL
    if charge == 100:
        url = _SECTION_STN10_100PC_URL

    parray_file = download_file(url, cache=True)
    parray0 = load_particle_array_from_npz(parray_file)

    beam_matching(parray0, [-5, 5],
                  [twiss0.alpha_x, twiss0.beta_x, 0],
                  [twiss0.alpha_y, twiss0.beta_y, 0],
                  remove_offsets=True)

    return parray0

def track_with_test_beam(
    tba,
    twiss0,
    *,
    csr: bool,
    charge: Literal[100, 250],
    rectangle=False,
    isr: bool = False,
) -> ParticleArray:
    parray0 = load_test_parray(twiss0, charge=charge)
    mlat = tba.mlat
    if rectangle:
        parray0 = make_pencil_beam(n=parray0.n, twiss0=twiss0)
    navi = Navigator(mlat, unit_step=0.1)  # type: ignore
    if csr:
        # It is lazy to add csr to teh whole lattice, but it's a lot easier.
        navi.add_physics_proc(
            CSR(sigma_min=parray0.tau().std() * 0.1, rk_traj=True, energy=parray0.E),
            mlat.sequence[0],
            mlat.sequence[-1],
        )
    # from IPython import embed; embed()
    if isr:
        for dipole in tba.dipoles:
            dipole_start = dipole.elements[0]
            dipole_stop = dipole.elements[-1]
            sre = SpontanRadEffects()
            sre.radius = dipole.radius
            sre.type = "dipole"
            navi.add_physics_proc(sre, dipole_start, dipole_stop)

    _, parray1 = track(mlat, parray0.copy(), navi, overwrite_progress=False)

    return parray1


def track_with_parray(tba, parray0, csr: bool) -> ParticleArray:
    # mlat = tba.mlat
    mlat = MagneticLattice(tba.mlat.sequence, method={"global": SecondTM})

    navi = Navigator(mlat, unit_step=0.1) # type: ignore
    if csr:
        # It is lazy to add csr to teh whole lattice, but it's a lot easier.
        navi.add_physics_proc(CSR(sigma_min=parray0.tau().std() * 0.1, rk_traj=False, energy=parray0.E),
                              mlat.sequence[0],
                              mlat.sequence[-1])

    _, parray1 = track(mlat, parray0.copy(), navi, overwrite_progress=False)

    return parray1

def make_pencil_beam(n: int = 500_000, twiss0: Twiss | None = None):
    charge = 250e-12
    parray = generate_parray(nparticles=n,
                             charge=charge,
                             sigma_x=1e-6,
                             sigma_px=1e-6,
                             shape="rect",
                             sigma_tau=5e-6,
                             chirp=0.0,
                             energy=14)

    if twiss0 is not None:
        beam_matching(parray, [-5, 5],
                      [twiss0.alpha_x, twiss0.beta_x, 0],
                      [twiss0.alpha_y, twiss0.beta_y, 0],
                      remove_offsets=True)

    return parray

def make_gaussian_beam(twiss0: Twiss, charge: int = 250, n: int = 200_000) -> ParticleArray:
    sigma_tau = 5e-6 / (250 / charge)
    if charge == 100:
        sigma_tau = 2e-6

    charge *= 1e-12

    # m_e_GeV = 16.0
    energy = 16.0
    # relgamma = (twiss0.E / m_e_GeV)
    relgamma = (energy / m_e_GeV)

    exn = eyn = 0.2e-6
    exg = exn / relgamma
    eyg = eyn / relgamma

    sigma_p = 2e-3 / energy # 2MeV, say

    cov_matrix = cov_matrix_from_twiss(exg, eyg, sigma_tau, sigma_p,
                                       alpha_x=twiss0.alpha_x,
                                       alpha_y=twiss0.alpha_y,
                                       beta_x=twiss0.beta_x,
                                       beta_y=twiss0.beta_y,
                                       dx=twiss0.Dx,
                                       dpx=twiss0.Dxp,
                                       dy=twiss0.Dy,
                                       dpy=twiss0.Dyp
                                       )
    parray = cov_matrix_to_parray([0] * 6, cov=cov_matrix, energy=energy, charge=charge, nparticles=n)
    return parray
