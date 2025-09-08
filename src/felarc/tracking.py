


from ocelot.cpbd.beam import ParticleArray, beam_matching
from ocelot.cpbd.io import load_particle_array
from ocelot.cpbd.navi import Navigator
from ocelot.cpbd.track import track


def load_test_parray(twiss0) -> ParticleArray:
    parray0 = load_particle_array("/Users/stuartwalker/repos/felarc/notebooks/section_STN10.npz")

    beam_matching(parray0, [-5, 5],
                  [twiss0.alpha_x, twiss0.beta_x, 0],
                  [twiss0.alpha_y, twiss0.beta_y, 0],
                  remove_offsets=True)

    return parray0

def track_with_test_beam(mlat, twiss0):
    parray0 = load_test_parray(twiss0)
    navi = Navigator(mlat, unit_step=0.1)
    _, parray1 = track(mlat, parray0.copy(), navi)

