from copy import copy, deepcopy
from dataclasses import dataclass

import numpy as np
import tabulate
from ocelot.cpbd.beam import Particle, ParticleArray, Twiss
from ocelot.cpbd.elements import Quadrupole
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.navi import Navigator
from ocelot.cpbd.optics import trace_z, twiss
from ocelot.cpbd.track import tracking_step
from ocelot.cpbd.transformations import SecondTM


from felarc.csr import LinearCSRTracker


def get_net_angle(seq) -> float:
    angle = 0.
    for element in seq:
        try:
            angle += element.angle
        except AttributeError:
            pass
    return angle


def get_length(seq) -> float:
    angle = 0.
    for element in seq:
        try:
            angle += element.l
        except AttributeError:
            pass
    return angle


def describe_arc_optics(mlat: MagneticLattice, twiss0: Twiss | None = None) -> str:
    df = twiss(mlat, twiss0, return_df=True) # type: ignore
    _, Rs, Ts, s = mlat.transfer_maps(energy=14, output_at_each_step=True) # type: ignore

    # mlat.method = {"global": SecondTM}
    # mlat.update_transfer_maps()


    optics_header = ["Parameter", "Start", "End", "Min", "Max"]

    optics_data = []

    matrix_data = [["R₁₆", f"{Rs[0][0, 5]:.3g}", f"{Rs[-1][0, 5]:.3g}", "", ""],
                   ["R₂₆", f"{Rs[0][1, 5]:.3g}", f"{Rs[-1][1, 5]:.3g}", "", ""],
                   ["R₃₆", f"{Rs[0][2, 5]:.3g}", f"{Rs[-1][2, 5]:.3g}", "", ""],
                   ["R₄₆", f"{Rs[0][3, 5]:.3g}", f"{Rs[-1][3, 5]:.3g}", "", ""],
                   ["R₅₆", f"{Rs[0][4, 5]:.3g}", f"{Rs[-1][4, 5]:.3g}", "", ""],
                   ["---", "---", "---", "", ""],

                   ["R₅₁", f"{Rs[0][4, 0]:.3g}", f"{Rs[-1][4, 0]:.3g}", "", ""],
                   ["R₅₂", f"{Rs[0][4, 1]:.3g}", f"{Rs[-1][4, 1]:.3g}", "", ""],
                   ["R₅₃", f"{Rs[0][4, 2]:.3g}", f"{Rs[-1][4, 2]:.3g}", "", ""],
                   ["R₅₄", f"{Rs[0][4, 3]:.3g}", f"{Rs[-1][4, 3]:.3g}", "", ""],
                   ["R₅₅", f"{Rs[0][4, 4]:.3g}", f"{Rs[-1][4, 4]:.3g}", "", ""],
                   ["R₅₆", f"{Rs[0][4, 5]:.3g}", f"{Rs[-1][4, 5]:.3g}", "", ""],
                   ["T₅₆₆", f"{Ts[0][4, 5, 5]:.3g}", f"{Ts[-1][4, 5, 5]:.3g}", "", ""]]
    optics_data.extend(matrix_data)

    if df is not None:
        df0 = df.iloc[0] # type: ignore
        df1 = df.iloc[-1] # type:ignore
        stable_optics = [["beta_x", df0.beta_x, df1.beta_x, f"{min(df.beta_x):.3g}", f"{max(df.beta_x):.3g}"],
                        ["beta_y", df0.beta_y, df1.beta_y, f"{min(df.beta_y):.3g}", f"{max(df.beta_y):.3g}"],
                        ["alpha_x", df0.alpha_x, df1.alpha_x, f"{min(df.alpha_x):.3g}", f"{max(df.alpha_x):.3g}"],
                        ["alpha_y", df0.alpha_y, df1.alpha_y, f"{min(df.alpha_y):.3g}", f"{max(df.alpha_y):.3g}"],
                        ["Dx", df0.Dx, df1.Dx, f"{min(df.Dx):.3g}", f"{max(df.Dx):.3g}"],
                        ["Dx'", df0.Dxp, df1.Dxp, f"{min(df.Dxp):.3g}", f"{max(df.Dxp):.3g}"]]

        dy = [["Dy", df0.Dy, df1.Dy, f"{min(df.Dy):.3g}", f"{max(df.Dy):.3g}"],
              ["Dy'", df0.Dyp, df1.Dyp, f"{min(df.Dyp):.3g}", f"{max(df.Dyp):.3g}"]]

        # from IPython import embed; embed()

        bend_stops = df[df.id.str.startswith("stop_bend")]
        bend_starts = df[df.id.str.startswith("start_bend")]


        for i, (mu_end_bend, mu_start_next_bend) in enumerate(zip(bend_stops.mux, bend_starts.mux[1:]), start=1):
            print(f"Bend Pair {i} phase advance end to start: {(mu_start_next_bend - mu_end_bend) / np.pi:.3g} 𝜋")

        optics_data.extend(stable_optics)
        optics_data.extend(dy)

    optics_table = tabulate.tabulate(optics_data, headers=optics_header #, tablefmt="html"
                                     )
    return optics_table


def describe_arc_geometry(tba) -> str:
    sequence = tba.mlat.sequence
    angles = [b.angle for b in tba.dipoles]
    sum_angle = sum(angles)
    lattice_header = ["Parameter", "Value"]
    lattice_data = [["Length", f"{get_length(sequence)}"],
                    ["𝜃₁ / deg", f"{angles[0] * 180 / np.pi:.3g}"],
                    ["𝜃₂ / deg", f"{angles[1] * 180 / np.pi:.3g}"],
                    ["𝜃₃ / deg", f"{angles[2] * 180 / np.pi:.3g}"],                    
                    ["𝛴𝜃ᵢ / deg", f"{sum_angle * 180 / np.pi:.3g}"]]


    return tabulate.tabulate(lattice_data, headers=lattice_header)#, tablefmt="html")



@dataclass
class PrincipalTrajectories:
    s: np.ndarray
    cx:np.ndarray
    sx: np.ndarray
    dx: np.ndarray
    cy: np.ndarray
    sy: np.ndarray
    dy: np.ndarray
    

def get_principal_trajectories(mlat: MagneticLattice, num: int) -> PrincipalTrajectories:
    # return C, S and D
    # C is the trajectory that a particle initially at (x, x') = (1, 0) traverses.
    # S is the trajectory that a particle initially at (x, x') = (0, 1) traverses.

    cx0 = Particle(x=1.)
    sx0 = Particle(px=1.)
    dx0 = Particle(p=1.)
    cy0 = Particle(y=1.)
    sy0 = Particle(py=1.)
    dy0 = Particle(p=1.)

    dz = mlat.totalLen / num
    particles_now = [cx0, sx0, dx0, cy0, sy0, dy0]
    navi = Navigator(mlat)
    s = [navi.z0]

    trajectories = [deepcopy(particles_now)]
    for i in range(num):
        tracking_step(mlat, particles_now, dz=dz, navi=navi)
        trajectories.append(deepcopy(particles_now))
        s.append(navi.z0)

    s = np.array(s)
    cx = np.array([t[0].x for t in trajectories])
    sx = np.array([t[1].x for t in trajectories])
    dx = np.array([t[2].x for t in trajectories])
    cy = np.array([t[3].y for t in trajectories])
    sy = np.array([t[4].y for t in trajectories])
    dy = np.array([t[5].y for t in trajectories])

    return PrincipalTrajectories(s, cx, sx, dx, cy, sy, dy)


def calculate_csr_dispersions(mlat):
    pass
    # l_steps = np.diff(s, prepend=0.0)

    # Linear

    # track_with_linear_csr(mlat, 10000)

    # angle_steps = l_steps * curvature

    # bending_radius = 1 / curvature

    # dxi = curvature ** (4/3) * (angle_steps - np.sin(angle_steps))
    # dxip = curvature ** (1/3) * (1 - np.cos(angle_steps))

    # xi = np.cumsum(dxi)
    # xip = np.cumsum(dxip)

    # return xi, xip

    # from IPython import embed; embed()


# # def calculate_csr_dispersion():
# #     pass
