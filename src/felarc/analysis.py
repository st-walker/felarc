import numpy as np
import tabulate
from ocelot.cpbd.beam import Twiss
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.optics import twiss
from ocelot.cpbd.transformations import SecondTM


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


def describe_arc_optics(mlat: MagneticLattice, tws0: Twiss | None = None) -> str:
    df = twiss(mlat, tws0, return_df=True) # type: ignore
    _, Rs, Ts, s = mlat.transfer_maps(energy=14, output_at_each_step=True) # type: ignore

    mlat.method = {"global": SecondTM}
    mlat.update_transfer_maps()

    df0 = df.iloc[0] # type: ignore
    df1 = df.iloc[-1] # type:ignore

    optics_header = ["Parameter", "Start", "End", "Min", "Max"]

    optics_data = [["R₅₆", f"{Rs[0][4, 5]:.3g}", f"{Rs[-1][4, 5]:.3g}", "", ""],
                   ["T₅₆₆", f"{Ts[0][4, 5, 5]:.3g}", f"{Ts[-1][4, 5, 5]:.3g}", "", ""],
                   ["beta_x", df0.beta_x, df1.beta_x, f"{min(df.beta_x):.3g}", f"{max(df.beta_x):.3g}"],
                   ["beta_y", df0.beta_y, df1.beta_y, f"{min(df.beta_y):.3g}", f"{max(df.beta_y):.3g}"],
                   ["alpha_x", df0.alpha_x, df1.alpha_x, f"{min(df.alpha_x):.3g}", f"{max(df.alpha_x):.3g}"],
                   ["alpha_y", df0.alpha_y, df1.alpha_y, f"{min(df.alpha_y):.3g}", f"{max(df.alpha_y):.3g}"],
                   ["Dx", df0.Dx, df1.Dx, f"{min(df.Dx):.3g}", f"{max(df.Dx):.3g}"],
                   ["Dx'", df0.Dxp, df1.Dxp, f"{min(df.Dxp):.3g}", f"{max(df.Dxp):.3g}"]]

    dy = [["Dy", df0.Dy, df1.Dy, f"{min(df.Dy):.3g}", f"{max(df.Dy):.3g}"],
          ["Dy'", df0.Dyp, df1.Dyp, f"{min(df.Dyp):.3g}", f"{max(df.Dyp):.3g}"]]

    bend_stops = df[df.id.str.startswith("stop_bend")]
    bend_starts = df[df.id.str.startswith("start_bend")]

    
    for i, (mu_end_bend, mu_start_next_bend) in enumerate(zip(bend_stops.mux, bend_starts.mux[1:]), start=1):
        print(f"Bend Pair {i} phase advance end to start: {(mu_start_next_bend - mu_end_bend) / np.pi:.3g} 𝜋")

    optics_data.extend(dy)

    optics_table = tabulate.tabulate(optics_data, headers=optics_header #, tablefmt="html"
                                     )
    return optics_table


def describe_arc_geometry(sequence) -> str:
    try:
        sequence = sequence.sequence
    except AttributeError:
        pass

    net_angle = get_net_angle(sequence)
    net_angle_deg = net_angle * 180. / np.pi
    lattice_header = ["Parameter", "Value"]
    lattice_data = [["Length", get_length(sequence)],
                    ["Angle / rad", f"{net_angle:.3g}"],
                    ["Angle / deg", f"{net_angle_deg:.3g}"]]


    return tabulate.tabulate(lattice_data, headers=lattice_header, tablefmt="html")



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
