



import matplotlib.pyplot as plt
import numpy as np


def first_dipole(dipole_length, dipole_angle, m21, s) -> tuple[float, float]:
    lb = dipole_length
    theta = dipole_angle
    radius = dipole_length / theta
    # theta_s = (dipole_length - s) / radius
    # Error in the paper, it should be theta_s = s /radius, because at s = 0 (i.e. at the very start of the
    # tba, of course R16 = R26 = 0 to end.  So ought to be be (s - s0) if we stay in global coordinates.
    theta_s = s / radius
    

    r16 = (lb * theta_s**2 * ((m21*lb)**2 - 3 * m21 * lb + 1 ) / (2 * theta)
           - lb * theta_s * ((m21*lb)**2 - 4 * m21 * lb + 3) / 2)
    r26 = ((lb * theta_s**2 * m21  * (m21 * lb - 2) / theta)
           - theta_s * ((m21*lb)**2 - 3 * m21 * lb + 1))

    # from IPython import embed; embed()
    # r16_second_diple, r26_second_dipole = second_dipole(dipole_length, dipole_angle, m21, 0)

    # return r16 + r16_second_diple, r26 + r26_second_dipole
    return r16, r26

def second_dipole(dipole_length, dipole_angle, m21, s):
    theta = dipole_angle
    radius = dipole_length / theta
    theta_s = (dipole_length - s) / radius
    lb = dipole_length
    r16 = (lb * theta_s**2 * (m21 * lb - 1) / (4 * theta)
           + lb * theta_s * (m21 * lb - 5) / 4 + theta * lb / 2
           )
    r26 = (m21 * lb * theta_s**2 / (2 * theta)) + theta_s * (m21 * lb - 4) / 2 + theta
    # There is an error in the paper in that in the first dipole equation, it does not account also for the impact
    # of the second two dipoles which follow.
    # Compare this with the middle dipole equation which asserts that the end of the dipole, there will still
    # be the independent-of-of-theta_s contribution from the following whole dipole
    # As a hack I just add the first element of the middle dipole to the whole of the first dipole
    # To get something reasonable.  In principle of course I should derive it myself!

    # r16_third_dipole, r26_third_dipole = third_dipole(dipole_length, dipole_angle, 0)

    return r16, r26

def third_dipole(dipole_length, dipole_angle, s):
    theta = dipole_angle
    radius = dipole_length / theta
    theta_s = (dipole_length - s) / radius
    lb = dipole_length

    r16 = lb * theta_s**2 / (2 * theta)
    r26 = theta_s
    return r16, r26

def main():
    dipole_length = 0.9
    drift_length = 1
    bending_angle = 0.176
    m21 = 15 / (2 * dipole_length)
    m21 = 1

    s_in_dipole = np.linspace(0, dipole_length, num=1000)

    z_dipole1 = s_in_dipole
    z_dipole2 = z_dipole1 + drift_length + dipole_length
    z_dipole3 = z_dipole2 + drift_length + dipole_length

    fig, (ax1, ax01, ax2, ax02) = plt.subplots(4, sharex=True)
    r16_d0, r26_d0 = first_dipole(dipole_length, bending_angle, m21, s_in_dipole)
    r16_d1, r26_d1 = second_dipole(dipole_length, bending_angle, m21, s_in_dipole)
    r16_d2, r26_d2 = third_dipole(dipole_length, bending_angle, s_in_dipole)
    ax1.axhline(0)
    ax2.axhline(0)

    ax1.plot(z_dipole1, r16_d0)
    ax2.plot(z_dipole1, r26_d0)

    ax1.plot([z_dipole1[-1], z_dipole2[0]],
              [r16_d0[-1], r16_d1[0]])

    ax2.plot([z_dipole1[-1], z_dipole2[0]],
              [r26_d0[-1], r26_d1[0]])

    ax1.plot(z_dipole2, r16_d1)
    ax2.plot(z_dipole2, r26_d1)

    ax1.plot([z_dipole2[-1], z_dipole3[0]],
              [r16_d1[-1], r16_d2[0]])

    


    ax2.plot([z_dipole2[-1], z_dipole3[0]],
              [r26_d1[-1], r26_d2[0]])


    ax1.plot(z_dipole3, r16_d2)
    ax2.plot(z_dipole3, r26_d2)

    ax02.set_xlabel("$s$ / m")
    ax1.set_ylabel(r"$R_{16}^{s\rightarrow s_f}$ / m")
    ax2.set_ylabel(r"$R_{26}^{s\rightarrow s_f}$ / rad")

    plt.show()

    pass


if __name__ == "__main__":
    main()

    # Derive Hill's equations
    # Derive R56.
