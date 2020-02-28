import os
import numpy as np

from scipy.integrate import ode

from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import LaserField

from configuration_interaction import CISD, TDCISD

from gauss_integrator import GaussIntegrator


class LaserPulse:
    def __init__(self, t0=0, td=5, omega=0.1, E=0.03):
        self.t0 = t0
        self.td = td
        self.omega = omega
        self.E = E  # Field strength

    def __call__(self, t):
        T = self.td
        delta_t = t - self.t0
        return (
            -(np.sin(np.pi * delta_t / T) ** 2)
            * np.heaviside(delta_t, 1.0)
            * np.heaviside(T - delta_t, 1.0)
            * np.cos(self.omega * delta_t)
            * self.E
        )


def test_tdcisd():
    omega = 2.873_564_3
    E = 100
    laser_duration = 5

    system = construct_pyscf_system_rhf(
        molecule="he 0.0 0.0 0.0", basis="cc-pvdz"
    )

    cisd = CISD(system, verbose=True).compute_ground_state()
    assert abs(cisd.compute_energy() - -2.887_594_831_090_936) < 1e-7

    tdcisd = TDCISD(system, verbose=True)
    r = ode(tdcisd).set_integrator("GaussIntegrator", s=3, eps=1e-6)
    r.set_initial_value(cisd.C[:, 0])

    assert abs(tdcisd.compute_energy(r.t, r.y) - -2.887_594_831_090_936) < 1e-7

    polarization = np.zeros(3)
    polarization[2] = 1
    system.set_time_evolution_operator(
        LaserField(
            LaserPulse(td=laser_duration, omega=omega, E=E),
            polarization_vector=polarization,
        )
    )

    dt = 1e-3
    T = 5
    num_steps = int(T // dt) + 1
    t_stop_laser = int(laser_duration // dt) + 1

    time_points = np.linspace(0, T, num_steps)

    td_energies = np.zeros(len(time_points), dtype=np.complex128)
    dip_z = np.zeros(len(time_points))
    td_overlap = np.zeros_like(dip_z)

    i = 0

    while r.successful() and r.t <= T:
        td_energies[i] = tdcisd.compute_energy(r.t, r.y)

        rho_qp = tdcisd.compute_one_body_density_matrix(r.t, r.y, tol=1e-3)
        rho_qp_hermitian = 0.5 * (rho_qp.conj().T + rho_qp)

        dip_z[i] = np.einsum(
            "qp,pq->", rho_qp_hermitian, system.dipole_moment[2]
        ).real
        td_overlap[i] = tdcisd.compute_time_dependent_overlap(
            r.t, r.y, cisd.C[:, 0]
        )

        r.integrate(r.t + dt)
        i += 1

    import matplotlib.pyplot as plt

    plt.plot(
        time_points, td_energies.real, label="New",
    )
    plt.plot(
        time_points,
        np.loadtxt(
            os.path.join("tests", "dat", "tdcisd_helium_energies_real.dat")
        ),
        label="Old",
    )
    plt.legend()
    plt.show()

    np.testing.assert_allclose(
        td_energies.real,
        np.loadtxt(
            os.path.join("tests", "dat", "tdcisd_helium_energies_real.dat")
        ),
        atol=1e-7,
    )

    np.testing.assert_allclose(
        td_overlap,
        np.loadtxt(os.path.join("tests", "dat", "tdcisd_helium_overlap.dat")),
        atol=1e-7,
    )

    np.testing.assert_allclose(
        dip_z,
        np.loadtxt(os.path.join("tests", "dat", "tdcisd_helium_dipole_z.dat")),
        atol=1e-7,
    )


if __name__ == "__main__":
    test_tdcisd()
