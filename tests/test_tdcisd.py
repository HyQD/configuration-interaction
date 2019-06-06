import os
import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import construct_psi4_system
from quantum_systems.time_evolution_operators import LaserField
from configuration_interaction import TDCISD
from configuration_interaction.integrators import GaussIntegrator


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
    # System parameters
    He = """
    He 0.0 0.0 0.0
    symmetry c1
    """

    options = {"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-6}
    omega = 2.873_564_3
    E = 100
    laser_duration = 5

    system = construct_psi4_system(He, options)
    system.change_to_hf_basis(verbose=True, tolerance=1e-15)

    integrator = GaussIntegrator(s=3, np=np, eps=1e-6)
    tdcisd = TDCISD(system, integrator=integrator, np=np, verbose=True)
    tdcisd.compute_ground_state()
    assert (
        abs(tdcisd.compute_ground_state_energy() - -2.887_594_831_090_936)
        < 1e-7
    )

    polarization = np.zeros(3)
    polarization[2] = 1
    system.set_time_evolution_operator(
        LaserField(
            LaserPulse(td=laser_duration, omega=omega, E=E),
            polarization_vector=polarization,
        )
    )

    tdcisd.set_initial_conditions()
    dt = 1e-3
    T = 5
    num_steps = int(T // dt) + 1
    t_stop_laser = int(laser_duration // dt) + 1

    time_points = np.linspace(0, T, num_steps)

    td_energies = np.zeros(len(time_points), dtype=np.complex128)
    dip_z = np.zeros(len(time_points))
    td_overlap = np.zeros_like(dip_z)

    rho_qp = tdcisd.compute_one_body_density_matrix(tol=1e6)
    rho_qp_hermitian = 0.5 * (rho_qp.conj().T + rho_qp)

    td_energies[0] = tdcisd.compute_energy()
    dip_z[0] = np.einsum(
        "qp,pq->", rho_qp_hermitian, system.dipole_moment[2]
    ).real
    td_overlap[0] = tdcisd.compute_time_dependent_overlap()

    for i, c in enumerate(tdcisd.solve(time_points)):
        td_energies[i + 1] = tdcisd.compute_energy()

        rho_qp = tdcisd.compute_one_body_density_matrix(tol=1e6)
        rho_qp_hermitian = 0.5 * (rho_qp.conj().T + rho_qp)

        dip_z[i + 1] = np.einsum(
            "qp,pq->", rho_qp_hermitian, system.dipole_moment[2]
        ).real
        td_overlap[i + 1] = tdcisd.compute_time_dependent_overlap()

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
