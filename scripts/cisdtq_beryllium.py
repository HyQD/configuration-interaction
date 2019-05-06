import numpy as np
import matplotlib.pyplot as plt
import tqdm

from quantum_systems import construct_psi4_system
from quantum_systems.time_evolution_operators import LaserField
from configuration_interaction import TDCISD, TDCISDTQ
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


# System parameters
Be = """
Be 0.0 0.0 0.0
symmetry c1
"""

options = {"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-6}
omega = 0.2
E = 1
laser_duration = 5

system = construct_psi4_system(Be, options)
system.change_to_hf_basis(verbose=True, tolerance=1e-15, max_iters=100)

integrator = GaussIntegrator(s=3, np=np, eps=1e-6)
tdcisdtq = TDCISDTQ(system, integrator=integrator, np=np, verbose=True)
tdcisdtq.compute_ground_state()
print(f"Ground state CISDTQ energy: {tdcisdtq.compute_ground_state_energy()}")
