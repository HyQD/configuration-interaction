import abc
import collections
import warnings
import time

from configuration_interaction.integrators import RungeKutta4
from configuration_interaction.ci_helper import (
    compute_particle_density,
    setup_hamiltonian,
    setup_hamiltonian_brute_force,
    construct_one_body_density_matrix,
    construct_one_body_density_matrix_brute_force,
)


class TimeDependentConfigurationInteraction(metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of a time-dependent
    configuration interaction solver.
    """

    def __init__(self, ci, system, np=None, integrator=None, td_verbose=False, **ci_kwargs):
        if np is None:
            import numpy as np

        self.np = np

        if not "np" in ci_kwargs:
            ci_kwargs["np"] = self.np

        self.verbose = td_verbose

        # Initialize ground state solver
        self.ci = ci(system, **ci_kwargs)
        self.system = system

        self.h = self.system.h
        self.u = self.system.u
        self.n = self.system.n
        self.l = self.system.l
        self.o = self.system.o
        self.v = self.system.v

        if integrator is None:
            integrator = RungeKutta4(np=self.np)

        self.integrator = integrator.set_rhs(self)
        self._c = None

        # Inherit functions from ground state solver
        self.compute_ground_state_energy = lambda K=0: self.ci.energies[K]
        self.compute_ground_state_particle_density = (
            self.ci.compute_particle_density
        )
        self.compute_ground_state_one_body_density_matrix = (
            self.ci.compute_one_body_density_matrix
        )

    def compute_ground_state(self, *args, **kwargs):
        # Setup ci-space in ground state solver
        self.ci.setup_ci_space()
        # Compute ground state
        self.ci.compute_ground_state(*args, **kwargs)
        self.hamiltonian = self.ci.hamiltonian.copy()

    def set_initial_conditions(self, c=None, K=0):
        if c is None:
            # Create copy of the ground state coefficients
            c = self.ci._C[:, K].copy()

        self._c_0 = c.copy()
        self._c = c

    @property
    def c(self):
        return self._c

    def solve(self, time_points):
        n = len(time_points)

        for i in range(n - 1):
            dt = time_points[i + 1] - time_points[i]
            c_t = self.integrator.step(self._c, time_points[i], dt)
            self._c = c_t

            yield self._c

    def compute_energy(self):
        """Function computing the energy of the time-evolved system with a
        time-evolved Hamiltonian.

            E(t) = <\Psi(t)| H(t) |\Psi(t)> / <\Psi(t)|\Psi(t)>
                = c^{*}(t) H(t) c(t) / [c^{*}(t) c(t)].
        """
        norm = self._c.conj() @ self._c
        energy = self._c.conj() @ self.hamiltonian @ self._c / norm

        return energy

    def compute_one_body_density_matrix(self, tol=1e-8):
        rho_qp = self.np.zeros((self.l, self.l), dtype=self._c.dtype)

        density_matrix_function = construct_one_body_density_matrix

        if self.ci.brute_force:
            density_matrix_function = (
                construct_one_body_density_matrix_brute_force
            )

        t0 = time.time()
        density_matrix_function(rho_qp, self.ci.states, self._c)
        t1 = time.time()

        if self.verbose:
            print(
                "Time spent computing one-body matrix: {0} sec".format(t1 - t0)
            )

        if self.np.abs(self.np.trace(rho_qp) - self.system.n) > tol:
            warn = "Trace of rho_qp = {0} != {1} = number of particles"
            warn = warn.format(self.np.trace(rho_qp), self.system.n)
            warnings.warn(warn)

        return rho_qp

    def compute_particle_density(self, tol=1e-8):
        rho_qp = self.compute_one_body_density_matrix(tol=tol)
        rho = compute_particle_density(rho_qp, self.system.spf, np=self.np)

        return rho

    def compute_time_dependent_overlap(self):
        norm_t = self._c.conj() @ self._c
        norm_0 = self._c_0.conj() @ self._c_0

        overlap = self.np.abs(self._c.conj() @ self._c_0) ** 2

        return overlap / norm_t / norm_0

    def __call__(self, prev_c, current_time):
        o, v = self.system.o, self.system.v

        self.h = self.system.h_t(current_time)
        self.u = self.system.u_t(current_time)

        # Empty Hamiltonian matrix
        self.hamiltonian.fill(0)

        # Choose Hamiltonian setup function
        hamiltonian_function = setup_hamiltonian

        if self.ci.brute_force:
            hamiltonian_function = setup_hamiltonian_brute_force

        t0 = time.time()
        # Compute new Hamiltonian
        hamiltonian_function(
            self.hamiltonian, self.ci.states, self.h, self.u, self.n, self.l
        )
        t1 = time.time()

        if self.verbose:
            print(f"Time spent constructing Hamiltonian: {t1 - t0} sec")

        # Compute dot-product of new Hamiltonian with the previous coefficient
        # vector and multiply with -1j.
        new_c = -1j * self.np.dot(self.hamiltonian, prev_c)

        return new_c
