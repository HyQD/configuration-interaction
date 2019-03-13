import abc
import time
from configuration_interaction.ci_helper import (
    setup_hamiltonian_brute_force,
    construct_one_body_density_matrix,
)


class ConfigurationInteraction(metaclass=abc.ABCMeta):
    def __init__(self, system, verbose=False, np=None):
        self.verbose = verbose

        if np is None:
            import numpy as np

        self.np = np

        self.system = system

        self.n = self.system.n
        self.l = self.system.l
        self.m = self.system.m
        self.o = self.system.o
        self.v = self.system.v

    @abc.abstractmethod
    def setup_ci_space(self):
        pass

    def compute_ground_state(self):
        """Function constructing the Hamiltonian of the system without any
        optimization such as the Slater-Condon rules, etc. Having constructed
        the Hamiltonian the function diagonalizes the matrix and stores the
        eigenenergies and the eigenvectors (coefficients) of the system.

        Note that the current solution assumes orthonormal Slater determinants.
        """
        np = self.np

        self.hamiltonian = np.zeros(
            (self.num_states, self.num_states), dtype=np.complex128
        )

        t0 = time.time()
        setup_hamiltonian_brute_force(
            self.hamiltonian,
            self.states,
            self.system.h,
            self.system.u,
            self.n,
            self.l,
        )
        t1 = time.time()

        if self.verbose:
            print(
                "Time spent constructing Hamiltonian: {0} sec".format(t1 - t0)
            )

        t0 = time.time()
        self._energies, self._C = np.linalg.eigh(self.hamiltonian)
        t1 = time.time()

        if self.verbose:
            print(
                "Time spent diagonalizing Hamiltonian: {0} sec".format(t1 - t0)
            )

    def compute_one_body_density_matrix(self, K=0):
        r"""Function computing the one-body density matrix \rho^{q}_{p} defined
        by

            \rho^{q}_{p} = <\Psi_K|c_{p}^{\dagger} c_{q} |\Psi_K>,

        where |\Psi_K> is the K'th eigenstate of the Hamiltonian defined by

            |\Psi_K> = C_{JK} |\Phi_J>,

        where |\Phi_J> is the J'th Slater determinant and C_{JK} is the JK
        element of the coefficient matrix found from diagonalizing the
        Hamiltonian."""

        assert 0 <= K < self.num_states

        rho_qp = np.zeros((self.l, self.l), dtype=np.complex128)
        construct_one_body_density_matrix(rho_qp, self.states, self._C[:, K])

        return rho_qp

    @property
    def energies(self):
        return self._energies

    @property
    def C(self):
        return self._C
