import abc
import time
from configuration_interaction.ci_helper import (
    BITTYPE,
    BITSTRING_SIZE,
    NUM_STATES,
    ORDER,
    create_reference_state,
    create_excited_states,
    setup_hamiltonian_brute_force,
    setup_hamiltonian,
    construct_one_body_density_matrix,
    construct_one_body_density_matrix_brute_force,
    compute_particle_density,
)


class ConfigurationInteraction(metaclass=abc.ABCMeta):
    def __init__(
        self, system, excitations, brute_force=False, verbose=False, np=None
    ):
        self.verbose = verbose

        if np is None:
            import numpy as np

        self.np = np
        self.brute_force = brute_force

        self.system = system
        self.excitations = excitations

        self.n = self.system.n
        self.l = self.system.l
        self.m = self.system.m
        self.o = self.system.o
        self.v = self.system.v

        # Count the reference state
        self.num_states = 1

        for excitation in self.excitations:
            self.num_states += NUM_STATES[excitation](self.n, self.m)

        if self.verbose:
            print("Number of states to create: {0}".format(self.num_states))

        # Find the shape of the states array
        # Each state is represented as a bit string padded to the nearest
        # 32-bit boundary
        shape = (self.num_states, self.l // BITSTRING_SIZE + 1)

        if self.verbose:
            print(
                "Size of a state in bytes: {0}".format(
                    np.dtype(BITTYPE).itemsize * 1
                )
            )

        self.states = np.zeros(shape, dtype=BITTYPE)

    # @abc.abstractmethod
    def setup_ci_space(self):
        t0 = time.time()
        create_reference_state(self.n, self.l, self.states)

        index = 1
        for excitation in self.excitations:
            index = create_excited_states(
                self.n,
                self.l,
                self.states,
                index=index,
                order=ORDER[excitation],
            )

        t1 = time.time()

        if self.verbose:
            print(
                f"Time spent setting up CI{''.join(self.excitations)} space: {t1 - t0} sec"
            )

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

        hamiltonian_function = setup_hamiltonian

        if self.brute_force:
            hamiltonian_function = setup_hamiltonian_brute_force

        t0 = time.time()
        hamiltonian_function(
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

        rho_qp = self.np.zeros((self.l, self.l), dtype=self._C.dtype)

        density_matrix_function = construct_one_body_density_matrix

        if self.brute_force:
            density_matrix_function = (
                construct_one_body_density_matrix_brute_force
            )

        t0 = time.time()
        density_matrix_function(rho_qp, self.states, self._C[:, K])
        t1 = time.time()

        if self.verbose:
            print(
                "Time spent computing one-body matrix: {0} sec".format(t1 - t0)
            )

        tr_rho = self.np.trace(rho_qp)
        error_str = (
            f"Trace of one-body density matrix (rho_qp = {tr_rho}) does "
            + f"not equal the number of particles (n = {self.n})"
        )
        assert abs(tr_rho - self.n) < 1e-8, error_str

        return rho_qp

    def compute_particle_density(self, K=0):
        rho_qp = self.compute_one_body_density_matrix(K=K)

        return compute_particle_density(rho_qp, self.system.spf, self.np)

    @property
    def energies(self):
        return self._energies

    @property
    def C(self):
        return self._C


class CIS(ConfigurationInteraction):
    def __init__(self, *args, **kwargs):
        excitations = ["S"]
        args = (*args, excitations)
        super().__init__(*args, **kwargs)


class CID(ConfigurationInteraction):
    def __init__(self, *args, **kwargs):
        excitations = ["D"]
        args = (*args, excitations)
        super().__init__(*args, **kwargs)


class CISD(ConfigurationInteraction):
    def __init__(self, *args, **kwargs):
        excitations = ["S", "D"]
        args = (*args, excitations)
        super().__init__(*args, **kwargs)


class CIDT(ConfigurationInteraction):
    def __init__(self, *args, **kwargs):
        excitations = ["D", "T"]
        args = (*args, excitations)
        super().__init__(*args, **kwargs)


class CISDT(ConfigurationInteraction):
    def __init__(self, *args, **kwargs):
        excitations = ["S", "D", "T"]
        args = (*args, excitations)
        super().__init__(*args, **kwargs)


class CIDTQ(ConfigurationInteraction):
    def __init__(self, *args, **kwargs):
        excitations = ["D", "T", "Q"]
        args = (*args, excitations)
        super().__init__(*args, **kwargs)


class CISDTQ(ConfigurationInteraction):
    def __init__(self, *args, **kwargs):
        excitations = ["S", "D", "T", "Q"]
        args = (*args, excitations)
        super().__init__(*args, **kwargs)
