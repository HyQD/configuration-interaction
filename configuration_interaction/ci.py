import abc
import time
from configuration_interaction.ci_helper import (
    BITTYPE,
    BITSTRING_SIZE,
    num_states,
    ORDER,
    create_reference_state,
    create_excited_states,
    setup_one_body_hamiltonian,
    setup_two_body_hamiltonian,
    construct_one_body_density_matrix,
    compute_particle_density,
    compute_spin_projection_eigenvalue,
    sort_states,
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

        # Count the reference state
        self.num_states = 1

        for excitation in self.excitations:
            self.num_states += num_states(self.n, self.m, ORDER[excitation])

        if self.verbose:
            print("Number of states to create: {0}".format(self.num_states))

        # Find the shape of the states array
        # Each state is represented as a bit string padded to the nearest
        # 32-bit boundary
        shape = (
            self.num_states,
            self.l // BITSTRING_SIZE + (self.l % BITSTRING_SIZE > 0),
        )

        if self.verbose:
            print(
                "Size of a state in bytes: {0}".format(
                    np.dtype(BITTYPE).itemsize * 1
                )
            )

        self.states = np.zeros(shape, dtype=BITTYPE)
        self._setup_ci_space()

    def _setup_ci_space(self):
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

        self.states = sort_states(self.states)

    def spin_reduce_states(self, s=0):
        """Function removing all states with spin different from `s`. This
        builds a new `self.states`-array and updates `self.num_states`.

        Parameters
        ----------
        s : int
            Spin projection number to keep.
        """
        np = self.np

        new_states = []

        for state in self.states:
            if compute_spin_projection_eigenvalue(state) == s:
                new_states.append(state)

        self.states = sort_states(np.array(new_states))
        self.num_states = len(self.states)

        if self.verbose:
            print(f"Number of states after spin-reduction: {self.num_states}")

    def compute_ground_state(self, k=None):
        """Function constructing the Hamiltonian of the system without any
        optimization such as the Slater-Condon rules, etc. Having constructed
        the Hamiltonian the function diagonalizes the matrix and stores the
        eigenenergies and the eigenvectors (coefficients) of the system.

        Note that the current solution assumes orthonormal Slater determinants.
        """
        np = self.np

        assert self.system.h.dtype == self.system.u.dtype

        self.hamiltonian = np.zeros(
            (self.num_states, self.num_states), dtype=self.system.h.dtype
        )
        self.one_body_hamiltonian = np.zeros_like(self.hamiltonian)
        self.two_body_hamiltonian = np.zeros_like(self.hamiltonian)

        t0 = time.time()
        setup_one_body_hamiltonian(
            self.one_body_hamiltonian,
            self.states,
            self.system.h,
            self.n,
            self.l,
        )
        t1 = time.time()

        if self.verbose:
            print(
                "Time spent constructing one-body Hamiltonian: {0} sec".format(
                    t1 - t0
                )
            )

        t0 = time.time()
        setup_two_body_hamiltonian(
            self.two_body_hamiltonian,
            self.states,
            self.system.u,
            self.n,
            self.l,
        )
        t1 = time.time()

        if self.verbose:
            print(
                "Time spent constructing two-body Hamiltonian: {0} sec".format(
                    t1 - t0
                )
            )

        self.hamiltonian += self.one_body_hamiltonian
        self.hamiltonian += self.two_body_hamiltonian

        t0 = time.time()
        if k is None:
            self._energies, self._C = np.linalg.eigh(self.hamiltonian)
        else:
            import scipy.sparse.linalg

            self._energies, self._C = scipy.sparse.linalg.eigsh(
                self.hamiltonian, k=k
            )
        t1 = time.time()

        if self.verbose:
            print(
                "Time spent diagonalizing Hamiltonian: {0} sec".format(t1 - t0)
            )
            print(
                f"{self.__class__.__name__} ground state energy: "
                + f"{self.energies[0]}"
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

        t0 = time.time()
        construct_one_body_density_matrix(rho_qp, self.states, self._C[:, K])
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


def excitation_string_handler(excitations):
    if isinstance(excitations, str):
        excitations = excitations.upper()

        if excitations.startswith("CI"):
            excitations = excitations[2:]

        excitations = [excitation for excitation in excitations]

    for excitation in excitations:
        assert excitation in ORDER, f'"{excitation}" is not a supported order'

    return list(map(lambda x: x.upper(), excitations))


def get_ci_class(excitations):
    """Function constructing a truncated CI-class with the specified
    excitations.

    Parameters
    ----------
    excitations : str, iterable
        The specified excitations to use in the CI-class. For example, to create
        a CISD class both `excitations="CISD"` and `excitations=["S", "D"]` are
        valid.

    Returns
    -------
    ci_class : class
        A subclass of `ConfigurationInteraction`.
    """
    excitations = excitation_string_handler(excitations)
    class_name = "CI" + "".join(excitations)

    ci_class = type(
        class_name, (ConfigurationInteraction,), dict(excitations=excitations)
    )

    return ci_class


CIS = get_ci_class("CIS")
CID = get_ci_class("CID")
CISD = get_ci_class("CISD")
CIDT = get_ci_class("CIDT")
CISDT = get_ci_class("CISDT")
CIDTQ = get_ci_class("CIDTQ")
CISDTQ = get_ci_class("CISDTQ")
