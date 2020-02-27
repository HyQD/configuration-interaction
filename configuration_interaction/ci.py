import abc
import time
from configuration_interaction.ci_helper import (
    BITTYPE,
    BITSTRING_SIZE,
    count_num_states,
    ORDER,
    create_reference_state,
    create_excited_states,
    setup_one_body_hamiltonian,
    setup_two_body_hamiltonian,
    construct_one_body_density_matrix,
    construct_overlap_one_body_density_matrix,
    compute_spin_projection_eigenvalue,
    sort_states,
)


class ConfigurationInteraction(metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of a truncated configuration
    interaction class. All subclasses need to provide the member
    ``excitations`` as this decides the basis of Slater determinants.

    Parameters
    ----------
    system : QuantumSystems
        Quantum systems instance.
    verbose : bool
        Print timer and logging info. Default value is ``False``.
    """

    def __init__(self, system, verbose=False):
        self.verbose = verbose

        self.system = system
        self.np = self.system.np

        self.n = self.system.n
        self.l = self.system.l
        self.m = self.system.m
        self.o = self.system.o
        self.v = self.system.v

        self.states = self.setup_ci_space(
            self.excitations, self.n, self.l, self.m, self.verbose, self.np
        )
        self.num_states = len(self.states)

    @staticmethod
    def setup_ci_space(excitations, n, l, m, verbose, np):
        # Count the reference state
        num_states = 1

        for excitation in excitations:
            num_states += count_num_states(n, m, ORDER[excitation])

        if verbose:
            print(f"Number of states to create: {num_states}")

        # Find the shape of the states array
        # Each state is represented as a bit string padded to the nearest
        # 32-bit boundary
        shape = (
            num_states,
            l // BITSTRING_SIZE + (l % BITSTRING_SIZE > 0),
        )

        if verbose:
            print(f"Size of a state in bytes: {np.dtype(BITTYPE).itemsize * 1}")

        states = np.zeros(shape, dtype=BITTYPE)

        t0 = time.time()
        create_reference_state(n, l, states)

        index = 1
        for excitation in excitations:
            index = create_excited_states(
                n, l, states, index=index, order=ORDER[excitation],
            )

        t1 = time.time()

        if verbose:
            print(
                f"Time spent setting up CI{''.join(excitations)} space: "
                + f"{t1 - t0} sec"
            )

        return sort_states(states)

    @staticmethod
    def compute_states_with_spin_projection(self, states, s, np):
        new_states = []

        for state in states:
            if compute_spin_projection_eigenvalue(state) == s:
                new_states.append(state)

        return sort_states(np.array(new_states))

    def spin_reduce_states(self, s=0):
        """Function removing all states with spin different from ``s``. This
        builds a new ``self.states``-array and updates ``self.num_states``.

        Parameters
        ----------
        s : int
            Spin projection number to keep.
        """

        self.states = self.compute_states_with_spin_projection(
            self.states, s, self.np
        )
        self.num_states = len(self.states)

        if self.verbose:
            print(f"Number of states after spin-reduction: {self.num_states}")

    def compute_ground_state(self, k=None):
        """Function constructing the Hamiltonian of the system without any
        optimization such as the Slater-Condon rules, etc. Having constructed
        the Hamiltonian the function diagonalizes the matrix and stores the
        eigenenergies and the eigenvectors (coefficients) of the system.

        Note that the current solution assumes orthonormal Slater determinants.

        Parameters
        ----------
        k : int
            The number of eigenpairs to compute using an iterative eigensolver.
            Default is ``None``, which means that all eigenpairs are computed.
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
        r"""Function computing the one-body density matrix
        :math:`(\rho_K)^{q}_{p}` defined by

        .. math:: (\rho_K)^{q}_{p} = \langle\Psi_K\rvert \hat{c}_{p}^{\dagger}
                \hat{c}_{q} \lvert\Psi_K\rangle,

        where :math:`\lvert\Psi_K\rangle` is the :math:`K`'th eigenstate of the
        Hamiltonian defined by

        .. math:: \lvert\Psi_K\rangle = C_{JK} \lvert\Phi_J\rangle,

        where :math:`\lvert\Phi_J\rangle` is the :math:`J`'th Slater
        determinant and :math:`C_{JK}` is the coefficient matrix found from
        diagonalizing the Hamiltonian.

        Parameters
        ----------
        K : int
            The eigenstate to compute the one-body density matrix of.

        Returns
        -------
        np.ndarray
            The one-body density matrix.
        """

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
        r"""Function computing the particle density :math:`\rho_K(x)` defined
        by

        .. math:: \rho_K(x) = \phi^{*}_{q}(x) (\rho_K)^{q}_{p} \phi_{p}(x),

        where :math:`\phi_p(x)` are the single-particle functions,
        :math:`(\rho_K)^{q}_{p}` the one-body density matrix for eigenstate
        :math:`K`, and :math:`x` some coordinate space.
        Note the use of the Einstein summation convention in the above expression.

        Parameters
        ----------
        K : int
            The eigenstate to compute the particle density of. Default is ``K =
            0``.

        Returns
        -------
        np.ndarray
            Particle density on the same grid as the single-particle functions.
        """
        rho_qp = self.compute_one_body_density_matrix(K=K)

        return self.system.compute_particle_density(rho_qp)

    def allowed_dipole_transition(self, I, J):
        assert 0 <= I < self.num_states
        assert 0 <= J < self.num_states

        np = self.np

        rho_qp_overlap = np.zeros((self.l, self.l), dtype=self._C.dtype)

        construct_overlap_one_body_density_matrix(
            rho_qp_overlap, self.states, self._C[:, I], self._C[:, J]
        )

        dip = self.system.dipole_moment

        return [np.trace(dip[i] @ rho_qp_overlap) for i in range(len(dip))]

    @property
    def energies(self):
        return self._energies

    @property
    def C(self):
        return self._C

    def compute_energy(self):
        return self._energies[0]
