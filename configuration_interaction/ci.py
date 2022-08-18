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
    construct_two_body_density_matrix,
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
    s : int
        Spin projection number to keep. Default is ``None`` and all
        determinants are kept.
    verbose : bool
        Print timer and logging info. Default value is ``False``.
    """

    def __init__(self, system, s=None, verbose=False):
        self.verbose = verbose

        self.system = system
        self.np = self.system.np

        # TODO: This should be inferred from the system
        self.spin_independent = True

        self.n = self.system.n
        self.l = self.system.l
        self.m = self.system.m
        self.o = self.system.o
        self.v = self.system.v

        self.states = self.setup_ci_space(
            self.excitations, self.n, self.l, self.m, self.verbose, self.np, s=s
        )
        self.num_states = len(self.states)

    @staticmethod
    def setup_ci_space(excitations, n, l, m, verbose, np, s=None):
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
                n,
                l,
                states,
                index=index,
                order=ORDER[excitation],
            )

        t1 = time.time()

        if verbose:
            print(
                f"Time spent setting up CI{''.join(excitations)} space: "
                + f"{t1 - t0} sec"
            )

        if s is not None:
            states = (
                ConfigurationInteraction.filter_states_with_spin_projection(
                    states, s, np
                )
            )

            if verbose:
                print(f"Number of states after spin-reduction: {len(states)}")

        return sort_states(states)

    @staticmethod
    def filter_states_with_spin_projection(states, s, np):
        return sort_states(
            np.array(
                list(
                    filter(
                        lambda state: compute_spin_projection_eigenvalue(state)
                        == s,
                        states,
                    )
                )
            )
        )

    def compute_ground_state(self, k=None, decimals=10):
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
        decimals : int
            Number of decimals to use in ``np.round`` when sorting the
            eigenvalues using ``np.lexsort``. This provides a hacky way of
            getting fuzzy sorting. Default is ``10``.
        """

        np = self.np

        assert self.system.h.dtype == self.system.u.dtype
        assert self.system.h.dtype == self.system.spin_2.dtype

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
        )
        t1 = time.time()

        if self.verbose:
            print(
                "Time spent constructing two-body Hamiltonian: {0} sec".format(
                    t1 - t0
                )
            )

        self.spin_z = 0
        self.spin_2 = 0

        if self.spin_independent:
            self.spin_z = np.zeros_like(self.hamiltonian)
            self.spin_2 = np.zeros_like(self.hamiltonian)

            np.testing.assert_allclose(
                self.system.spin_2.shape, self.system.h.shape
            )

            np.testing.assert_allclose(
                self.system.spin_2_tb.shape, self.system.u.shape
            )

            t0 = time.time()
            setup_one_body_hamiltonian(
                self.spin_z,
                self.states,
                self.system.spin_z,
                self.n,
            )
            t1 = time.time()

            if self.verbose:
                print("Time spent constructing S_z: {0} sec".format(t1 - t0))

            t0 = time.time()
            setup_one_body_hamiltonian(
                self.spin_2,
                self.states,
                self.system.spin_2,
                self.n,
            )

            setup_two_body_hamiltonian(
                self.spin_2,
                self.states,
                # Slater-Condon rules for two-body Hamiltonian contains a
                # factor 1/2 somewhere...
                2 * self.system.spin_2_tb,
                self.n,
            )
            t1 = time.time()

            if self.verbose:
                print("Time spent constructing S^2: {0} sec".format(t1 - t0))

        self.hamiltonian += self.one_body_hamiltonian
        self.hamiltonian += self.two_body_hamiltonian

        sum_mat = self.hamiltonian + self.spin_z + self.spin_2

        t0 = time.time()

        eigvals, self._C = np.linalg.eigh(sum_mat)

        self._energies = np.zeros(len(self._C))
        self._s_z = np.zeros_like(self._energies)
        self._s_2 = np.zeros_like(self._energies)

        for i in range(len(self._energies)):
            self._energies[i] = (
                self._C[:, i].T.conj() @ self.hamiltonian @ self._C[:, i]
            ).real
            self._s_z[i] = (
                self._C[:, i].T.conj() @ self.spin_z @ self._C[:, i]
            ).real
            self._s_2[i] = (
                self._C[:, i].T.conj() @ self.spin_2 @ self._C[:, i]
            ).real

        ind = np.lexsort(
            (
                np.round(self._s_z, decimals=decimals),
                np.round(self._s_2, decimals=decimals),
                np.round(self._energies, decimals=decimals),
            )
        )

        self._energies = self._energies[ind]
        self._s_z = self._s_z[ind]
        self._s_2 = self._s_2[ind]
        self._C = self._C[:, ind]

        # if k is None:
        #     self._energies, self._C = np.linalg.eigh(self.hamiltonian)
        # else:
        #     import scipy.sparse.linalg

        #     self._energies, self._C = scipy.sparse.linalg.eigsh(
        #         self.hamiltonian, k=k
        #     )
        t1 = time.time()

        if self.verbose:
            print(
                "Time spent diagonalizing Hamiltonian: {0} sec".format(t1 - t0)
            )
            print(
                f"{self.__class__.__name__} ground state energy: "
                + f"{self.energies[0]}"
            )

        return self

    def compute_one_body_expectation_value(self, mat, K=0):
        r"""Function computing the expectation value of a one-body operator.
        For a given one-body operator :math:`\hat{A}` we compute the
        expectation value by

        .. math:: \langle \hat{A} \rangle = \rho^{q}_{p} A^{p}_{q},

        where :math:`p, q` are general single-particle indices.

        Parameters
        ----------
        mat : np.ndarray
            The one-body operator to evalute, as a matrix. The dimensionality
            of the matrix must be the same as the one-body density matrix,
            i.e., the number of basis functions ``l``.
        K : int
            The eigenstate to use for the one-body density matrix.

        Returns
        -------
        complex
            The expectation value of the one-body operator.

        See Also
        --------
        ConfigurationInteraction.compute_one_body_density_matrix

        """
        rho_qp = self.compute_one_body_density_matrix(K=K)

        return self.np.trace(self.np.dot(rho_qp, mat))

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
            The eigenstate to compute the one-body density matrix from.

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

    def compute_two_body_expectation_value(self, op, K=0, tol=1e-8, asym=True):
        r"""Function computing the expectation value of a two-body operator.
        For a given two-body operator :math:`\hat{A}`, we compute the
        expectation value by

        .. math:: \langle \hat{A} \rangle
            = \frac{1}{2} \rho^{rs}_{pq} A^{pq}_{rs},

        where :math:`p, q, r, s` are general single-particle indices.

        Parameters
        ----------
        op : np.ndarray
            The two-body operator to evalute, as a 4-axis array. The
            dimensionality of the array must be the same as the two-body
            density matrix, i.e., the number of basis functions ``l``.
        K : int
            The eigenstate to use for the two-body density matrix. Default is
            ``0``, i.e., the ground state.
        tol: float
            Tolerance for the trace of the two-body density matrix to be
            :math:`n(n - 1)`, where :math:`n` is the number of particles.
            Default is ``1e-8``.
        asym: bool
            Toggle whether or not ``op`` is anti-symmetrized or not. This
            determines the prefactor when tracing the two-body density matrix
            with the two-body operator. Default is ``True``.

        Returns
        -------
        complex
            The expectation value of the two-body operator.

        See Also
        --------
        ConfigurationInteraction.compute_two_body_density_matrix

        """
        rho_rspq = self.compute_two_body_density_matrix(K=K, tol=tol)

        return (0.5 if asym else 1.0) * self.np.tensordot(
            op, rho_rspq, axes=((0, 1, 2, 3), (2, 3, 0, 1))
        )

    def compute_two_body_density_matrix(self, K=0, tol=1e-8):
        r"""Function computing the two-body density matrix
        :math:`(\rho_K)^{rs}_{pq}` defined by

        .. math:: (\rho_K)^{rs}_{pq}
                = \langle\Psi_K\rvert
                \hat{c}_{p}^{\dagger}
                \hat{c}_{q}^{\dagger}
                \hat{c}_{s}
                \hat{c}_{r}
                \lvert\Psi_K\rangle,

        where :math:`\lvert\Psi_K\rangle` is the :math:`K`'th eigenstate of the
        Hamiltonian defined by

        .. math:: \lvert\Psi_K\rangle = C_{JK} \lvert\Phi_J\rangle,

        where :math:`\lvert\Phi_J\rangle` is the :math:`J`'th Slater
        determinant and :math:`C_{JK}` is the coefficient matrix found from
        diagonalizing the Hamiltonian.

        Parameters
        ----------
        K : int
            The eigenstate to compute the two-body density matrix from.
        tol: float
            Tolerance for the trace of the two-body density matrix to be
            :math:`n(n - 1)`, where :math:`n` is the number of particles.
            Default is ``1e-8``.

        Returns
        -------
        np.ndarray
            The two-body density matrix.
        """

        assert 0 <= K < self.num_states

        rho_rspq = self.np.zeros(
            (self.l, self.l, self.l, self.l), dtype=self._C.dtype
        )

        t0 = time.time()
        construct_two_body_density_matrix(rho_rspq, self.states, self._C[:, K])
        t1 = time.time()

        if self.verbose:
            print(
                "Time spent computing two-body matrix: {0} sec".format(t1 - t0)
            )

        tr_rho = self.np.trace(self.np.trace(rho_rspq, axis1=0, axis2=2))
        error_str = (
            f"Trace of two-body density matrix (rho_rspq = {tr_rho}) does "
            + f"not equal (n * (n - 1) = {self.n * (self.n - 1)})"
        )
        assert abs(tr_rho - self.n * (self.n - 1)) < tol, error_str

        return rho_rspq

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

    def compute_one_body_transition_density_matrix(self, I, J):
        r"""Function constructing the one-body transition density matrix.
        This is defined as the one-body density matrix between two different
        eigenstates, viz.,

        .. math:: (\rho_{IJ})_{qp} \equiv \langle \Psi_I |
            \hat{c}_{p}^{\dagger} \hat{c}_{q} | \Psi_J \rangle,

        where :math:`p, q` are general single-particle indices, and
        :math:`| \Psi_I \rangle` and :math:`| \Psi_J \rangle` are the
        :math:`I` and :math:`J`'th eigenstates.

        Parameters
        ----------
        I : int
            The eigenstate of the bra-state.
        J : int
            The eigenstate of the ket-state.

        Returns
        -------
        np.ndarray
            The one-body transition density matrix.
        """
        assert 0 <= I < self.num_states
        assert 0 <= J < self.num_states

        np = self.np

        rho_qp_overlap = np.zeros((self.l, self.l), dtype=self._C.dtype)

        construct_overlap_one_body_density_matrix(
            rho_qp_overlap, self.states, self._C[:, I], self._C[:, J]
        )

        return rho_qp_overlap

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
