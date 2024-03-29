import abc
import collections
import warnings
import time

from .ci import ConfigurationInteraction
from configuration_interaction.ci_helper import (
    setup_one_body_hamiltonian,
    setup_two_body_hamiltonian,
    construct_one_body_density_matrix,
    construct_two_body_density_matrix,
)


class TimeDependentConfigurationInteraction(metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of a time-dependent
    configuration interaction solver.

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

        self.states = ConfigurationInteraction.setup_ci_space(
            self.excitations,
            self.system.n,
            self.system.l,
            self.system.m,
            self.verbose,
            self.np,
            s=s,
        )
        self.num_states = len(self.states)
        self.setup_initial_hamiltonian()

        # We initialize the last timestep to be None. This will make sure that
        # the first evaluation of update_hamiltonian will be run.
        self.last_timestep = None

    def setup_initial_hamiltonian(self):
        np = self.np

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
            self.system.n,
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
            self.system.n,
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

    def compute_energy(self, current_time, c):
        r"""Function computing the energy of the time-evolved system with a
        time-evolved Hamiltonian.

        .. math:: E(t) = \frac{
                \langle\Psi(t)\rvert \hat{H}(t) \lvert\Psi(t)\rangle
            }{
                \langle\Psi(t)\rvert\Psi(t)\rangle
            }
            = \frac{
                \mathbf{c}^{\dagger}(t) \mathbf{H}(t) \mathbf{c}(t)
            }{
                \mathbf{c}^{\dagger}(t) \mathbf{c}(t)
            },

        where :math:`\lvert\Psi(t)\rangle` is the time-evolved state given by

        .. math:: \lvert\Psi(t)\rangle = c_I(t)\lvert\Phi_I\rangle,

        with :math:`\lvert\Phi_I\rangle` being Slater determinants.

        Parameters
        ----------
        current_time : float
            Current timestep.
        c : np.ndarray
            Coefficient vector at current timestep.

        Returns
        -------
        float
            The time-dependent energy :math:`E(t)`.
        """

        # Update the Hamiltionian to the current timestep
        self.update_hamiltonian(current_time)

        norm = c.conj() @ c
        energy = c.conj() @ self.hamiltonian @ c / norm

        return energy

    def compute_one_body_expectation_value(
        self, current_time, c, mat, tol=1e-5
    ):
        r"""Function computing the expectation value of a one-body operator.
        For a given one-body operator :math:`\hat{A}` by

        .. math:: \langle \hat{A} \rangle = \rho^{q}_{p} A^{p}_{q},

        where :math:`p, q` are general single-particle indices.

        Parameters
        ----------
        current_time : float
            The current time step.
        c : np.ndarray
            The coefficient vector at the current time step.
        mat : np.ndarray
            The one-body operator to evalute, as a matrix. The dimensionality
            of the matrix must be the same as the one-body density matrix,
            i.e., the number of basis functions ``l``.
        tol : float
            Tolerance for the one-body density matrix construction. Default
            value is ``1e-5``.

        Returns
        -------
        complex
            The expectation value of the one-body operator.

        See Also
        --------
        TimeDependentConfigurationInteraction.compute_one_body_density_matrix
        """
        # Update the Hamiltionian to the current timestep
        self.update_hamiltonian(current_time)

        rho_qp = self.compute_one_body_density_matrix(current_time, c, tol=tol)

        return self.np.trace(self.np.dot(rho_qp, mat))

    def compute_two_body_expectation_value(self, current_time, c, op, tol=1e-8):
        r"""Function computing the expectation value of a two-body operator.
        For a given two-body operator :math:`\hat{A}`, we compute the
        expectation value by

        .. math:: \langle \hat{A} \rangle
            = \frac{1}{2} \rho^{rs}_{pq} A^{pq}_{rs},

        where :math:`p, q, r, s` are general single-particle indices.

        Parameters
        ----------
        current_time : float
            The current time step.
        c : np.ndarray
            The coefficient vector at the current time step.
        op : np.ndarray
            The two-body operator to evalute, as a 4-axis array. The
            dimensionality of the array must be the same as the two-body
            density matrix, i.e., the number of basis functions ``l``.
        tol: float
            Tolerance for the trace of the two-body density matrix to be
            :math:`n(n - 1)`, where :math:`n` is the number of particles.
            Default is ``1e-8``.

        Returns
        -------
        complex
            The expectation value of the two-body operator.

        See Also
        --------
        TimeDependentConfigurationInteraction.compute_two_body_density_matrix

        """
        rho_rspq = self.compute_two_body_density_matrix(
            current_time, c, tol=tol
        )

        return 0.5 * self.np.tensordot(
            op, rho_rspq, axes=((0, 1, 2, 3), (2, 3, 0, 1))
        )

    def compute_one_body_density_matrix(self, current_time, c, tol=1e-5):
        r"""Compute one-body density matrix for the time-dependent state
        :math:`\rvert\Psi(t)\rangle`,

        .. math:: \rho^{q}_{p}(t)
            = \langle\Psi(t)\rvert
            \hat{c}^{\dagger}_{p}
            \hat{c}_{q}
            \lvert\Psi(t)\rangle.

        Parameters
        ----------
        current_time : float
            Current timestep.
        c : np.ndarray
            Coefficient vector at current timestep.
        tol : float
            Tolerance of trace warning. Default is ``tol=1e-5``.

        Returns
        -------
        np.ndarray
            The one-body density matrix :math:`\rho^{q}_{p}(t)`.
        """

        rho_qp = self.np.zeros((self.system.l, self.system.l), dtype=c.dtype)
        construct_one_body_density_matrix(rho_qp, self.states, c)

        if self.np.abs(self.np.trace(rho_qp) - self.system.n) > tol:
            warn = "Trace of rho_qp = {0} != {1} = number of particles"
            warn = warn.format(self.np.trace(rho_qp), self.system.n)
            warnings.warn(warn)

        return rho_qp

    def compute_two_body_density_matrix(self, current_time, c, tol=1e-8):
        r"""Function computing the two-body density matrix
        :math:`\rho^{rs}_{pq}` defined by

        .. math:: \rho^{rs}_{pq}
                = \langle\Psi\rvert
                \hat{c}_{p}^{\dagger}
                \hat{c}_{q}^{\dagger}
                \hat{c}_{s}
                \hat{c}_{r}
                \lvert\Psi\rangle,

        where :math:`\lvert\Psi\rangle` is the state defined by the coefficient
        vector ``c``, defined by

        .. math:: \lvert\Psi\rangle = c_{J} \lvert\Phi_J\rangle,

        where :math:`\lvert\Phi_J\rangle` is the :math:`J`'th Slater
        determinant.

        Parameters
        ----------
        current_time : float
            Current timestep.
        c : np.ndarray
            Coefficient vector at current timestep.
        tol: float
            Tolerance for the trace of the two-body density matrix to be
            :math:`n(n - 1)`, where :math:`n` is the number of particles.
            Default is ``1e-8``.

        Returns
        -------
        np.ndarray
            The two-body density matrix.
        """

        rho_rspq = self.np.zeros(
            (self.system.l, self.system.l, self.system.l, self.system.l),
            dtype=c.dtype,
        )

        t0 = time.time()
        construct_two_body_density_matrix(rho_rspq, self.states, c)
        t1 = time.time()

        if self.verbose:
            print(
                "Time spent computing two-body matrix: {0} sec".format(t1 - t0)
            )

        tr_rho = self.np.trace(self.np.trace(rho_rspq, axis1=0, axis2=2))
        error_str = (
            f"Trace of two-body density matrix (rho_rspq = {tr_rho}) does "
            + f"not equal (n * (n - 1) = {self.system.n * (self.system.n - 1)})"
        )
        assert (
            abs(tr_rho - self.system.n * (self.system.n - 1)) < tol
        ), error_str

        return rho_rspq

    def compute_particle_density(self, current_time, c, tol=1e-5):
        r"""Compute particle density :math:`\rho(x, t)` for the time-dependent
        state :math:`\rvert\Psi(t)\rangle`,

        .. math:: \rho(x, t)
            = \phi^{*}_q(x) \rho^{q}_{p}(t) \phi_p(x).

        Parameters
        ----------
        current_time : float
            Current timestep.
        c : np.ndarray
            Coefficient vector at current timestep.
        tol : float
            Tolerance parameter for the one-body density matrix. Default is
            ``tol=1e-5``.

        Returns
        -------
        np.ndarray
            The particle density on the same grid as the single-particle
            functions.
        """

        rho_qp = self.compute_one_body_density_matrix(
            current_time=current_time, c=c, tol=tol
        )

        return self.system.compute_particle_density(rho_qp)

    def compute_overlap(self, current_time, c, c_0):
        r"""Function computing the overlap between two states, viz.

        .. math:: A(t, t_0) = \frac{
                \lvert \langle \Psi(t) \rvert \Psi(t_0) \rangle \rvert^2
            }{
                \langle\Psi(t)\rvert\Psi(t)\rangle
                \langle\Psi(t_0)\rvert\Psi(t_0)\rangle
            }
            =  \frac{
                \lvert \mathbf{c}^{\dagger}(t) \mathbf{c}(t_0) \rvert^2
            }{
                \lvert\mathbf{c}(t)\rvert^2
                \lvert\mathbf{c}(t_0)\rvert^2
            },

        where the :math:`\mathbf{c}(t)` are the coefficient vectors of the
        states :math:`\lvert\Psi(t)\rangle` at specificed time-points.

        Parameters
        ----------
        current_time : float
            Current timestep.
        c : np.ndarray
            Coefficient vector at current timestep.
        c_0 : np.ndarray
            The state to compare overlap with.

        Returns
        -------
        float
            The autocorrelation absolute squared.
        """

        assert c.shape == c_0.shape

        norm_t = c.conj() @ c
        norm_0 = c_0.conj() @ c_0

        overlap = self.np.abs(c.conj() @ c_0) ** 2

        return (overlap / norm_t / norm_0).real

    def update_hamiltonian(self, current_time):
        # Avoid updating the Hamiltonian to the same timestep several times
        if current_time == self.last_timestep:
            return

        self.h = self.system.h_t(current_time)
        self.u = self.system.u_t(current_time)

        if self.system.has_one_body_time_evolution_operator:
            self.one_body_hamiltonian.fill(0)

            # Compute new one-body Hamiltonian
            setup_one_body_hamiltonian(
                self.one_body_hamiltonian,
                self.states,
                self.h,
                self.system.n,
            )

        if self.system.has_two_body_time_evolution_operator:
            self.two_body_hamiltonian.fill(0)

            # Compute new two-body Hamiltonian
            setup_two_body_hamiltonian(
                self.two_body_hamiltonian,
                self.states,
                self.u,
                self.system.n,
            )

        # Empty Hamiltonian matrix
        self.hamiltonian.fill(0)
        self.hamiltonian = self.np.add(
            self.one_body_hamiltonian,
            self.two_body_hamiltonian,
            out=self.hamiltonian,
        )

        # Set last updated timestep to current timestep
        self.last_timestep = current_time

    def __call__(self, current_time, prev_c):
        r"""Function computing the right-hand side of the time-dependent
        Schrödinger equation for the coefficient vector :math:`\mathbf{c}(t)`.
        That is, this function finds the time-derivative of the coefficient
        vector from

        .. math:: \dot{\mathbf{c}} = -i\mathbf{H}(t)\mathbf{c}(t).

        This function is made to resemble the right-hand side functions
        typically used for differential equation solvers.

        Parameters
        ----------
        current_time : float
            Current timestep.
        prev_c : np.ndarray
            Coefficient vector at previous timestep.

        Returns
        -------
        np.ndarray
            Time-derivative of coefficient vector at current timestep.
        """

        # Update Hamiltonian matrix
        self.update_hamiltonian(current_time)

        # Compute dot-product of new Hamiltonian with the previous coefficient
        # vector and multiply with -1j.
        delta_c = -1j * self.np.dot(self.hamiltonian, prev_c)

        # Store current integration step as the last timestep
        self.last_timestep = current_time

        return delta_c
