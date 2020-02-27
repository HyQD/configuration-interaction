import abc
import collections
import warnings
import time

from .ci import ConfigurationInteraction
from configuration_interaction.ci_helper import (
    setup_one_body_hamiltonian,
    setup_two_body_hamiltonian,
    construct_one_body_density_matrix,
)


class TimeDependentConfigurationInteraction(metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of a time-dependent
    configuration interaction solver.

    Parameters
    ----------
    system : QuantumSystems
        Quantum systems instance.
    init_state : ConfigurationInteraction
        An initial state for the time-evolution.
    k : int
        Which eigenstate from ``init_state`` to use as the initial state.
        The default is ``0``, i.e., the ground state. Note that this argument
        is ignored if ``init_state == None``.
    s : int
        Spin projection number to keep. Default is ``None`` and all
        determinants are kept.
    verbose : bool
        Print timer and logging info. Default value is ``False``.
    """

    def __init__(self, system, init_state=None, k=0, s=None, verbose=False):
        self.verbose = verbose

        self.system = system
        self.np = self.system.np

        if init_state is None:
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
            self._c = self.np.zeros(self.num_states, dtype=self.system.h.dtype)
            self._c[0] = 1

        else:
            assert self.excitations == init_state.excitations

            self.states = init_state.states
            self.num_states = len(self.states)
            self.hamiltonian = init_state.hamiltonian
            self.one_body_hamiltonian = init_state.one_body_hamiltonian
            self.two_body_hamiltonian = init_state.two_body_hamiltonian
            self._c = self.init_state.C[:, k]

    def setup_initial_hamiltonian(self):
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
            self.system.l,
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
            self.system.l,
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

    @property
    def c(self):
        return self._c

    def compute_energy(self):
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

        Returns
        -------
        float
            The time-dependent energy :math:`E(t)`.
        """
        norm = self._c.conj() @ self._c
        energy = self._c.conj() @ self.hamiltonian @ self._c / norm

        return energy

    def compute_one_body_density_matrix(self, tol=1e-5):
        r"""Compute one-body density matrix for the time-dependent state
        :math:`\rvert\Psi(t)\rangle`,

        .. math:: \rho^{q}_{p}(t)
            = \langle\Psi(t)\rvert
            \hat{c}^{\dagger}_{p}
            \hat{c}_{q}
            \lvert\Psi(t)\rangle.

        Parameters
        ----------
        tol : float
            Tolerance of trace warning. Default is ``tol=1e-5``.

        Returns
        -------
        np.ndarray
            The one-body density matrix :math:`\rho^{q}_{p}(t)`.
        """

        rho_qp = self.np.zeros((self.l, self.l), dtype=self._c.dtype)
        construct_one_body_density_matrix(rho_qp, self.states, self._c)

        if self.np.abs(self.np.trace(rho_qp) - self.system.n) > tol:
            warn = "Trace of rho_qp = {0} != {1} = number of particles"
            warn = warn.format(self.np.trace(rho_qp), self.system.n)
            warnings.warn(warn)

        return rho_qp

    def compute_particle_density(self, tol=1e-5):
        r"""Compute particle density :math:`\rho(x, t)` for the time-dependent
        state :math:`\rvert\Psi(t)\rangle`,

        .. math:: \rho(x, t)
            = \phi^{*}_q(x) \rho^{q}_{p}(t) \phi_p(x).

        Parameters
        ----------
        tol : float
            Tolerance parameter for the one-body density matrix. Default is
            ``tol=1e-5``.

        Returns
        -------
        np.ndarray
            The particle density on the same grid as the single-particle
            functions.
        """

        rho_qp = self.compute_one_body_density_matrix(tol=tol)

        return self.system.compute_particle_density(rho_qp)

    def compute_time_dependent_overlap(self, c_0):
        r"""Function computing the autocorrelation by

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
        c_0 : np.ndarray
            The state to compare overlap with.

        Returns
        -------
        float
            The autocorrelation absolute squared.
        """

        assert self._c.shape == c_0.shape

        norm_t = self._c.conj() @ self._c
        norm_0 = c_0.conj() @ c_0

        overlap = self.np.abs(self._c.conj() @ c_0) ** 2

        return (overlap / norm_t / norm_0).real

    def solout(self, current_time, current_c):
        """Function to be called by the integrator after every successful step.

        Parameters
        ----------
        current_time : float
            Current time step.
        current_c : np.ndarray
            Current coefficient vector.

        See Also
        --------
        scipy.integrate.ode.set_solout
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.set_solout.html#scipy.integrate.ode.set_solout
        """

        self._c = current_c
        self.update_hamiltonian(current_time)

    def update_hamiltonian(self, current_time):
        self.h = self.system.h_t(current_time)
        self.u = self.system.u_t(current_time)

        if self.system.has_one_body_time_evolution_operator:
            self.one_body_hamiltonian.fill(0)

            # Compute new one-body Hamiltonian
            setup_one_body_hamiltonian(
                self.one_body_hamiltonian, self.states, self.h, self.n, self.l,
            )

        if self.system.has_two_body_time_evolution_operator:
            self.two_body_hamiltonian.fill(0)

            # Compute new two-body Hamiltonian
            setup_two_body_hamiltonian(
                self.two_body_hamiltonian, self.states, self.u, self.n, self.l,
            )

        # Empty Hamiltonian matrix
        self.hamiltonian.fill(0)
        self.hamiltonian = self.np.add(
            self.one_body_hamiltonian,
            self.two_body_hamiltonian,
            out=self.hamiltonian,
        )

    def __call__(self, prev_c, current_time):
        r"""Function computing the right-hand side of the time-dependent
        Schrödinger equation for the coefficient vector :math:`\mathbf{c}(t)`.
        That is, this function finds the time-derivative of the coefficient
        vector from

        .. math:: \dot{\mathbf{c}} = -i\mathbf{H}(t)\mathbf{c}(t).

        This function is made to resemble the right-hand side functions
        typically used for differential equation solvers.

        Parameters
        ----------
        prev_c : np.ndarray
            Coefficient vector at previous time-step.
        current_time : float
            Current time-step.

        Returns
        -------
        np.ndarray
            Time-derivative of coefficient vector at current time-step.
        """

        # Update Hamiltonian matrix
        self.update_hamiltonian(current_time)

        # Compute dot-product of new Hamiltonian with the previous coefficient
        # vector and multiply with -1j.
        new_c = -1j * self.np.dot(self.hamiltonian, prev_c)

        return new_c
