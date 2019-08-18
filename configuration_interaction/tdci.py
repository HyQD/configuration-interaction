import abc
import collections
import warnings
import time

from configuration_interaction import get_ci_class, excitation_string_handler
from configuration_interaction.integrators import RungeKutta4
from configuration_interaction.ci_helper import (
    compute_particle_density,
    setup_one_body_hamiltonian,
    setup_two_body_hamiltonian,
    construct_one_body_density_matrix,
)


class TimeDependentConfigurationInteraction(metaclass=abc.ABCMeta):
    """Abstract base class defining the skeleton of a time-dependent
    configuration interaction solver. All subclasses need to provide a subclass
    of ConfigurationInteraction in the member `self.ci_class`.

    Parameters
    ----------
    system : QuantumSystems
        Quantum systems instance.
    np : module
        Array library, defaults to ``numpy``.
    integrator : Integrator
        Differential equation integrator. The integrator class must implement a
        ``step``-function.
    verbose : bool
        Print timer and logging info. Default value is ``False``.
    **ci_kwargs : dict
        Keyword arguments to ground state solver class.
    """

    def __init__(
        self, system, np=None, integrator=None, verbose=False, **ci_kwargs
    ):
        if np is None:
            import numpy as np

        self.np = np

        ci_kwargs["verbose"] = verbose
        self.verbose = verbose

        if not "np" in ci_kwargs:
            ci_kwargs["np"] = self.np

        # Initialize ground state solver
        self.ci = self.ci_class(system, **ci_kwargs)

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
        self.spin_reduce_states = self.ci.spin_reduce_states

    def compute_ground_state(self, *args, **kwargs):
        r"""Compute the ground state from the defined ``self.ci_class`` and set
        initial values for the Hamiltonian matrix :math:`\mathbf{H}`.

        Parameters
        ----------
        *args
            Argument list for ground state class ``self.ci_class``.
        **kwargs
            Keyward argument for ground state class ``self.ci_class``.
        """

        # Compute ground state
        self.ci.compute_ground_state(*args, **kwargs)
        # Fetch pointers to the Hamiltonian
        self.one_body_hamiltonian = self.ci.one_body_hamiltonian
        self.two_body_hamiltonian = self.ci.two_body_hamiltonian
        self.hamiltonian = self.ci.hamiltonian.copy()

    def set_initial_conditions(self, c=None, K=0):
        r"""Set initial state for the differential equation solver.

        Parameters
        ----------
        c : np.array
            The initial coefficient vector :math:`\mathbf{c}(0)`. Default is
            ``None`` which defaults to state ``K`` from the ground state
            calculations.
        K : int
            Initial eigenstate from ground state calculations. This argument is
            ignored if a value for ``c`` is given. Default is ``K = 0`` which
            is the ground state.
        """

        if c is None:
            # Create copy of the ground state coefficients
            c = self.ci._C[:, K].copy()

        self._c_0 = c.copy()
        self._c = c

    @property
    def c(self):
        return self._c

    def solve(self, time_points, timestep_tol=1e-8):
        """Function creating a generator for stepping through the solution to
        the differential equation integrator.

        Parameters
        ----------
        time_points : np.array
            Discretized time-points to integrate over.
        timestep_tol : float
            Tolerance to check if the last timestep corresponds to the same
            timestep as used in the evaluation of the right-hand side in the
            integrator. Default is ``timestep_tol=1e-8``.

        Yields
        ------
        np.array
            The coefficient vector at each timestep.
        """

        n = len(time_points)

        for i in range(n - 1):
            dt = time_points[i + 1] - time_points[i]
            c_t = self.integrator.step(self._c, time_points[i], dt)
            self._c = c_t

            if abs(self.last_timestep - (time_points[i] + dt)) > timestep_tol:
                self.update_hamiltonian(time_points[i] + dt)
                self.last_timestep = time_points[i] + dt

            yield self._c

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
        construct_one_body_density_matrix(rho_qp, self.ci.states, self._c)

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
        rho = compute_particle_density(rho_qp, self.system.spf, np=self.np)

        return rho

    def compute_time_dependent_overlap(self):
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

        Returns
        -------
        float
            The real part of the autocorrelation absolute squared.
        """

        norm_t = self._c.conj() @ self._c
        norm_0 = self._c_0.conj() @ self._c_0

        overlap = self.np.abs(self._c.conj() @ self._c_0) ** 2

        return (overlap / norm_t / norm_0).real

    def update_hamiltonian(self, current_time):
        self.h = self.system.h_t(current_time)
        self.u = self.system.u_t(current_time)

        if self.system.has_one_body_time_evolution_operator:
            self.one_body_hamiltonian.fill(0)

            # Compute new one-body Hamiltonian
            setup_one_body_hamiltonian(
                self.one_body_hamiltonian,
                self.ci.states,
                self.h,
                self.n,
                self.l,
            )

        if self.system.has_two_body_time_evolution_operator:
            self.two_body_hamiltonian.fill(0)

            # Compute new two-body Hamiltonian
            setup_two_body_hamiltonian(
                self.two_body_hamiltonian,
                self.ci.states,
                self.u,
                self.n,
                self.l,
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

        o, v = self.system.o, self.system.v

        # Update Hamiltonian matrix
        self.update_hamiltonian(current_time)

        # Compute dot-product of new Hamiltonian with the previous coefficient
        # vector and multiply with -1j.
        new_c = -1j * self.np.dot(self.hamiltonian, prev_c)

        # Note the last timestep used to avoid updating the Hamiltonian in the
        # same timestep twice.
        self.last_timestep = current_time

        return new_c


def get_tdci_class(excitations):
    """Function constructing a truncated TDCI-class with the specified
    excitations.

    Parameters
    ----------
    excitations : str
        The specified excitations to use in the TDCI-class. For example, to
        create a TDCISD class both ``excitations="CISD"`` and
        ``excitations=["S", "D"]`` are valid.

    Returns
    -------
    TimeDependentConfigurationInteraction
        A subclass of ``TimeDependentConfigurationInteraction``.
    """
    ci_class = get_ci_class(excitations)
    excitations = excitation_string_handler(excitations)

    class_name = "TDCI" + "".join(excitations)

    tdci_class = type(
        class_name,
        (TimeDependentConfigurationInteraction,),
        dict(ci_class=ci_class),
    )

    return tdci_class


TDCIS = get_tdci_class("CIS")
TDCID = get_tdci_class("CID")
TDCISD = get_tdci_class("CISD")
TDCIDT = get_tdci_class("CIDT")
TDCISDT = get_tdci_class("CISDT")
TDCIDTQ = get_tdci_class("CIDTQ")
TDCISDTQ = get_tdci_class("CISDTQ")
