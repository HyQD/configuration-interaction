import pytest
import numpy as np

from configuration_interaction import CIS
from configuration_interaction.ci_helper import BITSTRING_SIZE, state_printer


def test_setup(odho_ti_small):
    cis = CIS(odho_ti_small, verbose=True)

    cis.setup_ci_space()

    assert cis.num_states == 21
    assert len(cis.states) == cis.num_states

    counter = 0
    for i in range(len(cis.states)):
        if cis.states[i, 0] > 0:
            counter += 1

    assert counter == cis.num_states


def test_slater_condon_hamiltonian(odho_ti_small):
    cis_b = CIS(odho_ti_small, brute_force=True, verbose=True)
    cis_b.setup_ci_space()

    cis = CIS(odho_ti_small, verbose=True)
    cis.setup_ci_space()

    cis_b.compute_ground_state()
    cis.compute_ground_state()

    np.testing.assert_allclose(cis_b.hamiltonian, cis.hamiltonian, atol=1e-7)
    np.testing.assert_allclose(cis_b.energies, cis.energies)


def test_slater_condon_density_matrix(odho_ti_small):
    cis_b = CIS(odho_ti_small, brute_force=True, verbose=True)
    cis_b.setup_ci_space()

    cis = CIS(odho_ti_small, verbose=True)
    cis.setup_ci_space()

    np.testing.assert_allclose(cis_b.states, cis.states)

    cis_b.compute_ground_state()
    cis.compute_ground_state()

    for K in range(cis_b.num_states):
        # Compare particle densities in order to implicitly compare one-body
        # density matrices.
        rho_b = cis_b.compute_particle_density(K=K)
        rho = cis.compute_particle_density(K=K)

        # Normalize particle densities
        rho_b = cis_b.n * rho_b / np.trapz(rho_b, x=odho_ti_small.grid)
        rho = cis.n * rho / np.trapz(rho, x=odho_ti_small.grid)

        np.testing.assert_allclose(rho_b, rho)
