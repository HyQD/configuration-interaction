import pytest
import numpy as np

from configuration_interaction import CISD
from configuration_interaction.ci_helper import BITSTRING_SIZE


def test_setup(odho_ti_small):
    cisd = CISD(odho_ti_small, verbose=True)

    cisd.setup_ci_space()
    assert cisd.num_states == 66
    assert len(cisd.states) == cisd.num_states

    counter = 0
    for i in range(len(cisd.states)):
        if cisd.states[i, 0] > 0:
            counter += 1

    assert counter == cisd.num_states


def test_slater_condon_hamiltonian(odho_ti_small):
    cisd_b = CISD(odho_ti_small, brute_force=True, verbose=True)
    cisd_b.setup_ci_space()

    cisd = CISD(odho_ti_small, verbose=True)
    cisd.setup_ci_space()

    cisd_b.compute_ground_state()
    cisd.compute_ground_state()

    np.testing.assert_allclose(cisd_b.hamiltonian, cisd.hamiltonian, atol=1e-7)
    np.testing.assert_allclose(cisd_b.energies, cisd.energies)


def test_slater_condon_density_matrix(odho_ti_small):
    cisd_b = CISD(odho_ti_small, brute_force=True, verbose=True)
    cisd_b.setup_ci_space()

    cisd = CISD(odho_ti_small, verbose=True)
    cisd.setup_ci_space()

    np.testing.assert_allclose(cisd_b.states, cisd.states)

    cisd_b.compute_ground_state()
    cisd.compute_ground_state()

    for K in range(cisd_b.num_states):
        # Compare particle densities in order to implicitly compare one-body
        # density matrices.
        rho_b = cisd_b.compute_particle_density(K=K)
        rho = cisd.compute_particle_density(K=K)

        # Normalize particle densities
        rho_b = cisd_b.n * rho_b / np.trapz(rho_b, x=odho_ti_small.grid)
        rho = cisd.n * rho / np.trapz(rho, x=odho_ti_small.grid)

        np.testing.assert_allclose(rho_b, rho)
