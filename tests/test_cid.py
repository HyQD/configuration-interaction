import pytest
import numpy as np

from configuration_interaction import CID
from configuration_interaction.ci_helper import (
    BITSTRING_SIZE,
    state_printer,
    state_diff,
)


def test_setup(odho_ti_small):
    cid = CID(odho_ti_small, verbose=True)

    cid.setup_ci_space()

    assert cid.num_states == 46
    assert len(cid.states) == cid.num_states

    counter = 0
    for i in range(len(cid.states)):
        if cid.states[i, 0] > 0:
            counter += 1

    assert counter == cid.num_states


def test_slater_condon_hamiltonian(odho_ti_small):
    cid_b = CID(odho_ti_small, brute_force=True, verbose=True)
    cid_b.setup_ci_space()

    cid = CID(odho_ti_small, verbose=True)
    cid.setup_ci_space()

    cid_b.compute_ground_state()
    cid.compute_ground_state()

    np.testing.assert_allclose(cid_b.hamiltonian, cid.hamiltonian, atol=1e-7)
    np.testing.assert_allclose(cid_b.energies, cid.energies)


def test_slater_condon_density_matrix(odho_ti_small):
    cid_b = CID(odho_ti_small, brute_force=True, verbose=True)
    cid_b.setup_ci_space()

    cid = CID(odho_ti_small, verbose=True)
    cid.setup_ci_space()

    np.testing.assert_allclose(cid_b.states, cid.states)

    cid_b.compute_ground_state()
    cid.compute_ground_state()

    for K in range(cid_b.num_states):
        # Compare particle densities in order to implicitly compare one-body
        # density matrices.
        rho_b = cid_b.compute_particle_density(K=K)
        rho = cid.compute_particle_density(K=K)

        # Normalize particle densities
        rho_b = cid_b.n * rho_b / np.trapz(rho_b, x=odho_ti_small.grid)
        rho = cid.n * rho / np.trapz(rho, x=odho_ti_small.grid)

        np.testing.assert_allclose(rho_b, rho)
