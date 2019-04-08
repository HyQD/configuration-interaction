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


@pytest.mark.skip
def test_slater_condon_density_matrix(odho_ti_small):
    cis_b = CIS(odho_ti_small, brute_force=True, verbose=True)
    cis_b.setup_ci_space()

    cis = CIS(odho_ti_small, verbose=True)
    cis.setup_ci_space()

    np.testing.assert_allclose(cis_b.states, cis.states)

    cis_b.compute_ground_state()
    cis.compute_ground_state()

    for K in range(len(cis.energies)):
        print(f"K = {K}")
        rho_b = cis_b.compute_one_body_density_matrix(K=K)
        rho = cis.compute_one_body_density_matrix(K=K)

        for i in np.ndindex(rho.shape):
            if not abs(rho_b[i] - rho[i]) < 1e-8:
                print(f"rho_b[{i}] = {rho_b[i]}\t|\trho[{i}] = {rho[i]}")

        np.testing.assert_allclose(rho_b, rho, atol=1e-7)
