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


@pytest.mark.skip
def test_slater_condon_density_matrix(odho_ti_small):
    cisd_b = CISD(odho_ti_small, brute_force=True, verbose=True)
    cisd_b.setup_ci_space()

    cisd = CISD(odho_ti_small, verbose=True)
    cisd.setup_ci_space()

    np.testing.assert_allclose(cisd_b.states, cisd.states)

    cisd_b.compute_ground_state()
    cisd.compute_ground_state()

    # Only check ground state as higher order states can be degenerate.
    # This leads to ambiguity as to which state to compare.
    K = 0

    rho_b = cisd_b.compute_one_body_density_matrix(K=K)
    rho = cisd.compute_one_body_density_matrix(K=K)

    for i in np.ndindex(rho.shape):
        if not abs(rho_b[i] - rho[i]) < 1e-8:
            print(f"rho_b[{i}] = {rho_b[i]}\t|\trho[{i}] = {rho[i]}")

    np.testing.assert_allclose(rho_b, rho, atol=1e-7)
