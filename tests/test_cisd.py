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
