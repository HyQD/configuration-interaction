from configuration_interaction import CIS
from configuration_interaction.ci_helper import BITSTRING_SIZE


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

    np.testing.assert_allclose(cis_b.hamiltonian, cis.hamiltonian)
    np.testing.assert_allclose(cis_b.energies, cis.energies)
    np.testing.assert_allclose(cis_b.C, cis.C)
