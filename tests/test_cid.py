from configuration_interaction import CID
from configuration_interaction.ci_helper import BITSTRING_SIZE


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

    np.testing.assert_allclose(cid_b.hamiltonian, cid.hamiltonian)
    np.testing.assert_allclose(cid_b.energies, cid.energies)
    np.testing.assert_allclose(cid_b.C, cid.C)
