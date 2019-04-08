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


@pytest.mark.skip
def test_slater_condon_density_matrix(odho_ti_small):
    cid_b = CID(odho_ti_small, brute_force=True, verbose=True)
    cid_b.setup_ci_space()

    cid = CID(odho_ti_small, verbose=True)
    cid.setup_ci_space()

    np.testing.assert_allclose(cid_b.states, cid.states)

    cid_b.compute_ground_state()
    cid.compute_ground_state()

    for K in range(len(cid.energies)):
        print(f"K = {K}")
        rho_b = cid_b.compute_one_body_density_matrix(K=K)
        rho = cid.compute_one_body_density_matrix(K=K)

        for i in np.ndindex(rho.shape):
            if not abs(rho_b[i] - rho[i]) < 1e-8:
                I, J = i
                print(f"rho_b[{i}] = {rho_b[i]}\t|\trho[{i}] = {rho[i]}")
                np.testing.assert_allclose(cid_b.states[I], cid.states[I])
                print(f"State I = {state_printer(cid_b.states[I])}")
                np.testing.assert_allclose(cid_b.states[J], cid.states[J])
                print(f"State J = {state_printer(cid_b.states[J])}")
                print(
                    f"Diff    = {state_printer(cid_b.states[I] ^ cid_b.states[J])}"
                )
                print("Diff =", state_diff(cid_b.states[I], cid_b.states[J]))

        np.testing.assert_allclose(rho_b, rho, atol=1e-7)
