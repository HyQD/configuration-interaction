import numpy as np

from configuration_interaction import CID
from configuration_interaction.ci_helper import (
    state_printer,
    create_excited_states,
    create_reference_state,
    compute_particle_density,
)
from tests.helper import (
    create_doubles_states,
    setup_hamiltonian_brute_force,
    construct_one_body_density_matrix_brute_force,
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


def test_states_setup(odho_ti_small):
    cid = CID(odho_ti_small, verbose=True)

    n, l = cid.n, cid.l
    states_c = cid.states.copy()
    create_reference_state(n, l, states_c)
    create_doubles_states(n, l, states_c, index=1)
    states_c = np.sort(states_c, axis=0)

    cid.setup_ci_space()
    for cid_state, state in zip(cid.states, states_c):
        print(f"{state_printer(cid_state)}\n{state_printer(state)}\n")

    np.testing.assert_allclose(cid.states, states_c)


def test_slater_condon_hamiltonian(odho_ti_small):
    cid = CID(odho_ti_small, verbose=True)
    cid.setup_ci_space()
    cid.compute_ground_state()

    hamiltonian_b = np.zeros_like(cid.hamiltonian)
    setup_hamiltonian_brute_force(
        hamiltonian_b,
        cid.states,
        odho_ti_small.h,
        odho_ti_small.u,
        odho_ti_small.n,
        odho_ti_small.l,
    )

    np.testing.assert_allclose(hamiltonian_b, cid.hamiltonian, atol=1e-7)


def test_slater_condon_density_matrix(odho_ti_small):
    cid = CID(odho_ti_small, verbose=True)
    cid.setup_ci_space()
    cid.compute_ground_state()

    for K in range(cid.num_states):
        # Compare particle densities in order to implicitly compare one-body
        # density matrices.
        rho = cid.compute_particle_density(K=K)
        rho_qp_b = np.zeros(
            (odho_ti_small.l, odho_ti_small.l), dtype=np.complex128
        )
        construct_one_body_density_matrix_brute_force(
            rho_qp_b, cid.states, cid.C[:, K]
        )
        rho_b = compute_particle_density(rho_qp_b, odho_ti_small.spf, np)

        # Normalize particle densities
        rho_b = cid.n * rho_b / np.trapz(rho_b, x=odho_ti_small.grid)
        rho = cid.n * rho / np.trapz(rho, x=odho_ti_small.grid)

        np.testing.assert_allclose(rho_b, rho)
