import numpy as np

from configuration_interaction import CIS
from configuration_interaction.ci_helper import (
    state_printer,
    create_excited_states,
    create_reference_state,
    compute_particle_density,
)
from tests.helper import (
    create_singles_states,
    setup_hamiltonian_brute_force,
    construct_one_body_density_matrix_brute_force,
)


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


def test_states_setup(odho_ti_small):
    cis = CIS(odho_ti_small, verbose=True)

    n, l = cis.n, cis.l
    states_c = cis.states.copy()
    create_reference_state(n, l, states_c)
    create_singles_states(n, l, states_c, index=1)
    states_c = np.sort(states_c, axis=0)

    cis.setup_ci_space()
    for cis_state, state in zip(cis.states, states_c):
        print(f"{state_printer(cis_state)}\n{state_printer(state)}\n")

    np.testing.assert_allclose(cis.states, states_c)


def test_slater_condon_hamiltonian(odho_ti_small):
    cis = CIS(odho_ti_small, verbose=True)
    cis.setup_ci_space()
    cis.compute_ground_state()

    hamiltonian_b = np.zeros_like(cis.hamiltonian)
    setup_hamiltonian_brute_force(
        hamiltonian_b,
        cis.states,
        odho_ti_small.h,
        odho_ti_small.u,
        odho_ti_small.n,
        odho_ti_small.l,
    )

    np.testing.assert_allclose(hamiltonian_b, cis.hamiltonian, atol=1e-7)


def test_slater_condon_density_matrix(odho_ti_small):
    cis = CIS(odho_ti_small, verbose=True)
    cis.setup_ci_space()
    cis.compute_ground_state()

    for K in range(cis.num_states):
        # Compare particle densities in order to implicitly compare one-body
        # density matrices.
        rho = cis.compute_particle_density(K=K)

        rho_qp_b = np.zeros(
            (odho_ti_small.l, odho_ti_small.l), dtype=np.complex128
        )
        construct_one_body_density_matrix_brute_force(
            rho_qp_b, cis.states, cis.C[:, K]
        )
        rho_b = compute_particle_density(rho_qp_b, odho_ti_small.spf, np)

        # Normalize particle densities
        rho_b = cis.n * rho_b / np.trapz(rho_b, x=odho_ti_small.grid)
        rho = cis.n * rho / np.trapz(rho, x=odho_ti_small.grid)

        np.testing.assert_allclose(rho_b, rho)
