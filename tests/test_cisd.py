import numpy as np

from configuration_interaction import CISD
from configuration_interaction.ci_helper import (
    state_printer,
    create_excited_states,
    create_reference_state,
    compute_particle_density,
)
from tests.helper import (
    create_singles_states,
    create_doubles_states,
    construct_one_body_density_matrix_brute_force,
    setup_hamiltonian_brute_force,
)


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


def test_states_setup(odho_ti_small):
    cisd = CISD(odho_ti_small, verbose=True)

    n, l = cisd.n, cisd.l
    states_c = cisd.states.copy()
    create_reference_state(n, l, states_c)
    index = create_singles_states(n, l, states_c, index=1)
    create_doubles_states(n, l, states_c, index=index)
    states_c = np.sort(states_c, axis=0)

    cisd.setup_ci_space()
    for cisd_state, state in zip(cisd.states, states_c):
        print(f"{state_printer(cisd_state)}\n{state_printer(state)}\n")

    np.testing.assert_allclose(cisd.states, states_c)


def test_slater_condon_hamiltonian(odho_ti_small):
    cisd = CISD(odho_ti_small, verbose=True)
    cisd.setup_ci_space()
    cisd.compute_ground_state()

    hamiltonian_b = np.zeros_like(cisd.hamiltonian)
    setup_hamiltonian_brute_force(
        hamiltonian_b,
        cisd.states,
        odho_ti_small.h,
        odho_ti_small.u,
        odho_ti_small.n,
        odho_ti_small.l,
    )

    np.testing.assert_allclose(hamiltonian_b, cisd.hamiltonian, atol=1e-7)


def test_slater_condon_density_matrix(odho_ti_small):
    cisd = CISD(odho_ti_small, verbose=True)
    cisd.setup_ci_space()
    cisd.compute_ground_state()

    for K in range(cisd.num_states):
        # Compare particle densities in order to implicitly compare one-body
        # density matrices.
        rho = cisd.compute_particle_density(K=K)
        rho_qp_b = np.zeros(
            (odho_ti_small.l, odho_ti_small.l), dtype=np.complex128
        )
        construct_one_body_density_matrix_brute_force(
            rho_qp_b, cisd.states, cisd.C[:, K]
        )
        rho_b = compute_particle_density(rho_qp_b, odho_ti_small.spf, np)

        # Normalize particle densities
        rho_b = cisd.n * rho_b / np.trapz(rho_b, x=odho_ti_small.grid)
        rho = cisd.n * rho / np.trapz(rho, x=odho_ti_small.grid)

        np.testing.assert_allclose(rho_b, rho)
