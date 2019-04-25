import numpy as np

from configuration_interaction import CISDT
from configuration_interaction.ci_helper import (
    BITSTRING_SIZE,
    NUM_SINGLES_STATES,
    NUM_DOUBLES_STATES,
    NUM_TRIPLES_STATES,
    state_printer,
    state_diff,
    create_singles_states,
    create_doubles_states,
    create_triples_states,
    create_excited_states,
    create_reference_state,
)


def test_setup(odho_ti):
    cisdt = CISDT(odho_ti, verbose=True)

    cisdt.setup_ci_space()

    num_states = 1
    num_states += NUM_SINGLES_STATES(odho_ti.n, odho_ti.m)
    num_states += NUM_DOUBLES_STATES(odho_ti.n, odho_ti.m)
    num_states += NUM_TRIPLES_STATES(odho_ti.n, odho_ti.m)
    assert cisdt.num_states == num_states
    assert len(cisdt.states) == cisdt.num_states

    counter = 0
    for i in range(len(cisdt.states)):
        if cisdt.states[i, 0] > 0:
            counter += 1

    assert counter == cisdt.num_states


def test_states_setup(odho_ti):
    cisdt = CISDT(odho_ti, verbose=True)

    n, l = cisdt.n, cisdt.l
    states_c = cisdt.states.copy()
    create_reference_state(n, l, states_c)
    index = create_singles_states(n, l, states_c, index=1)
    index = create_doubles_states(n, l, states_c, index=index)
    index = create_triples_states(n, l, states_c, index=index)

    cisdt.setup_ci_space()
    for cisdt_state, state in zip(
        np.sort(cisdt.states, axis=0), np.sort(states_c, axis=0)
    ):
        print(f"{state_printer(cisdt_state)}\n{state_printer(state)}\n")

    np.testing.assert_allclose(
        np.sort(cisdt.states, axis=0), np.sort(states_c, axis=0)
    )
