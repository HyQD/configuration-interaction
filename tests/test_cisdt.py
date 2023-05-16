import numpy as np

from configuration_interaction import CISDT
from configuration_interaction.ci_helper import (
    count_num_states,
    bit_state_printer,
    create_excited_states,
    create_reference_state,
)
from tests.helper import (
    create_singles_states,
    create_doubles_states,
    create_triples_states,
)


def test_setup(odho_ti):
    cisdt = CISDT(odho_ti, verbose=True)

    num_determinants = 1
    num_determinants += count_num_states(odho_ti.n, odho_ti.m, order=1)
    num_determinants += count_num_states(odho_ti.n, odho_ti.m, order=2)
    num_determinants += count_num_states(odho_ti.n, odho_ti.m, order=3)
    assert cisdt.num_states == num_determinants
    assert len(cisdt.states) == cisdt.num_states

    counter = 0
    for i in range(len(cisdt.states)):
        if cisdt.states[i, 0] > 0:
            counter += 1

    assert counter == cisdt.num_states


def test_states_setup(odho_ti):
    cisdt = CISDT(odho_ti, verbose=True)

    n, l = cisdt.n, cisdt.l
    states_c = np.zeros_like(cisdt.states)
    create_reference_state(n, l, states_c)
    index = create_singles_states(n, l, states_c, index=1)
    index = create_doubles_states(n, l, states_c, index=index)
    index = create_triples_states(n, l, states_c, index=index)

    for cisdt_state, state in zip(
        np.sort(cisdt.states, axis=0), np.sort(states_c, axis=0)
    ):
        print(f"{bit_state_printer(cisdt_state)}\n{bit_state_printer(state)}\n")

    np.testing.assert_allclose(
        np.sort(cisdt.states, axis=0), np.sort(states_c, axis=0)
    )
