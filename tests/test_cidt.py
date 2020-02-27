import numpy as np

from configuration_interaction import CIDT
from configuration_interaction.ci_helper import (
    count_num_states,
    state_printer,
    create_excited_states,
    create_reference_state,
)
from tests.helper import create_doubles_states, create_triples_states


def test_setup(odho_ti):
    cidt = CIDT(odho_ti, verbose=True)

    num_determinants = 1
    num_determinants += count_num_states(odho_ti.n, odho_ti.m, order=2)
    num_determinants += count_num_states(odho_ti.n, odho_ti.m, order=3)
    assert cidt.num_states == num_determinants
    assert len(cidt.states) == cidt.num_states

    counter = 0
    for i in range(len(cidt.states)):
        if cidt.states[i, 0] > 0:
            counter += 1

    assert counter == cidt.num_states


def test_states_setup(odho_ti):
    cidt = CIDT(odho_ti, verbose=True)

    n, l = cidt.n, cidt.l
    states_c = np.zeros_like(cidt.states)
    create_reference_state(n, l, states_c)
    index = create_doubles_states(n, l, states_c, index=1)
    index = create_triples_states(n, l, states_c, index=index)

    for cidt_state, state in zip(
        np.sort(cidt.states, axis=0), np.sort(states_c, axis=0)
    ):
        print(f"{state_printer(cidt_state)}\n{state_printer(state)}\n")

    np.testing.assert_allclose(
        np.sort(cidt.states, axis=0), np.sort(states_c, axis=0)
    )
