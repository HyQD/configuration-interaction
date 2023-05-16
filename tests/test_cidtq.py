import numpy as np

from configuration_interaction import CIDTQ
from configuration_interaction.ci_helper import (
    count_num_states,
    bit_state_printer,
    create_excited_states,
    create_reference_state,
)
from tests.helper import (
    create_doubles_states,
    create_triples_states,
    create_quadruples_states,
)


def test_setup(odho_ti):
    cidtq = CIDTQ(odho_ti, verbose=True)

    num_determinants = 1
    num_determinants += count_num_states(odho_ti.n, odho_ti.m, order=2)
    num_determinants += count_num_states(odho_ti.n, odho_ti.m, order=3)
    num_determinants += count_num_states(odho_ti.n, odho_ti.m, order=4)
    assert cidtq.num_states == num_determinants
    assert len(cidtq.states) == cidtq.num_states

    counter = 0
    for i in range(len(cidtq.states)):
        if cidtq.states[i, 0] > 0:
            counter += 1

    assert counter == cidtq.num_states


def test_states_setup(odho_ti):
    cidtq = CIDTQ(odho_ti, verbose=True)

    n, l = cidtq.n, cidtq.l
    states_c = np.zeros_like(cidtq.states)
    create_reference_state(n, l, states_c)
    index = create_doubles_states(n, l, states_c, index=1)
    index = create_triples_states(n, l, states_c, index=index)
    index = create_quadruples_states(n, l, states_c, index=index)

    for cidtq_state, state in zip(
        np.sort(cidtq.states, axis=0), np.sort(states_c, axis=0)
    ):
        print(f"{bit_state_printer(cidtq_state)}\n{bit_state_printer(state)}\n")

    np.testing.assert_allclose(
        np.sort(cidtq.states, axis=0), np.sort(states_c, axis=0)
    )
