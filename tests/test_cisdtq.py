import numpy as np

from configuration_interaction import CISDTQ
from configuration_interaction.ci_helper import (
    NUM_SINGLES_STATES,
    NUM_DOUBLES_STATES,
    NUM_TRIPLES_STATES,
    NUM_QUADRUPLES_STATES,
    state_printer,
    create_excited_states,
    create_reference_state,
)
from tests.helper import (
    create_singles_states,
    create_doubles_states,
    create_triples_states,
    create_quadruples_states,
)


def test_setup(odho_ti):
    cisdtq = CISDTQ(odho_ti, verbose=True)

    num_states = 1
    num_states += NUM_SINGLES_STATES(odho_ti.n, odho_ti.m)
    num_states += NUM_DOUBLES_STATES(odho_ti.n, odho_ti.m)
    num_states += NUM_TRIPLES_STATES(odho_ti.n, odho_ti.m)
    num_states += NUM_QUADRUPLES_STATES(odho_ti.n, odho_ti.m)
    assert cisdtq.num_states == num_states
    assert len(cisdtq.states) == cisdtq.num_states

    counter = 0
    for i in range(len(cisdtq.states)):
        if cisdtq.states[i, 0] > 0:
            counter += 1

    assert counter == cisdtq.num_states


def test_states_setup(odho_ti):
    cisdtq = CISDTQ(odho_ti, verbose=True)

    n, l = cisdtq.n, cisdtq.l
    states_c = np.zeros_like(cisdtq.states)
    create_reference_state(n, l, states_c)
    index = create_singles_states(n, l, states_c, index=1)
    index = create_doubles_states(n, l, states_c, index=index)
    index = create_triples_states(n, l, states_c, index=index)
    index = create_quadruples_states(n, l, states_c, index=index)

    for cisdtq_state, state in zip(
        np.sort(cisdtq.states, axis=0), np.sort(states_c, axis=0)
    ):
        print(f"{state_printer(cisdtq_state)}\n{state_printer(state)}\n")

    np.testing.assert_allclose(
        np.sort(cisdtq.states, axis=0), np.sort(states_c, axis=0)
    )
