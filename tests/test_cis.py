from configuration_interaction import CIS
from configuration_interaction.ci_helper import BITSTRING_SIZE


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
    cis.setup_ci_space()

    print(counter, len(cis.states))

    assert False
