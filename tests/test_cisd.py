from configuration_interaction import CISD
from configuration_interaction.ci_helper import BITSTRING_SIZE


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

