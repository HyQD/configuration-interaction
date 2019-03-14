from configuration_interaction import CID
from configuration_interaction.ci_helper import BITSTRING_SIZE


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

    print(counter, len(cid.states))
    assert False
