from configuration_interaction import CID
from configuration_interaction.ci_helper import BITSTRING_SIZE


def test_setup(odho_ti_small):
    cid = CID(odho_ti_small, verbose=True)

    cid.setup_ci_space()

    counter = 0
    for i in range(len(cid.states)):
        if cid.states[i, 0] > 0:
            counter += 1

        for elem in reversed(range(len(cid.states[i]))):
            print(bin(cid.states[i, elem])[2:].zfill(BITSTRING_SIZE), end=" ")
        print()

    print(counter, len(cid.states))
    assert False
