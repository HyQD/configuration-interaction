from configuration_interaction import CISD
from configuration_interaction.ci_helper import BITSTRING_SIZE


def test_setup(odho_ti_small):
    cisd = CISD(odho_ti_small, verbose=True)

    cisd.setup_ci_space()

    counter = 0
    for i in range(len(cisd.states)):
        if cisd.states[i, 0] > 0:
            counter += 1

        for elem in reversed(range(len(cisd.states[i]))):
            print(bin(cisd.states[i, elem])[2:].zfill(BITSTRING_SIZE), end=" ")
        print()

    print(counter, len(cisd.states))
    assert False
