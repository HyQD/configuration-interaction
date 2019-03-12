from configuration_interaction import CIS


def test_setup(odho_ti_small):
    cis = CIS(odho_ti_small, verbose=True)

    cis.setup_ci_space()

    counter = 0
    for i in range(len(cis.states)):
        if cis.states[i, 0] > 0:
            counter += 1

        for elem in reversed(range(len(cis.states[i]))):
            print(bin(cis.states[i, elem])[2:].zfill(32), end=" ")
        print()

    print(counter, len(cis.states))

    assert False
