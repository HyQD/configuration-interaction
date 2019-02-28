from configuration_interaction import CID


def test_setup(odho_ti_small):
    cid = CID(odho_ti_small, verbose=True)

    cid.setup_ci_space()

    counter = 0
    for i in range(len(cid.states)):
        if cid.states[i, 0] > 0:
            counter += 1
        print(
            bin(cid.states[i, 1])[2:].zfill(odho_ti_small.l - 32)
            + " "
            + bin(cid.states[i, 0])[2:].zfill(32)
        )

    print(counter, len(cid.states))
    assert False
