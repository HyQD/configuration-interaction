from configuration_interaction import CID


def test_setup(odho_ti_small):
    cid = CID(odho_ti_small, verbose=True)
    print(cid.states)
    print(cid.states.shape)

    cid.setup_ci_space()

    assert False
