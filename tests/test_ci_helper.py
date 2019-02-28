import numpy as np

from configuration_interaction.ci_helper import popcount_32


def test_popcount_32():
    zero = np.uint32(0b0)
    assert popcount_32(zero) == 0

    full = np.uint32(~0b0)
    assert popcount_32(full) == 32

    num = 0
    prev_num = 0
    set_bits = 0

    for i in np.random.randint(32, size=10000, dtype=np.uint32):
        num |= 1 << i

        if num != prev_num:
            set_bits += 1

        assert popcount_32(num) == set_bits

        prev_num = num
